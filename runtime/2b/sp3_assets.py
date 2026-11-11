from tp4_profile import PROFILE
LAYERS=28
GROUPS=PROFILE["sequence_parallel"]
OWN=PROFILE["layers_per_weight_group"]
"""Explicit reuse map for verified PP3 coefficients and unchanged TP4 weights.

This creates new layout/ownership metadata, never relabels PP3 evidence as an
SP3 numeric or hardware pass. Remote weights require metadata, not local files.
"""
import hashlib
import json
import time
from pathlib import Path
import os
from release_assets import fixed_task
from types import SimpleNamespace
import numpy as np
from model_runner import View
from c204_runtime import WeightView
from c209_fixed_weights import load_compact,LAYER_NAMES,GLOBAL_NAMES,weight_views
from c209_tensor_layout import PackedKVLayout

def compact_from_layout_proof(original,names):
    """Same 64-byte compaction as load_compact, using bounded JSON only."""
    arrays={};offset=0;limit=original.get('bytes');ranges=[]
    if type(limit)is not int or limit<=0 or limit%64:raise ValueError('Source weight layout extent')
    for name,entry in original.get('arrays',{}).items():
        dtype=np.dtype(entry['dtype']);shape=entry['shape'];begin=entry['offset']
        if dtype.hasobject or any(type(x)is not int or x<=0 for x in shape) or type(begin)is not int or begin<0 or begin%64:raise ValueError('Source weight layout entry')
        size=int(np.prod(shape))*dtype.itemsize
        if begin+size>limit:raise ValueError('Source weight layout overflow')
        ranges.append((begin,begin+size))
    ranges.sort()
    if any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):raise ValueError('Source weight layout overlap')
    for name in names:
        entry=original['arrays'][name];dtype=np.dtype(entry['dtype'])
        if name.endswith('.q') and dtype!=np.dtype('<i2'):raise ValueError('Unchanged W16 layout required')
        offset=(offset+63)//64*64;size=int(np.prod(entry['shape']))*dtype.itemsize
        arrays[name]=dict(entry,offset=offset,source_offset=entry['offset'],bytes=size);offset+=size
    return dict(bytes=(offset+63)//64*64,arrays=arrays,source_bytes=limit)

def validate_reuse_bindings(manifest,original,role,read_layout):
    """Bind semantic positions, not merely membership in a hash file table."""
    rank=str(role.rank);entries=manifest['ranks'][rank];old=original['ranks'][rank]
    for layer in range(LAYERS):
        item=entries['layers'][str(layer)];name=old['layers'][str(layer)]
        if item.get('file')!=name or item.get('sha256')!=original['files'][name] or item.get('owner')!=layer//OWN:raise ValueError('SP3 layer-to-original-shard mapping changed')
        if item.get('layout')!=compact_from_layout_proof(read_layout(name),LAYER_NAMES):raise ValueError('SP3 compact layer layout changed')
    caches={str(g):original['stages'][str(g)]['ranks'][rank] for g in range(GROUPS)}
    if entries.get('caches')!=caches:raise ValueError('SP3 fixed coefficient/cross-KV position mapping changed')
    if manifest.get('global_file')!=original.get('global') or manifest.get('global_layout')!=compact_from_layout_proof(read_layout(original['global']),GLOBAL_NAMES):raise ValueError('SP3 global weight mapping changed')
    if any(manifest.get(k)!=original.get(k) for k in ('rope','schedule','noise_reference')):raise ValueError('SP3 fixed input mapping changed')

def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''):h.update(data)
        if hasattr(os, 'posix_fadvise') and stream.tell() >= 8*1024**2:
            # Avoid pinning cold asset verification reads in CMA before allocation.
            try:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            except OSError:pass
    return h.hexdigest()

def load_fixed_2b_binding(folder):
    """Fresh preparation receipt; the frozen hardware profile never imports 14B hashes."""
    b=json.loads((Path(folder)/'fixed_2b_binding.json').read_text(encoding='utf-8'))
    task=dict(PROFILE['fixed_task'])
    expected=dict(kind='C209_2B_FIXED_ASSET_BINDING_V1',model=PROFILE['model'],hardware_id=PROFILE['hardware_id'],
        hidden=2048,layers=28,heads=16,tensor_parallel=2,sequence_parallel=GROUPS,
        positive=task['positive'],negative=task['negative'],stored_keys=task['stored_keys'],logical_keys=task['logical_keys'])
    if any(b.get(k)!=v for k,v in expected.items()):raise ValueError('Fresh 2B fixed-task binding/geometry required')
    for key in ('asset_manifest_sha256','input_manifest_sha256','source_binding_sha256'):
        v=b.get(key)
        if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):raise ValueError('Missing prepared 2B digest '+key)
    if b.get('asset_updates')!=10 or b.get('preset')!=task['preset']:raise ValueError('2B schedule binding differs')
    task.update({key:b[key] for key in ('asset_manifest_sha256','input_manifest_sha256','source_binding_sha256')})
    return task

def validate_fixed_apple_manifest(raw,original,task=None):
    """Admit the selected immutable asset metadata before touching weight files.

    The input-manifest digest binds the literal apple prompts and full schedule;
    seed remains a separately generated sampler input and is not restricted here.
    """
    from tp4_profile import PROFILE
    task=PROFILE['fixed_task'] if task is None else task
    if not task.get('asset_manifest_sha256') or not task.get('input_manifest_sha256'):
        raise ValueError('2B fixed assets are not prepared/bound; 14B manifests cannot be reused')
    if hashlib.sha256(raw).hexdigest()!=task['asset_manifest_sha256']:
        raise ValueError('Only the selected fixed-apple asset manifest is permitted')
    if (original.get('model')!=PROFILE['model'] or original.get('preset_id')!=task['preset'] or
        original.get('execution',{}).get('updates')!=task['asset_updates'] or task['asset_updates']!=10 or
        original.get('source_binding',{}).get('input_manifest_sha256')!=task['input_manifest_sha256']):
        raise ValueError('Fixed-apple model, preset or complete schedule binding differs')
    for condition,stored in zip(('positive','negative'),task['stored_keys']):
        context=original.get('contexts',{}).get(condition,{})
        if context.get('stored_keys')!=stored or context.get('logical_keys')!=task['logical_keys']:
            raise ValueError('Fixed-apple context extent differs')
    return dict(asset_manifest_sha256=task['asset_manifest_sha256'],input_manifest_sha256=task['input_manifest_sha256'],
                preset=task['preset'],asset_updates=10,prompt=dict(positive=task['positive'],negative=task['negative']),seed_is_variable=True)


class SP3Assets:
    def __init__(self,folder,role,profile_sha256):
        self.folder=Path(folder).resolve();self.role=role
        self.manifest=json.loads((self.folder/'sp3_assets.json').read_text(encoding='utf-8'));m=self.manifest
        from tp4_profile import PROFILE
        accepted_profiles={profile_sha256,PROFILE.get('reusable_asset_profile_sha256')}-{None}
        if m.get('kind')!='C209_TP2_SP12_2B_ASSETS_V1' or m.get('profile_sha256') not in accepted_profiles:raise ValueError('SP3 asset binding')
        self.asset_profile_sha256=m['profile_sha256'];self.execution_profile_sha256=profile_sha256
        original_raw=self.path('tp4_assets.json').read_bytes();original=json.loads(original_raw)
        task=load_fixed_2b_binding(self.folder)
        release=fixed_task(self.folder,PROFILE,original)
        if any(task[name]!=release[name] for name in ('input_manifest_sha256','asset_manifest_sha256','preset','asset_updates')):
            raise ValueError('Selected role assets and package disagree')
        self.fixed_task=validate_fixed_apple_manifest(original_raw,original,task)
        if hashlib.sha256(original_raw).hexdigest()!=m['source_manifest_sha256'] or original.get('files')!=m['files'] or original.get('source_binding_sha256')!=m['source_binding_sha256']:
            raise ValueError('SP3 reuse map no longer matches preserved PP3 manifest')
        if original.get('status')!='VERIFIED_PRECOMPUTE_ACTUAL_WEIGHTS_NOT_FULL_MODEL' or original.get('profile_sha256')!=m['source_profile_sha256']:raise ValueError('PP3 evidence provenance')
        self.updates=m['updates'];entry=m['ranks'][str(role.rank)];self.layers=[];self._verified=set()
        if self.updates!=original.get('execution',{}).get('updates',19) or not 1<=self.updates<=19:raise ValueError('Complete original schedule update count')
        proof=m['source_numerical_validation']
        if proof!=original.get('numerical_validation'):raise ValueError('Source numerical proof binding')
        self.verify(proof['file'])
        receipt=json.loads(self.path(proof['file']).read_text(encoding='utf-8'))
        if receipt.get('status')!='PASS_ALL_PRECOMPUTE_ARRAYS_BIT_EXACT_NOT_FULL_MODEL' or receipt.get('source_binding_sha256')!=m['source_binding_sha256']:raise ValueError('PP3 numerical receipt is not reusable')
        def read_layout(name):
            proof_name=str(Path(name).with_suffix('.layout.json')).replace('\\','/')
            self.verify(proof_name)
            return json.loads(self.path(proof_name).read_text(encoding='utf-8'))
        validate_reuse_bindings(m,original,role,read_layout)
        from integrated_assets import overlay,validate_layer,K_LAYOUT
        self.integrated_assets=overlay(self.folder,m['source_manifest_sha256'],file_digest(self.folder/'sp3_assets.json'))
        packed=self.integrated_assets;new_rank=packed['ranks'][str(role.rank)]
        validate_layer(m['global_layout'],packed['global']['layout'])
        if packed['global']['source_file']!=m['global_file'] or packed['global']['source_sha256']!=m['files'][m['global_file']]:raise ValueError('Packed global source binding')

        for layer in range(LAYERS):
            item=entry['layers'][str(layer)]
            if item['owner']!=layer//OWN:raise ValueError('Weight ownership changed')
            replacement=new_rank['layers'][str(layer)];validate_layer(item['layout'],replacement['layout'])
            if replacement['owner']!=item['owner'] or replacement['source_file']!=item['file'] or replacement['source_sha256']!=item['sha256']:raise ValueError('Packed layer source binding')
            path=self.path(replacement['file'],required=layer in role.owned_layers)
            if layer in role.owned_layers:self.verify(replacement['file'])
            self.layers.append((path,replacement['layout']))
        self.global_path=self.path(packed['global']['file']);self.verify(packed['global']['file']);self.global_layout=packed['global']['layout']
        self.caches=new_rank['caches'];self.cache_entries=[item for g in self.caches.values() for item in g.values()]
        for group,entries in self.caches.items():
            for condition,item in entries.items():
                source=entry['caches'][group][condition]
                if (item.get('source_file')!=source['file'] or item.get('source_sha256')!=source['sha256'] or item.get('k_layout')!=K_LAYOUT
                    or item['arrays']!=source['arrays'] or item['bytes']!=source['bytes']):raise ValueError('Head-major K cache binding')
        for group in self.caches.values():
            for condition,stored in (('positive',16),('negative',11)):
                if group[condition].get('stored_keys')!=stored:raise ValueError('Fixed-apple cached context differs from IP12')
        for entry in self.cache_entries:
            if entry['sha256']!=packed['files'][entry['file']]:raise ValueError('Cache identity')
            self.verify(entry['file'])
        for name in ('rope','schedule','noise_reference'):self.verify(m[name])
        self.cache_views={}
    def path(self,name,required=True):
        path=(self.folder/name).resolve()
        if not path.is_relative_to(self.folder) or (required and not path.is_file()):raise FileNotFoundError('SP3 source asset '+str(name))
        return path
    safe=path
    def verify(self,name):
        if name not in self._verified:
            expected=getattr(self,'integrated_assets',{}).get('files',{}).get(name,self.manifest['files'].get(name))
            if expected is None or file_digest(self.path(name))!=expected:raise ValueError('Changed source bytes '+name)
            self._verified.add(name)
    def load_cache(self,board):
        if self.cache_views:raise RuntimeError('Cached coefficients/KV already loaded; refuse double VS folding')
        from overlap_scale_policy import fold_v_scales
        for group,entries in self.caches.items():
            self.cache_views[int(group)]={}
            for condition,entry in entries.items():
                parent=board.make((entry['bytes'],),'u1')
                with self.path(entry['file']).open('rb') as stream:
                    if stream.readinto(memoryview(parent.array))!=entry['bytes']:raise IOError('Truncated SP3 cache')
                views={name:View(parent,item['offset'],item['shape'],item['dtype']) for name,item in entry['arrays'].items()}
                # Alter only the fresh owned DDR copy, never an asset file.
                # New PV consumes this scale directly, rather than repeating
                # the probability conversion in every output element.
                for name,value in views.items():
                    if name.endswith('.vs'):
                        if value.dtype!=np.dtype('<f4'):raise ValueError('Cached V scales require FP32 values')
                        value.array[:]=fold_v_scales(value.array)
                parent.root.flush()
                self.cache_views[int(group)][condition]=views
        self.vs_policy='PV_PROB_SCALE_FOLDED_V1'
        self._prepare_fixed_coefficients(board)
        return self.cache_views
    def _prepare_fixed_coefficients(self,board):
        """Derive FP32 (1 + modulation) into separate owned tensors.

        The new fixed operator consumes this explicitly different coefficient
        ABI. Each addition rounds once to FP32, at the old PL addition boundary.
        The gamma vector is exactly one and is absent from the new operation.
        """
        if getattr(self,'coefficients_prepared',False):raise RuntimeError('Fixed coefficients already prepared')
        self.fixed_modulation=board.make((self.updates,LAYERS,3,2048),'<f4')
        self.fixed_final=board.make((self.updates,2048),'<f4')
        from overlap_scale_policy import build_scale_record
        self.fixed_scale_records=board.make((self.updates,LAYERS,3,16),'<u4')
        self.final_scale_records=board.make((self.updates,16),'<u4')
        self.fixed_scale_records.array[:]=0;self.final_scale_records.array[:]=0
        for group,views in self.cache_views.items():
            parent=views['common']['modulation']
            for step in range(self.updates):
                for local_layer in range(OWN):
                    for branch in range(3):
                        offset=((step*OWN+local_layer)*3+branch)*6144*4+2048*4
                        scale=View(parent,offset,(2048,),'<f4')
                        if not np.isfinite(scale.array).all():raise ValueError('Nonfinite fixed modulation')
                        target=self.fixed_modulation.array[step,group*OWN+local_layer,branch]
                        np.add(np.float32(1),scale.array,out=target,dtype=np.float32)
                        shift=View(parent,offset-2048*4,(2048,),'<f4')
                        self.fixed_scale_records.array[step,group*OWN+local_layer,branch,:8]=build_scale_record(shift.array,target)
        parent=self.cache_views[0]['common']['final']
        for step in range(self.updates):
            scale=View(parent,(step*4096+2048)*4,(2048,),'<f4')
            if not np.isfinite(scale.array).all():raise ValueError('Nonfinite fixed final modulation')
            np.add(np.float32(1),scale.array,out=self.fixed_final.array[step],dtype=np.float32)
            shift=View(parent,step*4096*4,(2048,),'<f4')
            self.final_scale_records.array[step,:8]=build_scale_record(shift.array,self.fixed_final.array[step])
        self.fixed_modulation.root.flush();self.fixed_final.root.flush()
        self.fixed_scale_records.root.flush();self.final_scale_records.root.flush();self.coefficients_prepared=True
    def modulation(self,step,layer,branch):
        if not getattr(self,'coefficients_prepared',False):raise RuntimeError('New fixed coefficient ABI is not prepared')
        if not 0<=step<self.updates or layer not in range(LAYERS) or branch not in range(3):raise ValueError('Coefficient position')
        parent=self.cache_views[layer//OWN]['common']['modulation'];offset=((step*OWN+layer%OWN)*3+branch)*6144*4
        shift=View(parent,offset,(2048,),'<f4');gate=View(parent,offset+4096*4,(2048,),'<f4')
        gain=View(self.fixed_modulation,((step*LAYERS+layer)*3+branch)*2048*4,(2048,),'<f4')
        return shift,gain,gate
    def final(self,step):
        if not getattr(self,'coefficients_prepared',False):raise RuntimeError('New fixed coefficient ABI is not prepared')
        parent=self.cache_views[0]['common']['final'];offset=step*4096*4
        return View(parent,offset,(2048,),'<f4'),View(self.fixed_final,step*2048*4,(2048,),'<f4')
    def scale_policy(self,step,layer,operator):
        if not getattr(self,'coefficients_prepared',False) or not 0<=step<self.updates or layer not in range(LAYERS) or operator not in range(3):
            raise ValueError('Fixed scale policy position')
        return View(self.fixed_scale_records,((step*LAYERS+layer)*3+operator)*64,(8,),'<u4')
    def final_scale_policy(self,step):
        if not getattr(self,'coefficients_prepared',False) or not 0<=step<self.updates:raise ValueError('Final scale policy position')
        return View(self.final_scale_records,step*64,(8,),'<u4')
    def prepared(self,condition,layer):
        entry=self.caches[str(layer//OWN)][condition];layout=PackedKVLayout(entry['stored_keys'],8)
        if entry['stored_keys']!={'positive':16,'negative':11}[condition]:raise ValueError('Fixed apple cross-KV extent differs from IP12')
        views={part:self.cache_views[layer//OWN][condition][f'layer{layer:02d}.{part}'] for part in ('k','v','ks','vs')}
        if getattr(self,'vs_policy',None)!='PV_PROB_SCALE_FOLDED_V1':raise RuntimeError('Cross KV requires folded VS cache')
        return SimpleNamespace(layout=layout,buffers=views,ready=True,vs_policy=self.vs_policy)


class SharedWeightStore:
    """Two hot slots, owner disk prefetch, exact paired-branch read lifetime.

    Network prefetch follows both joined self-KV transactions; layer0 has one
    shared transaction covering both branches. Remaining computation uses
    core/TP P0/P1. The next acquisition joins this ordered SP P2/P3 future.
    Layer35 prefetches the next requested step's layer0, with no DMA at a gate.
    """
    def __init__(self,board,assets,transport):
        from concurrent.futures import ThreadPoolExecutor
        self.board,self.assets,self.transport=board,assets,transport
        size=max(layout['bytes'] for _,layout in assets.layers)
        self.slots=[board.make((size,),'u1') for _ in range(2)]
        self.executor=ThreadPoolExecutor(max_workers=1);self.network=ThreadPoolExecutor(max_workers=1)
        self.pending={};self.network_pending={};self.network_ready={};self.readers={};self.active=None;self.closed=False
        self.load_wait_records=[]
    def _wait_local_load(self,layer,*,step,reason):
        # Keep source file loading distinct from receiver-side SFP waiting.
        # This is the same one-use future ownership as the original pop/result.
        future=self.pending.pop(layer)
        began=time.monotonic_ns();cpu=time.thread_time_ns();ok=False
        try:
            result=future.result();ok=True;return result
        finally:
            self.load_wait_records.append(dict(step=step,layer=layer,reason=reason,
                start_ns=began,end_ns=time.monotonic_ns(),
                thread_cpu_ns=time.thread_time_ns()-cpu,ok=ok))
    def prefetch(self,layer):
        if layer not in range(LAYERS) or layer in self.pending or layer//OWN!=self.assets.role.group:return
        slot=layer%2
        if slot in self.readers:raise RuntimeError('Cannot prefetch over live paired-CFG weights')
        for old,f in list(self.pending.items()):
            if old%2==slot:f.result();del self.pending[old]
        path,layout=self.assets.layers[layer]
        self.pending[layer]=self.executor.submit(load_compact,self.board,path,layout,self.slots[slot])
    def acquire(self,layer,*,step):
        if self.closed or self.active is not None:raise RuntimeError('Weight lifetime not released')
        slot=layer%2;parent=self.slots[slot];path,layout=self.assets.layers[layer]
        key=(step,layer);prefetched=self.network_pending.pop(key,None)
        ready=self.network_ready.pop(key,None)
        if prefetched is not None:prefetched.result()
        elif ready is not None:
            if ready!=slot:raise RuntimeError('Prefetched weight slot changed')
        else:
            self.prefetch(layer)
            if layer//OWN==self.assets.role.group:self._wait_local_load(layer,step=step,reason='acquire')
            payload=View(parent,0,(layout['bytes'],),'u1')
            self.transport.distribute_weight(payload,layer,step=step)
        parent.root.invalidate()
        self.readers[slot]=(step,layer);self.active=(step,layer,slot);self.used=[];self.kv_done=set();self.network_launched=False
        self.prefetch(layer+1)
        return weight_views(parent,layout)
    def prefetch_network(self,layer,*,step):
        if layer not in range(LAYERS):raise ValueError('Use prefetch_after_self for the next-step boundary')
        if self.active is None or self.active[:2]!=(step,layer-1) or self.kv_done!={'positive','negative'}:raise RuntimeError('Network prefetch requires both drained self-KV transactions')
        self._launch_network(layer,step)
    def self_kv_complete(self,condition,*,step,layer):
        if condition not in ('positive','negative') or self.active is None or self.active[:2]!=(step,layer):
            raise RuntimeError('Self KV completion belongs to another layer')
        if condition in self.kv_done:raise RuntimeError('Duplicate self KV completion')
        self.kv_done.add(condition)
    def prefetch_after_self(self,*,step,layer,execution_steps):
        """Both KV readers have joined; layer0 shares its one completed KV."""
        if type(execution_steps)is not int or not 0<=step<execution_steps or layer not in range(LAYERS):raise ValueError('Prefetch schedule boundary')
        if self.active is None or self.active[:2]!=(step,layer) or self.kv_done!={'positive','negative'} or self.network_launched:
            raise RuntimeError('Prefetch requires both KV completions and a fresh weight transaction')
        self.network_launched=True
        if layer<LAYERS-1:self._launch_network(layer+1,step)
        elif step+1<execution_steps:self._launch_network(0,step+1)
    def _launch_network(self,layer,step):
        key=(step,layer)
        if key in self.network_pending or key in self.network_ready:raise RuntimeError('Duplicate next-layer broadcast')
        slot=layer%2
        if slot in self.readers:raise RuntimeError('Network prefetch target has live readers')
        self.prefetch(layer)
        def fill():
            if layer//OWN==self.assets.role.group:self._wait_local_load(layer,step=step,reason='network_prefetch')
            parent=self.slots[slot];layout=self.assets.layers[layer][1]
            self.transport.distribute_weight(View(parent,0,(layout['bytes'],),'u1'),layer,step=step)
            parent.root.invalidate()
        self.network_pending[key]=self.network.submit(fill)
    def join_prefetch(self):
        """Leave immutable ready bytes, but no live DMA, at an explicit step gate."""
        for key,future in list(self.network_pending.items()):
            future.result();self.network_ready[key]=key[1]%2;del self.network_pending[key]
        for future in list(self.pending.values()):future.result()
    def branch_complete(self,condition):
        expected=('positive','negative')[len(self.used)] if len(self.used)<2 else None
        if self.active is None or condition!=expected:raise RuntimeError('Weights require positive then negative')
        self.used.append(condition)
    def release(self):
        if self.active is None or self.used!=['positive','negative']:raise RuntimeError('Both CFG branches must drain before weight release')
        del self.readers[self.active[2]];self.active=None
    def close(self):
        self.network.shutdown(wait=True)
        self.executor.shutdown(wait=True)
        if self.active is not None:raise RuntimeError('Closing a live weight reader')
        self.closed=True
