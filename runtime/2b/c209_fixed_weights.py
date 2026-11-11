"""Compact immutable hot weight views; removed PC-only weights never enter DDR."""
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import numpy as np
from c204_runtime import aligned,weight_layout,WeightView

GLOBAL_NAMES=('patch.q','patch.s','final.q','final.s')
LAYER_NAMES=tuple(name+suffix for name in (
    'self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
    'self_attn.output_proj.weight','cross_attn.q_proj.weight','cross_attn.output_proj.weight',
    'mlp.layer1.weight','mlp.layer2.weight') for suffix in ('.q','.s'))+(
    'self_attn.q_norm.weight','self_attn.k_norm.weight','cross_attn.q_norm.weight')


def compact_layout(path,names):
    original=weight_layout(path);arrays={};offset=0
    for name in names:
        if name not in original['arrays']:raise ValueError('Missing fixed hot weight '+name)
        entry=original['arrays'][name];dtype=np.dtype(entry['dtype'])
        if name.endswith('.q') and dtype!=np.dtype('<i2'):raise ValueError('Fixed hot weights must be W16')
        offset=aligned(offset);size=int(np.prod(entry['shape']))*dtype.itemsize
        arrays[name]=dict(entry,offset=offset,source_offset=entry['offset'],bytes=size);offset+=size
    return dict(bytes=aligned(offset),arrays=arrays,source_bytes=original['bytes'])


def weight_views(parent,layout):
    views={name:WeightView(parent,entry) for name,entry in layout['arrays'].items()}
    for name,view in views.items():
        entry=layout['arrays'][name]
        if name.endswith('.q'):
            view.packed_b_layout=entry.get('matrix_layout')
            view.logical_shape=tuple(entry.get('logical_shape',entry['shape']))
    return views


def _process_storage_reads():
    try:
        fields=dict(line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())
        return int(fields['read_bytes'])
    except (OSError,ValueError,KeyError):return None


def load_compact(board,path,layout,parent=None):
    # Keep the identical file-to-hot-slot data path and synchronous flush.
    # Buffered diagnostics separate source read/copy from flush and peer wait;
    # process-wide I/O deltas may include another concurrent file reader.
    started=time.monotonic_ns();cpu=time.thread_time_ns();storage_before=_process_storage_reads()
    path=Path(path);raw=path.with_suffix('.bin')
    record=dict(path=str(path),raw_path=str(raw),start_ns=started,thread=threading.get_ident(),
        required_bytes=layout['bytes'],source_read_copy_seconds=0.,flush_seconds=0.,ok=False,
        storage_counter_scope='PROCESS_WIDE_READ_BYTES_DELTA_NOT_EXCLUSIVE_THREAD_IO')
    try:
        if parent is None:parent=board.make((layout['bytes'],),'u1')
        if parent.nbytes<layout['bytes']:raise ValueError('Fixed hot-weight slot too small')
        views=weight_views(parent,layout)
        began=time.monotonic_ns()
        try:
            if raw.is_file():
                record['source_format']='RAW_BIN'
                with raw.open('rb') as stream:
                    for name,entry in layout['arrays'].items():
                        stream.seek(entry['source_offset']);view=views[name].array.view('u1').reshape(-1)
                        if stream.readinto(memoryview(view))!=entry['bytes']:raise IOError('Truncated fixed hot weight '+name)
            else:
                record['source_format']='NPZ'
                with np.load(path,allow_pickle=False) as archive:
                    for name,entry in layout['arrays'].items():
                        value=archive[name]
                        if value.shape!=tuple(entry['shape']) or value.dtype!=np.dtype(entry['dtype']):raise ValueError('Changed hot-weight layout')
                        views[name].array[:]=value
        finally:record['source_read_copy_seconds']=(time.monotonic_ns()-began)/1e9
        began=time.monotonic_ns()
        try:parent.root.flush()
        finally:record['flush_seconds']=(time.monotonic_ns()-began)/1e9
        record['ok']=True
        return parent,views
    finally:
        ended=time.monotonic_ns();storage_after=_process_storage_reads()
        record.update(end_ns=ended,total_seconds=(ended-started)/1e9,
            thread_cpu_seconds=(time.thread_time_ns()-cpu)/1e9,
            process_storage_read_bytes=None if storage_before is None or storage_after is None else storage_after-storage_before)
        # CPython list append is atomic; initialization occurs before the
        # single owned-weight loading executor can start concurrent loads.
        records=getattr(board,'weight_load_records',None)
        if records is None:records=[];board.weight_load_records=records
        records.append(record)


class FixedWeightStore:
    """Retain the existing two-slot read-ahead policy, without removed arrays."""
    def __init__(self,board,folder,rank,policy,layouts):
        if policy not in ('resident','prefetch2'):raise ValueError('Fixed weight policy')
        self.board,self.policy=board,policy
        self.paths=[Path(folder)/f'rank{rank}/layer{layer:02d}.npz' for layer in range(28)]
        self.layouts=layouts
        sizes=[x['bytes'] for x in layouts] if policy=='resident' else [max(x['bytes'] for x in layouts)]*2
        allocate=getattr(board,'make_weight',board.make)
        self.parents=[allocate((size,),'u1') for size in sizes]
        self.entries={};self.slots=OrderedDict();self.pending={};self.closed=False
        self.executor=ThreadPoolExecutor(max_workers=1) if policy=='prefetch2' else None
        self.plan=dict(policy=policy,compact_fixed_weights=True,arrays=LAYER_NAMES,
                       source_layer_bytes=sum(x['source_bytes'] for x in layouts),
                       hot_layer_bytes=sum(x['bytes'] for x in layouts),
                       allocated_weight_bytes=sum(aligned(x+256,4096) for x in sizes))

    def _slot(self,index):
        if len(self.slots)<len(self.parents):slot=min(set(range(len(self.parents)))-set(self.slots.values()))
        else:
            old,slot=self.slots.popitem(last=False)
            if old in self.pending:self.pending.pop(old).result()
            self.entries.pop(old,None)
        self.slots[index]=slot;return slot

    def _load(self,index,slot):return load_compact(self.board,self.paths[index],self.layouts[index],self.parents[slot])[1]

    def get(self,index):
        if self.closed or type(index) is not int or not 0<=index<28:raise ValueError('Invalid fixed layer')
        start=time.monotonic();loaded=False
        if index in self.pending:self.entries[index]=self.pending.pop(index).result();loaded=True
        if index not in self.entries:
            slot=index if self.policy=='resident' else self._slot(index)
            self.entries[index]=self._load(index,slot);loaded=True
        if self.policy=='prefetch2':self.slots.move_to_end(index)
        return self.entries[index],(time.monotonic()-start)*1000 if loaded else 0.

    def prefetch(self,index):
        if self.closed or self.policy=='resident' or not 0<=index<28 or index in self.entries or index in self.pending:return
        slot=self._slot(index);self.pending[index]=self.executor.submit(self._load,index,slot)

    def preload(self):
        if self.policy!='resident':raise ValueError('Complete preload requires resident policy')
        for index in range(28):self.get(index)

    def wait_idle(self,timeout=None):
        completed=[]
        for index,future in list(self.pending.items()):
            self.entries[index]=future.result(timeout=timeout)
            self.pending.pop(index);completed.append(index)
        return dict(pending_prefetch=0,completed_prefetch=completed,allocations_retained=True)

    def close(self):
        if self.executor:self.executor.shutdown(wait=True)
        self.entries.clear();self.pending.clear();self.closed=True
