"""Explicit DMA arenas and bounded ownership. Cold files are not CMA residency."""
from dataclasses import dataclass
import threading
from tp4_profile import PROFILE
from c209_tensor_layout import PackedKVLayout

def guarded(n):return (n+256+4095)//4096*4096

def memory_plan(layer_bytes,common_bytes,cross_bytes,global_bytes,*,budget_bytes=880*1024**2):
    p=PROFILE;N,D,W,F=p['tokens'],p['hidden'],p['local_width'],p['local_ffn']
    shapes={'state0':((N,D),'<f4'),'state1':((N,D),'<f4'),'quant':((N,F),'<i2'),
        'sa':((N,),'<f4'),'q':((N,W),'<f2'),'k':((N,W),'<f2'),'v':((N,W),'<f2'),
        'attn':((N,W),'<f2'),'ffn':((N,F),'<f2'),'gamma':((D,),'<f4'),
        'input':((N,80),'<f4'),'prediction':((N,64),'<f4'),'positive_prediction':((N,64),'<f4'),
        'cos':((N,128),'<f4'),'sin':((N,128),'<f4'),
        'gate_padded':((((D*4+8191)//8192)*2048,),'<f4')}
    for kind in ('partial','sum'):
        for slot in range(2):shapes[f'{kind}{slot}']=((p['tile_rows'],D),'<f4')
    sizes={name:__import__('math').prod(shape)*(2 if dtype.endswith('2') else 4) for name,(shape,dtype) in shapes.items()}
    kv=PackedKVLayout(N,p['local_heads'])
    fixed=sum(guarded(n) for n in sizes.values())+kv.guarded_bytes+guarded(global_bytes)
    weights=2*guarded(max(layer_bytes))
    caches=guarded(common_bytes)+sum(guarded(n) for n in cross_bytes)
    transport=16*1024**2;reserve=32*1024**2;descriptor=guarded(128*128)
    required=fixed+weights+caches+transport+reserve+descriptor
    if required>budget_bytes:raise MemoryError(f'TP4 CMA plan {required} exceeds {budget_bytes}')
    return dict(shapes=shapes,tensor_bytes=sizes,self_kv_bytes=kv.guarded_bytes,hot_weight_bytes=weights,
        cache_bytes=caches,global_bytes=global_bytes,transport_reserve_bytes=transport,engineering_reserve_bytes=reserve,
        required_cma_bytes=required,budget_bytes=budget_bytes,cold_file_bytes=sum(layer_bytes),
        cold_page_cache_policy='OPTIONAL_RECLAIMABLE_ORDINARY_DDR_NOT_PINNED',resident_weights=False)

class SlotPool:
    """Acquire BEFORE producer writes; release only after every native reader drains."""
    def __init__(self,count):
        self._condition=threading.Condition();self._free=list(range(count));self._owners={};self._next=0;self._failure=None
    def acquire(self,owner,timeout=None):
        with self._condition:
            if not self._condition.wait_for(lambda:bool(self._free) or self._failure is not None,timeout):raise TimeoutError('No free DMA slot')
            if self._failure is not None:raise RuntimeError('DMA slot pool poisoned; allocations retained') from self._failure
            slot=self._free.pop(0);self._next+=1;token=(slot,self._next,owner);self._owners[slot]=token;return token
    def release(self,token):
        with self._condition:
            if self._owners.get(token[0])!=token:raise RuntimeError('Stale/foreign slot release')
            del self._owners[token[0]];self._free.append(token[0]);self._condition.notify()
    @property
    def busy(self):
        with self._condition:return bool(self._owners)
    def fail(self,error):
        with self._condition:self._failure=error;self._condition.notify_all()

class HiddenSlots(SlotPool):
    def __init__(self):super().__init__(2);self.states={}
    def reserve(self,step,condition):
        token=self.acquire((step,condition),timeout=0);self.states[token]='RECEIVING';return token
    def transition(self,token,before,after):
        if self.states.get(token)!=before:raise RuntimeError('Invalid hidden lifetime transition')
        self.states[token]=after
    def acknowledge(self,token):
        if self.states.get(token)!='OUTBOUND':raise RuntimeError('Hidden released before delivery acknowledgement')
        del self.states[token]
        self.release(token)
