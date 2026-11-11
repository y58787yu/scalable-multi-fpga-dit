"""TP2 P1 collectives and one serialized P2 full-duplex SP2 peer link.

Complete K16/V16 spans retain their original addresses, lengths and epochs.
The transport lock orders weight, KV, latent and prediction transactions.
"""
import threading
from model_runner import View
from sp3_layout import ROWS,STARTS,kv_spans
from tp4_profile import PROFILE
GROUPS=PROFILE["sequence_parallel"]
OWN=PROFILE["layers_per_weight_group"]

class SP3Transport:
    def __init__(self,board,role,generation):
        if type(generation)is not int or not 1<=generation<=4095:raise ValueError('SP3 wire generation 1..4095')
        self.b,self.role,self.generation=board,role,generation
        self.serial=0;self.lock=threading.Lock();self.records=[]
        self.closed=False
        self._next_kv=None
    def _begin(self,name,context):
        if self.closed:raise RuntimeError('SP3 transport is closed')
        self.serial+=1
        if self.serial>=3900:raise RuntimeError('SP3 run exhausted bounded transaction namespace')
        self.record=dict(name=name,context=dict(context),transaction=self.serial,useful_bytes=0,wire_payload_bytes=0,paired_dummy_bytes=0)
        self.fragment=0
    def _span(self,port,peer,tensor,sending,step):
        # All roles allocate every fragment ID, even when they do not take part.
        epoch=(self.generation<<20)+(self.serial<<8)+self.fragment
        if self.fragment>=256 or epoch>0xffffffff:raise RuntimeError('SP3 fragment namespace')
        self.fragment+=1
        if port is None:return
        self.b.raw_span(port,peer,tensor,sending=sending,epoch=epoch,step=step,channel=0)
        self.record['wire_payload_bytes']+=2*tensor.nbytes
        self.record['paired_dummy_bytes']+=tensor.nbytes
        if sending:self.record['useful_bytes']+=tensor.nbytes
    def _end(self):self.records.append(self.record)
    @staticmethod
    def byte_view(parent,offset,n):return View(parent,offset,(n,),'u1')
    def _group_broadcast(self,tensor,root,step):
        if root not in (0,1):raise ValueError('TP2 root')
        self._span(1,(self.role.rank^1)+1,tensor,self.role.rank==root,step)
    def broadcast(self,tensor,*,root=0,step,identity):
        with self.lock:
            self._begin('tp_broadcast',dict(step=step,identity=identity,root=root))
            self._group_broadcast(tensor,root,step);self._end()
    def _sp_fanout(self,tensor,owner,step):
        if owner not in range(GROUPS):raise ValueError('SP owner')
        if GROUPS==1:return
        sending=self.role.group==owner
        # Both SP groups retain physical P2; one owner publishes each transaction.
        self._span(2,(self.role.group^1)+1,tensor,sending,step)
    def distribute_weight(self,parent,layer,*,step):
        with self.lock:
            self._begin('weight',dict(step=step,layer=layer,owner=layer//OWN))
            self._sp_fanout(parent,layer//OWN,step);self._end()
    def allgather_kv(self,workspace,*,step,layer,condition):
        with self.lock:
            if type(step)is not int or not 0<=step<19 or type(layer)is not int or not 0<=layer<28 or condition not in ('positive','negative'):
                raise ValueError('Fixed self-KV context')
            context=(step,layer,condition)
            if self._next_kv is None:
                if layer!=0 or condition!='positive':raise RuntimeError('KV plan must start with the shared layer0 prefix')
            elif context!=self._next_kv:raise RuntimeError('KV transaction order differs from the fixed CFG plan')
            self._begin('self_kv',dict(step=step,layer=layer,condition=condition))
            if workspace.layout.tokens!=4096 or workspace.layout.heads!=8:
                raise ValueError('Full Y3 KV16 workspace required')
            spans=[tuple(kv_spans(group,8)) for group in range(GROUPS)]
            count=len(spans[0])
            if any(len(v)!=count for v in spans) or count>256:raise ValueError('KV fragment contract')
            if GROUPS==2:
                peer=self.role.group^1
                # Exactly one component is live on P2. Both peers traverse the
                # same index order, so READY/credit cannot wait on a different
                # component selected by a competing worker. A native return
                # proves its complete TX/RX byte-count and tail drain.
                for index in range(count):
                    part,_,offset,n=spans[self.role.group][index]
                    other,_,incoming_offset,incoming_n=spans[peer][index]
                    if other!=part:raise ValueError('KV component ordering')
                    outgoing=self.byte_view(workspace.buffers[part],offset,n)
                    incoming=self.byte_view(workspace.buffers[part],incoming_offset,incoming_n)
                    epoch=(self.generation<<20)+(self.serial<<8)+index
                    if epoch>0xffffffff:raise ValueError('KV epoch overflow')
                    try:
                        self.b.raw_duplex_span(2,peer+1,outgoing,incoming,epoch=epoch,step=step,channel=0)
                    except BaseException:
                        self.b.cancel_native();raise
                    self.record['useful_bytes']+=n
                    self.record['wire_payload_bytes']+=2*max(n,incoming_n)
                    self.record['paired_dummy_bytes']+=abs(n-incoming_n)
            self.fragment=count if GROUPS==2 else 0
            self.record['layout']='SELF_KV16_K_HEAD_TOKEN_FOLDED_VS_V2'
            self.record['exchange']='TP2_SP2_P2_SERIAL_FULL_DUPLEX_V1' if GROUPS==2 else 'SP1_LOCAL_COMPLETE_KV'
            self._end()
            if layer==0:self._next_kv=(step,1,'positive')
            elif condition=='positive':self._next_kv=(step,layer,'negative')
            elif layer<27:self._next_kv=(step,layer+1,'positive')
            else:self._next_kv=(step+1,0,'positive')
    def gather_prediction(self,local,full,*,step,condition):
        # Rank0 of each sequence group supplies its distinct token slice to
        # group0/rank0. Other TP ranks keep matching transaction numbering.
        with self.lock:
            self._begin('prediction',dict(step=step,condition=condition))
            for owner in range(GROUPS):
                target=View(full,STARTS[owner]*64*4,(ROWS[owner],64),'<f4')
                if self.role.group==owner and self.role.rank==0:
                    local.root.invalidate();target.array[:]=local.array;target.root.flush()
                if owner==0:continue
                if self.role.rank==0 and self.role.group in (0,owner):
                    sending=self.role.group==owner
                    port=2
                    self._span(port,(0 if sending else owner)+1,target,sending,step)
                else:self._span(None,None,None,False,step)
            self._end()
    def share_latent(self,latent,*,step):
        with self.lock:
            self._begin('latent',dict(step=step))
            # Only rank0 uses SP links, then each TP square gets the result.
            if self.role.rank==0:self._sp_fanout(latent,0,step)
            else:
                
                if GROUPS==2:self._span(None,None,None,False,step)
            self._group_broadcast(latent,0,step);self._end()
    def close(self):
        with self.lock:
            if not self.closed:
                self.closed=True
