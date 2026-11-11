"""Deterministic TP square and SP three-node ring collectives.

Weights retain the original raw broadcast. Self KV uses two full-duplex
raw spans with same-head K16/V16 layout and explicit asymmetric tails.
"""
import threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from model_runner import View
from sp3_layout import ROWS,STARTS,kv_spans

class SP3Transport:
    def __init__(self,board,role,generation):
        if type(generation)is not int or not 1<=generation<=4095:raise ValueError('SP3 wire generation 1..4095')
        self.b,self.role,self.generation=board,role,generation
        self.serial=0;self.lock=threading.Lock();self.records=[]
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='sp3-fanout')
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
    def _join_native(self,futures,failure=None):
        # A failure on either lane must cancel its sibling promptly; every
        # submitted native task still joins before tensor ownership returns.
        if failure is not None:self.b.cancel_native()
        for future in as_completed(futures):
            try:future.result()
            except BaseException as error:
                if failure is None:
                    failure=error;self.b.cancel_native()
        if failure is not None:raise failure
    @staticmethod
    def byte_view(parent,offset,n):return View(parent,offset,(n,),'u1')
    def _group_broadcast(self,tensor,root,step):
        # XOR1 then XOR2: 0->1, then 0->2 and 1->3, generalized by root.
        relative=self.role.rank^root
        for round_id in range(2):
            bit=1<<round_id
            if relative<2*bit:
                peer=(self.role.rank^bit)+1;sending=relative<bit
                self._span(round_id,peer,tensor,sending,step)
            else:self._span(None,None,None,False,step)
    def broadcast(self,tensor,*,root=0,step,identity):
        with self.lock:
            self._begin('tp_broadcast',dict(step=step,identity=identity,root=root))
            self._group_broadcast(tensor,root,step);self._end()
    def _sp_fanout(self,tensor,owner,step):
        # Reserve both deterministic fragments before dispatching either link.
        # The owner uses independent P2/P3 lanes concurrently; each destination
        # still receives the exact same delta-specific epoch as the old order.
        # Workers perform only native span calls. All namespace/accounting
        # mutation stays in this calling thread under the transport lock.
        if self.fragment+2>256:raise RuntimeError('SP3 fragment namespace')
        operations=[]
        for delta in (1,2):
            epoch=(self.generation<<20)+(self.serial<<8)+self.fragment
            if epoch>0xffffffff:raise RuntimeError('SP3 fragment namespace')
            self.fragment+=1
            target=(owner+delta)%3
            if self.role.group==owner:port=2 if delta==1 else 3;peer=target+1;sending=True
            elif self.role.group==target:port=3 if delta==1 else 2;peer=owner+1;sending=False
            else:port=peer=None;sending=False
            if port is not None:operations.append((port,peer,sending,epoch))
        futures=[];failure=None
        try:
            for port,peer,sending,epoch in operations:
                futures.append((self.pool.submit(self.b.raw_span,port,peer,tensor,
                    sending=sending,epoch=epoch,step=step,channel=0),sending))
        except BaseException as error:failure=error
        # Always join every submitted operation, even after an error, before
        # the caller can release tensor/weight ownership or close the pool.
        self._join_native([future for future,_ in futures],failure)
        for future,sending in futures:
            self.record['wire_payload_bytes']+=2*tensor.nbytes
            self.record['paired_dummy_bytes']+=tensor.nbytes
            if sending:self.record['useful_bytes']+=tensor.nbytes
    def distribute_weight(self,parent,layer,*,step):
        with self.lock:
            self._begin('weight',dict(step=step,layer=layer,owner=layer//12))
            self._sp_fanout(parent,layer//12,step);self._end()
    def allgather_kv(self,workspace,*,step,layer,condition):
        with self.lock:
            if type(step)is not int or not 0<=step<19 or type(layer)is not int or not 0<=layer<36 or condition not in ('positive','negative'):
                raise ValueError('Fixed self-KV context')
            context=(step,layer,condition)
            if self._next_kv is None:
                if layer!=0 or condition!='positive':raise RuntimeError('KV plan must start with the shared layer0 prefix')
            elif context!=self._next_kv:raise RuntimeError('KV transaction order differs from the fixed CFG plan')
            self._begin('self_kv',dict(step=step,layer=layer,condition=condition))
            if workspace.layout.tokens!=4096 or workspace.layout.heads!=10:
                raise ValueError('Full Y3 KV16 workspace required')
            spans=[tuple(kv_spans(group,workspace.layout.heads)) for group in range(3)]
            count=len(spans[0])
            if any(len(s)!=count for s in spans) or count*3>256:raise ValueError('KV fragment contract')
            edges={(0,1):0,(1,2):1,(0,2):2}
            for index,local in enumerate(spans[self.role.group]):
                part,_,offset,n=local
                outgoing=self.byte_view(workspace.buffers[part],offset,n)
                futures=[];sizes=[];failure=None
                # Every endpoint derives the same undirected-edge epoch.
                # Different local completion order cannot renumber packets.
                try:
                    for port,delta in ((2,1),(3,2)):
                        peer=(self.role.group+delta)%3
                        incoming_part,_,incoming_offset,incoming_n=spans[peer][index]
                        if incoming_part!=part:raise ValueError('KV component ordering')
                        incoming=self.byte_view(workspace.buffers[part],incoming_offset,incoming_n)
                        edge=edges[tuple(sorted((self.role.group,peer)))]
                        epoch=(self.generation<<20)+(self.serial<<8)+index*3+edge
                        if epoch>0xffffffff:raise ValueError('KV epoch overflow')
                        futures.append(self.pool.submit(self.b.raw_duplex_span,port,peer+1,outgoing,incoming,
                            epoch=epoch,step=step,channel=0))
                        sizes.append((n,incoming_n))
                except BaseException as error:failure=error
                self._join_native(futures,failure)
                for tx_bytes,rx_bytes in sizes:
                    self.record['useful_bytes']+=tx_bytes
                    self.record['wire_payload_bytes']+=2*max(tx_bytes,rx_bytes)
                    self.record['paired_dummy_bytes']+=abs(tx_bytes-rx_bytes)
            self.fragment=count*3
            self.record['layout']='SELF_KV16_K_HEAD_TOKEN_FOLDED_VS_V2'
            self.record['exchange']='THREE_NODE_FULL_DUPLEX_V1'
            self._end()
            if layer==0:self._next_kv=(step,1,'positive')
            elif condition=='positive':self._next_kv=(step,layer,'negative')
            elif layer<35:self._next_kv=(step,layer+1,'positive')
            else:self._next_kv=(step+1,0,'positive')
    def gather_prediction(self,local,full,*,step,condition):
        # Rank0 of each sequence group supplies its distinct token slice to
        # group0/rank0. Other TP ranks keep matching transaction numbering.
        with self.lock:
            self._begin('prediction',dict(step=step,condition=condition))
            for owner in range(3):
                target=View(full,STARTS[owner]*64*4,(ROWS[owner],64),'<f4')
                if self.role.group==owner and self.role.rank==0:
                    local.root.invalidate();target.array[:]=local.array;target.root.flush()
                if owner==0:continue
                if self.role.rank==0 and self.role.group in (0,owner):
                    sending=self.role.group==owner
                    port=(2 if owner==2 else 3) if sending else (2 if owner==1 else 3)
                    self._span(port,(0 if sending else owner)+1,target,sending,step)
                else:self._span(None,None,None,False,step)
            self._end()
    def share_latent(self,latent,*,step):
        with self.lock:
            self._begin('latent',dict(step=step))
            # Only rank0 uses SP links, then each TP square gets the result.
            if self.role.rank==0:self._sp_fanout(latent,0,step)
            else:
                self._span(None,None,None,False,step);self._span(None,None,None,False,step)
            self._group_broadcast(latent,0,step);self._end()
    def close(self):
        with self.lock:
            if not self.closed:
                self.closed=True;self.pool.shutdown(wait=True)
