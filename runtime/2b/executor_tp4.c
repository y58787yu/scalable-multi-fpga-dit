/* TP2 dual-port shell: P1 FUSED and P2 RAW, no P0/P3 MMIO. */
#include "tp4_contract.h"
#define COSMOS_C209_BIT_ID TP4_BIT_ID
/* Let the compute waiter observe the same cooperative cancellation as TP/SP.
 * Returning from the waiter never asserts that the PL core has stopped. */
static int tp4_is_cancelled(void);
#define C209_SERIAL_CANCEL_REQUESTED() tp4_is_cancelled()
#include "executor_c209.c"
static atomic_uint tp4_cancelled=0;
void cosmos_tp4_cancel(uint32_t enabled){atomic_store_explicit(&tp4_cancelled,enabled!=0,memory_order_release);}
static int tp4_is_cancelled(void){return atomic_load_explicit(&tp4_cancelled,memory_order_acquire)!=0;}
static int tp4_direct_peer(uint32_t node,uint32_t peer,uint32_t lane){
 if(lane==1||lane==2)return node>=1&&node<=2&&peer==(((node-1)^1U)+1);
 return 0;
}
typedef struct {
 C206Lane lane;
 uint64_t local_phys,output_phys,residual_phys,gate_phys;
 uint32_t rows,cols,first_row,node,peer,lane_id,fused,epoch,step,condition,layer,operator_id,timeout_ms;
} TP4PairJob;
uint32_t cosmos_tp4_runtime_identity(void){return TP4_RUNTIME_ID;}
/* Dual-port package: P0/P3 are absent; only P1 is fused-capable. */
uint32_t cosmos_tp4_sum_only_mask(void){return 0U;}
size_t cosmos_tp4_pair_job_size(void){return sizeof(TP4PairJob);}

/* One in-flight credited packet per physical lane. Polling two independent
 * lanes allows SUM(i+1) to overlap FUSED(i); each packet still retires only
 * after its original DMA/reducer/link drain checks. */
typedef struct {
 C206Lane *lane;
 C206Pending pending;
 uint64_t tx,started;
 uint32_t bytes,polls;
 int mode;
} TP4Transfer;
static int tp4_transfer_begin(TP4Transfer*t,C206Lane*l,uint64_t tx,uint64_t out,uint64_t residual,uint32_t n,
 uint32_t node,uint32_t peer,uint32_t lane,uint32_t group,uint32_t epoch,uint32_t step,
 uint32_t condition,uint32_t layer,uint32_t op,uint32_t chunk,uint32_t ms,int mode,int same_segment){
 /* mode0=SUM,1=FUSED,2=RAW. Peer switches only at a drained operation boundary. */
 if(tp4_is_cancelled())return -15;
 if(!l->dma||!l->link||!l->reducer||(mode==1&&!l->residual_dma))return -12;
 if(!same_segment){
 if(R(l->link,0)!=0xc206e001U||R(l->reducer,0)!=0xc206f001U||R(l->link,0x7c)!=TP4_BIT_ID)return -12;
 if((R(l->link,4)&1)||(R(l->reducer,4)&1))return -4;
 if((R(l->dma,0x28)&&!(R(l->dma,4)&3))||(R(l->dma,0x58)&&!(R(l->dma,0x34)&3)))return -4;
 W(l->link,0x90,node);W(l->link,0x94,peer);
 if(R(l->link,0x90)!=node||R(l->link,0x94)!=peer)return -12;
 }
 if(!(R(l->link,4)&32))return -2;
 *t=(TP4Transfer){0};t->lane=l;t->tx=tx;t->bytes=n;t->mode=mode;t->started=NOW();
 C206Pending p={0};p.id=0x40000000U|chunk;
 W(l->reducer,8,2);
 /* Only rows inside this one native pair/column segment reuse config. The
  * caller holds its lane lock, and every prior row returned after full drain.
  * RTL reducer CLEAR retains bytes/base/mode/raw config; link ARM retains
  * context config. Re-publish after every gate preload/column transition and
  * for every separate native call. No process-global CSR cache is used. */
 if(!same_segment){
  W(l->reducer,0x24,mode==2);W(l->reducer,0xc,n);
  W(l->reducer,0x10,0);W(l->reducer,0x14,mode==1);
 }
 c206_write(l->dma,out,n);
 if(!same_segment){
  W(l->link,0x9c,group);W(l->link,0xa0,epoch);W(l->link,0xa4,step);
  W(l->link,0xa8,condition);W(l->link,0xac,layer);W(l->link,0xb0,op);W(l->link,0xb8,lane);
 }
 W(l->link,0xb4,chunk);
 W(l->link,0x20,p.id);W(l->link,0x24,n);W(l->link,0x28,1);W(l->link,0x2c,n/4);W(l->link,4,4);
 W(l->reducer,8,1);if(mode==1)c206_read(l->residual_dma,residual,n);
 p.phase=1;p.deadline=NOW()+(uint64_t)ms*1000000ULL;t->pending=p;
 return 0;
}
/* 1=still active,0=fully drained,negative=unchanged native error. */
static int tp4_transfer_poll(TP4Transfer*t){
 C206Lane*l=t->lane;C206Pending *p=&t->pending;
 uint32_t n=t->bytes;int mode=t->mode;
  if(tp4_is_cancelled())return -15;
  int e=0;if((t->polls++&31u)==0u){e=c206_fault(l,mode!=1);if(e)return e;}
  if(NOW()>p->deadline)return -1;
  uint32_t st=R(l->link,4);
  if(p->phase==1&&!(st&1)){p->before=c206_tx(l->link);W(l->link,0x10,p->id);W(l->link,0x14,0);W(l->link,0x18,2);W(l->link,0x1c,0);W(l->link,4,16);p->phase=2;}
  else if(p->phase==2&&c206_tx(l->link)!=p->before&&(st&3)==2)p->phase=3;
  else if(p->phase==3&&(st&64)&&R(l->link,0x80)){
   if(R(l->link,0x30)!=2||R(l->link,0x34)!=p->id)return -7;
   W(l->link,0x84,1);p->before=c206_tx(l->link);W(l->link,0x10,p->id);W(l->link,0x14,n);W(l->link,0x18,1);W(l->link,0x1c,n/4);W(l->link,4,1);p->phase=4;
  }else if(p->phase==4&&c206_tx(l->link)!=p->before){c206_read(l->dma,t->tx,n);p->phase=5;}
  else if(p->phase==5&&(R(l->dma,4)&0x1000)&&(R(l->dma,0x34)&0x1000)&&
      (mode!=1||(R(l->residual_dma,4)&0x1000))&&(R(l->reducer,4)&2)&&!(st&1)&&(st&4)){
   if(R(l->link,0x40)!=n||R(l->reducer,0x1c)!=n||R(l->reducer,0x20)!=n||(mode==2&&R(l->reducer,0x28)!=n))return -3;
   if((e=c206_fault(l,mode!=1)))return e;
   return 0;
  }
 return 1;
}
static int tp4_transfer(C206Lane*l,uint64_t tx,uint64_t out,uint64_t residual,uint32_t n,
 uint32_t node,uint32_t peer,uint32_t lane,uint32_t group,uint32_t epoch,uint32_t step,
 uint32_t condition,uint32_t layer,uint32_t op,uint32_t chunk,uint32_t ms,int mode,int same_segment){
 TP4Transfer transfer;int e=tp4_transfer_begin(&transfer,l,tx,out,residual,n,node,peer,lane,group,epoch,step,condition,layer,op,chunk,ms,mode,same_segment);
 if(e)return e;
 do {e=tp4_transfer_poll(&transfer);}while(e==1);
 return e;
}

int cosmos_tp4_pair(TP4PairJob*j,C206StreamStats*s){
 if(!j||!s)return -5;
 *s=(C206StreamStats){0};uint64_t start=NOW();int e=-5;
 if(j->cols!=TP4_HIDDEN||(j->rows!=TP4_TILE_ROWS&&j->rows!=TP4_TAIL_ROWS)||j->first_row%128||j->first_row>TP4_MAX_LOCAL_ROWS-j->rows||
    !j->lane.dma||!j->lane.link||!j->lane.reducer||(j->fused&&!j->lane.residual_dma)||
    j->node<1||j->node>TP4_TENSOR_PARALLEL||j->fused!=1||j->lane_id!=1||
    j->peer!=(((j->node-1)^1U)+1)||!j->timeout_ms||!j->epoch||
    j->step>=TP4_UPDATES||j->condition>1||j->layer>=TP4_LAYERS||j->operator_id>=3||
    !j->local_phys||!j->output_phys||j->local_phys%64||j->output_phys%64||
    (j->fused&&(!j->residual_phys||!j->gate_phys||j->residual_phys%64||j->gate_phys%64)))goto done;
 if(R(j->lane.link,0)!=0xc206e001U||R(j->lane.reducer,0)!=0xc206f001U||R(j->lane.link,0x7c)!=TP4_BIT_ID){e=-12;goto done;}
 if((R(j->lane.link,4)&1)||(R(j->lane.reducer,4)&1)||
    (j->fused&&(R(j->lane.residual_dma,0x28)&&!(R(j->lane.residual_dma,4)&3)))){e=-4;goto done;}
 for(uint32_t col=0;col<j->cols;col+=TP4_PACKET_MAX_BYTES/4){
  uint32_t n=(j->cols-col)*4;if(n>TP4_PACKET_MAX_BYTES)n=TP4_PACKET_MAX_BYTES;
  if(j->fused){
   W(j->lane.reducer,8,2);W(j->lane.reducer,0x24,0);W(j->lane.reducer,0xc,TP4_PACKET_MAX_BYTES);W(j->lane.reducer,8,4);
   c206_read(j->lane.residual_dma,j->gate_phys+(uint64_t)col*4,TP4_PACKET_MAX_BYTES);
   uint64_t end=NOW()+(uint64_t)j->timeout_ms*1000000ULL;
   while(!(R(j->lane.residual_dma,4)&0x1000)||!(R(j->lane.reducer,4)&8)){
    if(tp4_is_cancelled()){e=-15;goto done;}
    if((e=c206_fault(&j->lane,0)))goto done;if(NOW()>end){e=-1;goto done;}
   }
  }
  uint32_t segments=(j->cols*4+TP4_PACKET_MAX_BYTES-1)/TP4_PACKET_MAX_BYTES;
  for(uint32_t row=0;row<j->rows;row++){
  uint64_t offset=((uint64_t)row*j->cols+col)*4;
  uint32_t chunk=j->first_row*segments+(col/(TP4_PACKET_MAX_BYTES/4))*j->rows+row;
  e=tp4_transfer(&j->lane,j->local_phys+offset,j->output_phys+offset,j->residual_phys+offset,n,
      j->node,j->peer,j->lane_id,j->condition,j->epoch,j->step,j->condition,j->layer,j->operator_id,chunk,j->timeout_ms,j->fused,row!=0);
  if(e)goto done;s->chunks++;s->tx_bytes+=n;s->rx_bytes+=n;
  }
 }
 e=0;
done:s->elapsed_ns=NOW()-start;s->error=e?(uint32_t)-e:0;return e;
}

uint32_t cosmos_tp4_chunk_pipeline_revision(void){return 1U;}
static int tp4_pipeline_job_valid(const TP4PairJob*j,unsigned fused){
 return j&&j->cols==TP4_HIDDEN&&(j->rows==TP4_TILE_ROWS||j->rows==TP4_TAIL_ROWS)&&
  j->first_row%128==0&&j->first_row<=TP4_MAX_LOCAL_ROWS-j->rows&&
  j->lane.dma&&j->lane.link&&j->lane.reducer&&(!fused||j->lane.residual_dma)&&
  j->node>=1&&j->node<=TP4_TENSOR_PARALLEL&&fused==1&&j->fused==1&&j->lane_id==1&&
  j->peer==(((j->node-1)^1U)+1)&&j->timeout_ms&&j->epoch&&
  j->step<TP4_UPDATES&&j->condition<=1&&j->layer<TP4_LAYERS&&j->operator_id<3&&
  j->local_phys&&j->output_phys&&j->local_phys%64==0&&j->output_phys%64==0&&
  (!fused||(j->residual_phys&&j->gate_phys&&j->residual_phys%64==0&&j->gate_phys%64==0));
}
/* Preserve SUM->FUSED arithmetic order for EVERY word. The intermediate
 * full-tile buffer already exists; a completed SUM packet publishes one
 * contiguous row segment, which alone may enter the second lane. A lane
 * never has more than its original one credited packet in flight.
 * No new PL buffers, AXI masters, threads, or speculative DMA reads. */
int cosmos_tp4_pair_pipeline(TP4PairJob*unused,TP4PairJob*fused,C206StreamStats*stats){
 (void)unused;if(!stats)return -5;stats[0]=(C206StreamStats){0};
 return cosmos_tp4_pair(fused,&stats[1]);
}

int cosmos_tp4_bulk(C206BulkJob*j,C206StreamStats*s){
 if(!j||!s||!j->bytes||j->bytes>TP4_PACKET_MAX_BYTES||j->bytes%8||!j->tx_phys||!j->rx_phys||j->tx_phys%64||j->rx_phys%64||
    (j->lane_id!=1&&j->lane_id!=2)||!j->timeout_ms||!j->node||!j->peer||j->node==j->peer||
    j->node>(j->lane_id<2?TP4_TENSOR_PARALLEL:TP4_SEQUENCE_PARALLEL)||j->peer>(j->lane_id<2?TP4_TENSOR_PARALLEL:TP4_SEQUENCE_PARALLEL)||j->channel>255||
    !tp4_direct_peer(j->node,j->peer,j->lane_id)||!j->job_id||j->step>=TP4_UPDATES)return -5;
 uint64_t start=NOW();*s=(C206StreamStats){0};
 int e=tp4_transfer(&j->lane,j->tx_phys,j->rx_phys,0,j->bytes,j->node,j->peer,j->lane_id<2?j->lane_id:2,0x80,j->job_id,j->step,0x80,0,j->channel,j->sequence,j->timeout_ms,2,0);
 s->elapsed_ns=NOW()-start;s->error=e?(uint32_t)-e:0;
 if(!e){s->chunks=1;s->tx_bytes=j->bytes;s->rx_bytes=j->bytes;}return e;
}

typedef struct {
 C206Lane lane;
 uint64_t tensor_phys,dummy_tx_phys,dummy_rx_phys;
 uint32_t bytes,node,peer,lane_id,epoch,step,channel,timeout_ms,sending;
} TP4SpanJob;
size_t cosmos_tp4_span_job_size(void){return sizeof(TP4SpanJob);}
int cosmos_tp4_bulk_span(TP4SpanJob*j,C206StreamStats*s){
 if(!j||!s||!j->bytes||j->bytes>512U*1024U*1024U||j->bytes%8||j->sending>1||!j->timeout_ms||(j->lane_id!=1&&j->lane_id!=2)||
    !j->node||!j->peer||j->node==j->peer||j->node>(j->lane_id<2?TP4_TENSOR_PARALLEL:TP4_SEQUENCE_PARALLEL)||
    j->peer>(j->lane_id<2?TP4_TENSOR_PARALLEL:TP4_SEQUENCE_PARALLEL)||j->channel>255||
    !tp4_direct_peer(j->node,j->peer,j->lane_id)||!j->epoch||j->step>=TP4_UPDATES||
    !j->tensor_phys||!j->dummy_tx_phys||!j->dummy_rx_phys||j->tensor_phys%64||j->dummy_tx_phys%64||j->dummy_rx_phys%64)return -5;
 *s=(C206StreamStats){0};uint64_t start=NOW(),deadline=start+(uint64_t)j->timeout_ms*1000000ULL;
 int error=0;
 for(uint32_t offset=0,seq=0;offset<j->bytes;offset+=TP4_PACKET_MAX_BYTES,seq++){
  if(tp4_is_cancelled()){error=-15;break;}uint64_t now=NOW();if(now>=deadline){error=-1;break;}
  uint32_t n=j->bytes-offset;if(n>TP4_PACKET_MAX_BYTES)n=TP4_PACKET_MAX_BYTES;
  uint64_t remaining=(deadline-now)/1000000ULL;if(!remaining){error=-1;break;}
  /* This one call owns the lane and keeps its node/peer/context. Cache CSR
   * configuration across equal-size chunks; re-publish for a short final chunk. */
  error=tp4_transfer(&j->lane,j->sending?j->tensor_phys+offset:j->dummy_tx_phys,
      j->sending?j->dummy_rx_phys:j->tensor_phys+offset,0,n,j->node,j->peer,j->lane_id<2?j->lane_id:2,
      0x80,j->epoch,j->step,0x80,0,j->channel,seq,(uint32_t)remaining,2,offset!=0&&n==TP4_PACKET_MAX_BYTES);
  if(error)break;
  s->chunks++;s->tx_bytes+=n;s->rx_bytes+=n;
 }
 s->elapsed_ns=NOW()-start;s->error=error?(uint32_t)-error:0;return error;
}

#include "executor_duplex.h"

#include "executor_tp4_batch.h"

#include "executor_fixed_plan.h"
