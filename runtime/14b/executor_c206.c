/* C206: K4 routed TP2 stripes and raw, credited neighbour transfers.
 * Compute ABI is the unchanged, real C205 HLS ABI via executor_c204.c.
 * Lanes advance independently: even/odd rows map to logical lane0/1.
 * Logical lane2 is reserved for cross-pair fences; its physical port still relays.
 * No CPU floating-point operations, peer tensor copies or slow-lane batch fence.
 */
#include "executor_c204.c"

typedef struct { uint64_t dma, residual_dma, link, reducer; } C206Lane;
typedef struct {
 C206Lane lanes[3];
 uint64_t local_phys,state_phys,gate_phys;
 uint32_t bytes,cols,epoch,step,branch,layer,operator_id,group,node,peer,timeout_ms;
 uint32_t first_row,preload_gate;
} C206StreamJob;
typedef struct { uint64_t elapsed_ns; uint32_t error,chunks,rx_bytes,tx_bytes; } C206StreamStats;
typedef struct {
 C206Lane lane;
 uint64_t tx_phys,rx_phys;
 uint32_t bytes,node,peer,lane_id,job_id,step,channel,sequence,timeout_ms;
} C206BulkJob;

/* Tests replace MMIO at the boundary; production always performs volatile MMIO. */
#ifdef C206_MMIO_TEST
extern uint32_t c206_test_read(uint64_t,uint32_t);
extern void c206_test_write(uint64_t,uint32_t,uint32_t);
extern uint64_t c206_test_now(void);
#define R c206_test_read
#define W c206_test_write
#define NOW c206_test_now
#else
#define R rd
#define W wr
#define NOW now_ns
#endif
static void c206_addr(uint64_t b,uint32_t off,uint64_t p){W(b,off,(uint32_t)p);W(b,off+4,(uint32_t)(p>>32));}
static uint64_t c206_tx(uint64_t b){uint32_t hi=R(b,0x6c),lo=R(b,0x68);return ((uint64_t)hi<<32)|lo;}
static void c206_read(uint64_t b,uint64_t p,uint32_t n){W(b,4,0x1000);c206_addr(b,0x18,p);W(b,0x28,n);}
static void c206_write(uint64_t b,uint64_t p,uint32_t n){W(b,0x34,0x1000);c206_addr(b,0x48,p);W(b,0x58,n);}
static uint32_t c206_row_lane(uint32_t row){return row&1;}
static uint32_t c206_next_row(uint32_t first,uint32_t end,uint32_t lane){if(lane==2)return end;while(first<end&&c206_row_lane(first)!=lane)first++;return first;}
static uint32_t c206_neighbour(uint32_t node,uint32_t lane){static const uint32_t n[4][3]={{2,3,4},{1,4,3},{4,1,2},{3,2,1}};return n[node-1][lane];}
#ifndef COSMOS_TP_BIT_ID
#define COSMOS_TP_BIT_ID(node) ((node)%2?0xc206a001U:0xc206b002U)
#endif
static int c206_identity(C206Lane*l,uint32_t node,uint32_t peer){
 if(!l->dma||!l->residual_dma||!l->link||!l->reducer)return -12;
 if(R(l->link,0)!=0xc206e001U||R(l->reducer,0)!=0xc206f001U||
    R(l->link,0x7c)!=COSMOS_TP_BIT_ID(node)||
    R(l->link,0x90)!=node||R(l->link,0x94)!=peer)return -12;
 return 0;
}
static int c206_fault(C206Lane*l,int bulk){
 if((R(l->link,4)&8)||R(l->link,0x4c)||R(l->link,0x50)||R(l->link,0xd4)||
    (R(l->dma,4)&0x70)||(R(l->dma,0x34)&0x70)||
    (!bulk&&(R(l->residual_dma,4)&0x70))||(R(l->reducer,4)&4))return -3;
 return 0;
}
typedef struct {uint32_t row,id,phase;uint64_t deadline,before;} C206Pending;
static void c206_arm(C206Lane*l,C206Pending*p,uint32_t n,uint64_t tx,uint64_t rx,
 uint32_t group,uint32_t epoch,uint32_t step,uint32_t branch,uint32_t layer,
 uint32_t op,uint32_t chunk,uint32_t lane,uint32_t ms,int bulk){
 (void)tx;
 W(l->reducer,8,2);W(l->reducer,0x24,bulk?1:0);W(l->reducer,0xc,n);
 W(l->reducer,0x10,0);W(l->reducer,0x14,1);
 c206_write(l->dma,rx,n);
 W(l->link,0x9c,group);W(l->link,0xa0,epoch);W(l->link,0xa4,step);
 W(l->link,0xa8,branch);W(l->link,0xac,layer);W(l->link,0xb0,op);
 W(l->link,0xb4,chunk);W(l->link,0xb8,lane);
 W(l->link,0x20,p->id);W(l->link,0x24,n);W(l->link,0x28,1);W(l->link,0x2c,n/4);W(l->link,4,4);
 W(l->reducer,8,1);
 if(!bulk)c206_read(l->residual_dma,rx,n);
 p->phase=1;p->deadline=NOW()+(uint64_t)ms*1000000ULL;
}
/* Returns 1 only after both DMA legs, link and reducer have drained. */
static int c206_advance(C206Lane*l,C206Pending*p,uint64_t tx,uint32_t n,int bulk){
 int e=c206_fault(l,bulk);if(e)return e;
 if(NOW()>p->deadline)return -1;
 uint32_t status=R(l->link,4);
 if(p->phase==1&&!(status&1)){
  p->before=c206_tx(l->link);W(l->link,0x10,p->id);W(l->link,0x14,0);
  W(l->link,0x18,2);W(l->link,0x1c,0);W(l->link,4,16);p->phase=2;
 }else if(p->phase==2&&c206_tx(l->link)!=p->before&&(status&3)==2){p->phase=3;
 }else if(p->phase==3&&(status&64)&&R(l->link,0x80)){
  if(R(l->link,0x30)!=2||R(l->link,0x34)!=p->id)return -7;
  W(l->link,0x84,1);p->before=c206_tx(l->link);
  W(l->link,0x10,p->id);W(l->link,0x14,n);W(l->link,0x18,1);W(l->link,0x1c,n/4);W(l->link,4,1);p->phase=4;
 }else if(p->phase==4&&c206_tx(l->link)!=p->before){c206_read(l->dma,tx,n);p->phase=5;
 }else if(p->phase==5&&(R(l->dma,4)&0x1000)&&(R(l->dma,0x34)&0x1000)&&
          (bulk||(R(l->residual_dma,4)&0x1000))&&(R(l->reducer,4)&2)&&!(status&1)&&(status&4)){
  if(R(l->link,0x40)!=n||R(l->reducer,0x1c)!=n||R(l->reducer,0x20)!=n||
     (bulk&&R(l->reducer,0x28)!=n))return -3;
  p->phase=0;return 1;
 }
 return 0;
}
/* Private common engine. Public legacy admission below retains its original
 * step-only epoch rule; an explicit C209 resident entry validates another
 * namespace before calling this same MMIO/credit/drain implementation. */
static int c206_stream_reduce_expected_epoch(C206StreamJob*j,C206StreamStats*s,uint32_t expected_epoch){
 if(!j||!s||!j->timeout_ms||!j->bytes||j->bytes>4096*8192U||j->bytes%8192||j->cols!=2048||
    !j->local_phys||!j->state_phys||!j->gate_phys||j->local_phys%64||j->state_phys%64||j->gate_phys%64||
    j->step>=19||j->layer>=28||j->operator_id>=3||j->group>1||j->branch!=j->group||
    j->node<1||j->node>4||j->peer!=(j->node%2?j->node+1:j->node-1)||j->group!=(j->node>2)||
    j->epoch!=expected_epoch||j->preload_gate>1||j->first_row>4096-j->bytes/8192)return -5;
 uint64_t start=NOW(),deadline=start+(uint64_t)j->timeout_ms*1000000ULL;
 s->error=0;s->chunks=0;s->rx_bytes=0;s->tx_bytes=0;int error=0;
 for(uint32_t lane=0;lane<3;lane++){
  C206Lane*l=&j->lanes[lane];if((error=c206_identity(l,j->node,j->peer)))goto failed;
  while(!(R(l->link,4)&32)){if((error=c206_fault(l,0)))goto failed;if(NOW()>deadline){error=-1;goto failed;}}
  if((error=c206_fault(l,0)))goto failed;
  if(lane==2)continue; /* No endpoint TP context/DMA traffic on the fence lane. */
  W(l->reducer,0x24,0);
  if(j->preload_gate){W(l->reducer,8,2);W(l->reducer,0xc,8192);W(l->reducer,8,4);c206_read(l->residual_dma,j->gate_phys,8192);}
  else if(!(R(l->reducer,4)&8)){error=-13;goto failed;}
 }
 if(j->preload_gate){
  for(uint32_t lane=0;lane<2;lane++){
   C206Lane*l=&j->lanes[lane];
   while(!(R(l->residual_dma,4)&0x1000)||!(R(l->reducer,4)&8)){
    if((error=c206_fault(l,0)))goto failed;
    if(NOW()>deadline){error=-1;goto failed;}
   }
  }
 }
 {
  C206Pending pending[3]={{0}};uint32_t end=j->first_row+j->bytes/8192;
  for(uint32_t lane=0;lane<3;lane++)pending[lane].row=c206_next_row(j->first_row,end,lane);
  while(s->chunks<j->bytes/8192){
   for(uint32_t lane=0;lane<3;lane++){
    C206Pending*p=&pending[lane];C206Lane*l=&j->lanes[lane];if(p->row>=end)continue;
    uint64_t off=(uint64_t)(p->row-j->first_row)*8192;
    if(!p->phase){p->id=(j->step<<24)|(j->layer<<16)|(j->operator_id<<14)|p->row;
     c206_arm(l,p,8192,j->local_phys+off,j->state_phys+off,j->group,j->epoch,j->step,j->branch,j->layer,j->operator_id,p->row,lane,j->timeout_ms,0);
    }
    int r=c206_advance(l,p,j->local_phys+off,8192,0);if(r<0){error=r;goto failed;}
    if(r){s->chunks++;s->rx_bytes+=8192;s->tx_bytes+=8192;p->row=c206_next_row(p->row+1,end,lane);}
   }
  }
 }
 s->elapsed_ns=NOW()-start;return 0;
failed:s->elapsed_ns=NOW()-start;s->error=(uint32_t)-error;return error;
}
int cosmos_stream_reduce(C206StreamJob*j,C206StreamStats*s){
 if(!j)return -5;
 return c206_stream_reduce_expected_epoch(j,s,100000U+j->step*131072U);
}
int cosmos_c206_bulk_exchange(C206BulkJob*j,C206StreamStats*s){
 if(!j||!s||!j->timeout_ms||!j->bytes||j->bytes>8192||j->bytes%8||
    !j->tx_phys||!j->rx_phys||j->tx_phys%64||j->rx_phys%64||j->node<1||j->node>4||
    j->lane_id>2||j->peer!=c206_neighbour(j->node,j->lane_id)||j->channel>255)return -5;
 uint64_t start=NOW();s->error=0;s->chunks=0;s->rx_bytes=0;s->tx_bytes=0;
 int error=c206_identity(&j->lane,j->node,j->peer);if(error)goto failed;
 if(!(R(j->lane.link,4)&32)){error=-2;goto failed;}
 {
  C206Pending p={0};p.id=0x80000000U|j->sequence;
  c206_arm(&j->lane,&p,j->bytes,j->tx_phys,j->rx_phys,0x80,j->job_id,j->step,0x80,0,j->channel,j->sequence,j->lane_id,j->timeout_ms,1);
  for(;;){int r=c206_advance(&j->lane,&p,j->tx_phys,j->bytes,1);if(r<0){error=r;goto failed;}if(r)break;}
 }
 s->chunks=1;s->rx_bytes=j->bytes;s->tx_bytes=j->bytes;s->elapsed_ns=NOW()-start;return 0;
failed:s->elapsed_ns=NOW()-start;s->error=(uint32_t)-error;return error;
}
size_t cosmos_stream_job_size(void){return sizeof(C206StreamJob);}
size_t cosmos_stream_stats_size(void){return sizeof(C206StreamStats);}
size_t cosmos_c206_bulk_job_size(void){return sizeof(C206BulkJob);}
