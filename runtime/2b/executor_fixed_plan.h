#ifndef EXECUTOR_FIXED_PLAN_H
#define EXECUTOR_FIXED_PLAN_H
/* The existing compute owner and TP worker call opposite ends of this queue.
 * READY means a fresh successful compute DONE, FREE means both DMA rounds
 * completed with byte counts checked. A failure never releases CMA ownership. */
typedef struct {Stats stats;uint32_t cycles[8];uint64_t start_ns,end_ns;} FixedCoreCompletion;
typedef struct {
 uint64_t controller;CoreJob *cores;TP4PairJob *round0,*round1;
 const uint32_t *bases,*counter_regs;FixedCoreCompletion *compute;C206StreamStats *reduce;
 uint64_t *tp_start_ns,*tp_end_ns;uint32_t count,timeout_ms;
 atomic_uint slots[2];atomic_int failed;atomic_uint produced,retired;
} FixedProjection;
static uint64_t fixed_trace_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;}
uint32_t cosmos_fixed_plan_revision(void){return 1U;}
/* 2 = corrected hardware batch controls the existing SFP/PL SUM->FUSED
 * packet pipeline. The native queue still owns all buffer retirement. */
uint32_t cosmos_fixed_tp_control_mode(void){return 5U;}
size_t cosmos_fixed_completion_size(void){return sizeof(FixedCoreCompletion);}
size_t cosmos_fixed_projection_size(void){return sizeof(FixedProjection);}
int cosmos_fixed_stage(CoreJob*j,const uint32_t*regs,FixedCoreCompletion*out){
 if(!j||!regs||!out)return -5;
 for(unsigned i=0;i<8;i++)if(regs[i]<0x10||regs[i]>=0x1000||(regs[i]&3))return -5;
 *out=(FixedCoreCompletion){0};out->start_ns=fixed_trace_ns();
 int e=cosmos_stage(j,&out->stats);
 if(!e){uint64_t sum=0;for(unsigned i=0;i<8;i++){out->cycles[i]=rd(j->kernel,regs[i]);if(i)sum+=out->cycles[i];}
  if(sum!=out->cycles[0])e=-18;
 }
 out->end_ns=fixed_trace_ns();return e;
}
static int fixed_fail(FixedProjection*p,int e){int zero=0;atomic_compare_exchange_strong(&p->failed,&zero,e);return e;}
int cosmos_fixed_project_cancel(FixedProjection*p,int e){return p?fixed_fail(p,e?e:-15):-5;}
static int fixed_wait(FixedProjection*p,unsigned slot,unsigned state){
 uint64_t deadline=NOW()+(uint64_t)p->timeout_ms*1000000ULL;unsigned checks=0;
 for(;;){int e=atomic_load_explicit(&p->failed,memory_order_acquire);if(e)return e;
  if(atomic_load_explicit(&p->slots[slot],memory_order_acquire)==state)return 0;
  if((++checks&63U)==0){if(tp4_is_cancelled())return fixed_fail(p,-15);if(NOW()>deadline)return fixed_fail(p,-1);}
 }
}
static int fixed_plan_valid(FixedProjection*p){return p&&p->count&&p->count<=16&&p->timeout_ms&&p->cores&&p->round0&&p->round1&&p->bases&&p->counter_regs&&p->compute&&p->reduce&&p->tp_start_ns&&p->tp_end_ns;}
int cosmos_fixed_project_compute(FixedProjection*p){
 if(!fixed_plan_valid(p))return p?fixed_fail(p,-5):-5;
 for(unsigned i=0;i<p->count;i++){
  int e=fixed_wait(p,i&1U,0);if(e)return e;
  e=cosmos_fixed_stage(&p->cores[i],p->counter_regs,&p->compute[i]);if(e)return fixed_fail(p,e);
  atomic_store_explicit(&p->produced,i+1,memory_order_release);
  atomic_store_explicit(&p->slots[i&1U],1,memory_order_release);
 }
 return 0;
}
int cosmos_fixed_project_reduce(FixedProjection*p){
 if(!fixed_plan_valid(p))return p?fixed_fail(p,-5):-5;
 for(unsigned i=0;i<p->count;i++){
  int e=fixed_wait(p,i&1U,1);if(e)return e;
  p->tp_start_ns[i]=fixed_trace_ns();
  /* The preceding batch must release its lane registers before START. */
  if(!p->controller||(R(p->controller,4)&1U))return fixed_fail(p,-4);
  e=cosmos_tp4_pair_batch(p->controller,&p->round0[i],&p->round1[i],p->bases,&p->reduce[2*i]);
  p->tp_end_ns[i]=fixed_trace_ns();if(e)return fixed_fail(p,e);
  uint32_t bytes=p->round0[i].rows*p->round0[i].cols*4U;
  C206StreamStats*z=&p->reduce[2*i],*s=&p->reduce[2*i+1];
  if(z->error||z->tx_bytes||z->rx_bytes||z->chunks||s->error||s->tx_bytes!=bytes||s->rx_bytes!=bytes||s->chunks!=p->round1[i].rows)return fixed_fail(p,-19);
  atomic_store_explicit(&p->retired,i+1,memory_order_release);
  atomic_store_explicit(&p->slots[i&1U],0,memory_order_release);
 }
 return 0;
}
#endif
