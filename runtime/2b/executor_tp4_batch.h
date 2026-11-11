/* Fixed FP32 CSR batch controller. Uses original SUM/FUSED payload units.
 * ERROR/ABORT never means DMA drained. Caller poisons the hardware owner and
 * retains CMA allocations until its existing owned reset proves quiescence. */
#ifndef C209_EXECUTOR_TP4_BATCH_H
#define C209_EXECUTOR_TP4_BATCH_H
uint32_t cosmos_tp4_batch_revision(void){return 2U;}
int cosmos_tp4_batch_busy(uint64_t control){return control?(R(control,4)&1U)!=0:-1;}
int cosmos_tp4_pair_batch(uint64_t control,TP4PairJob*sum,TP4PairJob*fused,
 const uint32_t*bases,C206StreamStats*stats){
 if(!stats)return -5;
 stats[0]=(C206StreamStats){0};stats[1]=(C206StreamStats){0};
 uint64_t start=NOW();int error=-5;
 if(!control||!bases||!sum||!tp4_pipeline_job_valid(fused,1))goto done;
 if(sum->local_phys!=fused->local_phys||sum->rows!=fused->rows||sum->cols!=fused->cols||
    sum->first_row!=fused->first_row||sum->node!=fused->node||sum->epoch!=fused->epoch||
    sum->step!=fused->step||sum->condition!=fused->condition||sum->layer!=fused->layer||
    sum->operator_id!=fused->operator_id||fused->timeout_ms>UINT32_MAX/1000U)goto done;
 {uint64_t n=(uint64_t)fused->rows*fused->cols*4;
  uint64_t starts[3]={fused->local_phys,fused->output_phys,fused->residual_phys};
  for(unsigned k=0;k<3;k++)if(starts[k]>UINT64_MAX-n)goto done;
  for(unsigned k=0;k<2;k++){uint64_t a=starts[k?2:0],b=fused->output_phys;
   if(a!=b&&a<b+n&&b<a+n)goto done;}
 }
 for(unsigned k=3;k<7;k++)if(!bases[k]||(bases[k]&4095U))goto done;
 if(tp4_is_cancelled()){error=-15;goto done;}
 if(R(control,0)!=0xC209B201U){error=-12;goto done;}
 if(R(control,4)&1U){error=-4;goto done;}
 W(control,8,2);W(control,0x14,sum->rows);W(control,0x18,sum->first_row);W(control,0x1c,sum->node);
 W(control,0x20,sum->epoch);W(control,0x24,sum->step);W(control,0x28,sum->condition);
 W(control,0x2c,sum->layer);W(control,0x30,sum->operator_id);W(control,0x34,sum->timeout_ms*1000U);
 c206_addr(control,0x40,sum->local_phys);c206_addr(control,0x48,0);
 c206_addr(control,0x50,fused->residual_phys);c206_addr(control,0x58,fused->output_phys);c206_addr(control,0x60,fused->gate_phys);
 W(control,0x70,bases[0]);W(control,0x74,bases[1]);W(control,0x78,bases[2]);
 W(control,0x80,bases[3]);W(control,0x84,bases[4]);W(control,0x88,bases[5]);W(control,0x8c,bases[6]);
 atomic_thread_fence(memory_order_seq_cst);W(control,8,1);
 /* Each packet has its original hardware deadline. A whole tile gets the
  * sum of packet deadlines; cancellation is responsive without polling or
  * writing any target DMA/link/reducer register while the master owns them. */
 uint64_t end=start+(uint64_t)sum->timeout_ms*1000000ULL*(sum->rows+3ULL);
 int cancelling=0,first_error=0;uint64_t cancel_end=0,next_poll=0;
 for(;;){
  uint64_t now=NOW();if(now<next_poll)continue;next_poll=now+50000ULL;
  uint32_t status=R(control,4);
  /* A lane watchdog may already have failed while the CSR master is still
   * waiting for an AXI-Lite response. BUSY must stay set in that case, but
   * waiting for the whole-tile deadline would hide a known fault for hours.
   * Preserve the first observed fault/cancel even if normal DONE races it. */
  if((status&4U)&&!first_error){uint32_t e=R(control,12);first_error=e?-(int)e:-3;}
  if(!first_error&&tp4_is_cancelled())first_error=-15;
  if(!first_error&&now>end)first_error=-1;
  if(!(status&1U)&&((status&2U)||first_error)){
   if(first_error)error=first_error;
   else if(!(status&8U)||R(control,0x38)!=0U||R(control,0x3c)!=sum->rows)error=-3;
   else error=0;
   break;
  }
  if(first_error&&!cancelling){
   W(control,8,4);cancelling=1;cancel_end=now+1000000000ULL;
  }
  if(cancelling&&now>=cancel_end){error=first_error;break;} /* A still-owned master retains all DMA allocations. */
 }
 if(!error){
  uint32_t n=sum->rows*sum->cols*4U;
  stats[1].chunks=sum->rows;stats[1].tx_bytes=n;stats[1].rx_bytes=n;stats[1].elapsed_ns=(uint64_t)R(control,0x98)*1000ULL;
 }
done:
 for(unsigned lane=0;lane<2;lane++){if(error)stats[lane].elapsed_ns=NOW()-start;stats[lane].error=error?(uint32_t)-error:0;}
 return error;
}
#endif
