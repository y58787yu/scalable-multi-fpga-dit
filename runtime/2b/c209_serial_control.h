#ifndef C209_SERIAL_CONTROL_H
#define C209_SERIAL_CONTROL_H
/* C209 control only: physical IDLE admission first, then one-use fresh DONE.
 * No general ctrl==0 acceptance. No relaxed physical quiescence/CMA release.
 * CoreJob/Stats ABI and all compute/settings/communication semantics unchanged.
 */
#include <stdatomic.h>
#include <unistd.h>
#ifndef C209_SERIAL_CANCEL_REQUESTED
#define C209_SERIAL_CANCEL_REQUESTED() 0
#endif
typedef struct {
 uint64_t kernel,overlay_epoch,serial,completed_serial,owner_pid;
 uint32_t certificate_valid,busy,before_ctrl,completion_ctrl,admission;
 int32_t result_code;
 uint32_t error,tiles;
} C209SerialState;
static C209SerialState c209_serial_state;
static atomic_flag c209_serial_guard=ATOMIC_FLAG_INIT;
#ifdef C209_SERIAL_MMIO_TEST
extern uint32_t c209_serial_test_read(uint64_t,uint32_t);
extern void c209_serial_test_write(uint64_t,uint32_t,uint32_t);
extern uint64_t c209_serial_test_now(void);
#define DREAD c209_serial_test_read
#define DWRITE c209_serial_test_write
#define DNOW c209_serial_test_now
#else
#define DREAD rd
#define DWRITE wr
#define DNOW now_ns
#endif
uint32_t cosmos_c209_serial_identity(void){return 0xc209c004U;}
size_t cosmos_c209_serial_state_size(void){return sizeof(C209SerialState);}
int cosmos_c209_serial_snapshot(C209SerialState*out){
 if(!out)return -5;
 if(atomic_flag_test_and_set_explicit(&c209_serial_guard,memory_order_acquire))return -14;
 *out=c209_serial_state;
 atomic_flag_clear_explicit(&c209_serial_guard,memory_order_release);return 0;
}
int cosmos_c209_serial_reset(uint64_t kernel,uint64_t epoch){
 if(!kernel||!epoch)return -5;
 if(atomic_flag_test_and_set_explicit(&c209_serial_guard,memory_order_acquire))return -14;
 C209SerialState zero={0};c209_serial_state=zero;
 c209_serial_state.kernel=kernel;c209_serial_state.overlay_epoch=epoch;c209_serial_state.owner_pid=(uint64_t)getpid();
 atomic_flag_clear_explicit(&c209_serial_guard,memory_order_release);return 0;
}
int cosmos_c209_serial_invalidate(uint64_t kernel,uint64_t epoch){
 if(atomic_flag_test_and_set_explicit(&c209_serial_guard,memory_order_acquire))return -14;
 c209_serial_state.certificate_valid=0;
 int code=(kernel==c209_serial_state.kernel&&epoch==c209_serial_state.overlay_epoch&&c209_serial_state.owner_pid==(uint64_t)getpid())?0:-4;
 atomic_flag_clear_explicit(&c209_serial_guard,memory_order_release);return code;
}
int cosmos_stage(CoreJob*j,Stats*s){
 if(atomic_flag_test_and_set_explicit(&c209_serial_guard,memory_order_acquire))return -14;
 int code=-5;uint64_t begin=0;uint32_t permit=c209_serial_state.certificate_valid;
 /* Consume before every validation/read/write; any rejected call invalidates. */
 c209_serial_state.certificate_valid=0;c209_serial_state.busy=1;
 c209_serial_state.admission=0;c209_serial_state.before_ctrl=0;c209_serial_state.completion_ctrl=0;
 c209_serial_state.error=0;c209_serial_state.tiles=0;
 if(s){Stats zero={0};*s=zero;}
 if(!j||!s||!j->settings||!j->count||j->count>64||!j->timeout_ms)goto finished;
 if(C209_SERIAL_CANCEL_REQUESTED()){code=-15;goto finished;}
 if(c209_serial_state.owner_pid!=(uint64_t)getpid()||j->kernel!=c209_serial_state.kernel||!c209_serial_state.overlay_epoch){code=-4;goto finished;}
 for(uint32_t i=0;i<j->count;i++){uint32_t o=j->settings[i].offset;if(o<0x10||o>=0x1000||o%4)goto finished;}
 uint32_t control=DREAD(j->kernel,0);c209_serial_state.before_ctrl=control;
 /* Reject pending START, auto-restart, and stale DONE; never recycle a latch. */
 if(control&0x83U){code=-4;goto finished;}
 if(!c209_serial_state.serial){if(!(control&4U)||permit){code=-4;goto finished;}c209_serial_state.admission=1;}
 else {if(!permit||c209_serial_state.completed_serial!=c209_serial_state.serial){code=-4;goto finished;}c209_serial_state.admission=2;}
 c209_serial_state.serial++;
 for(uint32_t i=0;i<j->count;i++)DWRITE(j->kernel,j->settings[i].offset,j->settings[i].value);
 begin=DNOW();DWRITE(j->kernel,0,1);
 uint64_t deadline=begin+(uint64_t)j->timeout_ms*1000000ULL;unsigned checks=0;
 for(;;){
  uint32_t value=DREAD(j->kernel,0);
  if(value&2U){c209_serial_state.completion_ctrl=value;break;}
  if((++checks&63U)==0){
   if(C209_SERIAL_CANCEL_REQUESTED()){code=-15;goto finished;}
   if(DNOW()>deadline){code=-1;goto finished;}
  }
 }
 s->elapsed_ns=DNOW()-begin;
 s->error=DREAD(j->kernel,j->error_reg);s->tiles=DREAD(j->kernel,j->tiles_reg);
 c209_serial_state.error=s->error;c209_serial_state.tiles=s->tiles;
 if(c209_serial_state.completion_ctrl&0x81U){code=-4;goto finished;}
 if(s->error){code=-6;goto finished;}
 c209_serial_state.completed_serial=c209_serial_state.serial;c209_serial_state.certificate_valid=1;code=0;
finished:
 if(s&&begin&&code){s->elapsed_ns=DNOW()-begin;if(code==-1)s->error=0xffffffffU;}
 c209_serial_state.result_code=code;c209_serial_state.busy=0;
 atomic_flag_clear_explicit(&c209_serial_guard,memory_order_release);return code;
}
#endif


