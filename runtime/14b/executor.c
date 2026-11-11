#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <time.h>

// Values are coarse-stage settings assembled from the ACTUAL generated HLS
// register header. No arithmetic, tensor copies or per-microtile host launches.
typedef struct {uint32_t offset,value;} RegWrite;
typedef struct {uint64_t kernel;const RegWrite *settings;uint32_t count,error_reg,tiles_reg,timeout_ms;} CoreJob;
typedef struct {uint64_t dma,link,tx_phys,rx_phys;uint32_t bytes,count,transaction,meta,timeout_ms;} ExchangeJob;
typedef struct {uint64_t elapsed_ns;uint32_t error,tiles,rx_bytes,rx_errors,tx_frames,rx_frames;} Stats;
static uint64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;}
static uint32_t rd(uint64_t base,uint32_t off){return ((volatile uint32_t*)(uintptr_t)base)[off/4];}
static void wr(uint64_t base,uint32_t off,uint32_t v){((volatile uint32_t*)(uintptr_t)base)[off/4]=v;}
static void addr(uint64_t base,uint32_t off,uint64_t v){wr(base,off,(uint32_t)v);wr(base,off+4,(uint32_t)(v>>32));}
static uint64_t tx_begin(uint64_t link){uint32_t hi=rd(link,0x6c),lo=rd(link,0x68);return ((uint64_t)hi<<32)|lo;}
static int wait_new_tx(uint64_t link,uint64_t before,uint32_t ms){uint64_t end=now_ns()+(uint64_t)ms*1000000ULL;unsigned checks=0;for(;;){if(tx_begin(link)!=before)return 0;if((++checks&63)==0&&now_ns()>end)return -11;}}
static int wait_bits(uint64_t base,uint32_t off,uint32_t mask,uint32_t value,uint32_t faults,uint32_t ms){
 uint64_t end=now_ns()+(uint64_t)ms*1000000ULL;unsigned checks=0;
 for(;;){uint32_t v=rd(base,off);if(v&faults)return -2;if((v&mask)==value)return 0;if((++checks&63)==0&&now_ns()>end)return -1;}
}
int cosmos_stage(CoreJob*j,Stats*s){
 if(!j||!s||!j->settings||j->count>64||!j->timeout_ms)return -5;
 if(!(rd(j->kernel,0)&4))return -4;
 for(uint32_t i=0;i<j->count;i++){uint32_t o=j->settings[i].offset;if(o<0x10||o>=0x1000||o%4)return -5;wr(j->kernel,o,j->settings[i].value);}
 uint64_t t=now_ns();wr(j->kernel,0,1);int e=wait_bits(j->kernel,0,2,2,0,j->timeout_ms);
 s->elapsed_ns=now_ns()-t;s->error=e?0xffffffff:rd(j->kernel,j->error_reg);s->tiles=e?0:rd(j->kernel,j->tiles_reg);
 return e?e:(s->error?-6:0);
}
static int send_control(uint64_t link,uint32_t kind,uint32_t id,uint32_t ms){
 int e=wait_bits(link,4,1,0,8,ms);if(e)return e;
 uint64_t before=tx_begin(link);
 wr(link,0x10,id);wr(link,0x14,0);wr(link,0x18,kind);wr(link,0x1c,0);wr(link,4,16);
 e=wait_new_tx(link,before,ms);if(e)return e;
 return wait_bits(link,4,3,2,8,ms);
}
static int wait_credit(uint64_t link,uint32_t id,uint32_t ms){
 uint64_t end=now_ns()+(uint64_t)ms*1000000ULL;unsigned checks=0;
 for(;;){
  if(rd(link,0x4c)||rd(link,0x50))return -3;
  if(rd(link,0x80)){uint32_t kind=rd(link,0x30),got=rd(link,0x34);if(kind!=2||got!=id)return -7;wr(link,0x84,1);return 0;}
  if((++checks&63)==0&&now_ns()>end)return -1;
 }
}
int cosmos_exchange(ExchangeJob*j,Stats*s){
 if(!j||!s||!j->bytes||j->bytes>=0x800000||j->bytes%8||j->tx_phys%64||j->rx_phys%64||!j->timeout_ms)return -5;
 // Arm S2MM and the entire receive transaction BEFORE advertising credit.
 wr(j->dma,0x34,0x1000);addr(j->dma,0x48,j->rx_phys);wr(j->dma,0x58,j->bytes);
 wr(j->link,0x20,j->transaction);wr(j->link,0x24,j->bytes);wr(j->link,0x28,j->meta);wr(j->link,0x2c,j->count);wr(j->link,4,4);
 uint64_t t=now_ns();int e=send_control(j->link,2,j->transaction,j->timeout_ms);if(e)return e;
 e=wait_credit(j->link,j->transaction,j->timeout_ms);if(e)return e;
 uint64_t before=tx_begin(j->link);
 wr(j->link,0x10,j->transaction);wr(j->link,0x14,j->bytes);wr(j->link,0x18,j->meta);wr(j->link,0x1c,j->count);wr(j->link,4,1);
 e=wait_new_tx(j->link,before,j->timeout_ms);if(e)return e;
 wr(j->dma,4,0x1000);addr(j->dma,0x18,j->tx_phys);wr(j->dma,0x28,j->bytes);
 e=wait_bits(j->dma,4,2,2,0x70,j->timeout_ms);if(e)return e;
 e=wait_bits(j->link,4,1,0,8,j->timeout_ms);if(e)return e;
 e=wait_bits(j->link,4,4,4,8,j->timeout_ms);if(e)return e;
 e=wait_bits(j->dma,0x34,2,2,0x70,j->timeout_ms);if(e)return e;
 s->elapsed_ns=now_ns()-t;s->rx_bytes=rd(j->link,0x40);s->rx_errors=rd(j->link,0x4c);s->tx_frames=rd(j->link,0x48);s->rx_frames=rd(j->link,0x44);
 if(s->rx_bytes!=j->bytes||s->rx_errors||rd(j->link,0x50))return -3;
 return 0; // Caller next launches FPGA VECTOR/ADD, never a CPU tensor sum.
}
size_t cosmos_core_job_size(void){return sizeof(CoreJob);}
size_t cosmos_exchange_job_size(void){return sizeof(ExchangeJob);}
size_t cosmos_stats_size(void){return sizeof(Stats);}
