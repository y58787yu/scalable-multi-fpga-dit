/* Real C209 MMIO/DMA executor. The third lane remains raw/control-only. */
#define cosmos_stage cosmos_c206_legacy_stage
#ifndef COSMOS_C209_BIT_ID
#define COSMOS_C209_BIT_ID 0xc2090900U
#endif
#define COSMOS_TP_BIT_ID(node) COSMOS_C209_BIT_ID
#include "executor_c206.c"
#undef cosmos_stage
#include "c209_serial_control.h"
uint32_t cosmos_c209_runtime_identity(void){return 0xc2099001U;}
#include "c209_resident_epoch.h"
uint32_t cosmos_c209_transport_bit_identity(void){return COSMOS_C209_BIT_ID;}
uint32_t cosmos_c209_resident_runtime_identity(void){return C209_RESIDENT_RUNTIME_ID;}
int cosmos_c209_resident_epoch(uint64_t generation,uint32_t *epoch){return c209_resident_epoch_checked(generation,epoch);}
int cosmos_c209_stream_reduce_resident(C206StreamJob*j,C206StreamStats*s,uint32_t generation){
 uint32_t epoch;
 if(COSMOS_C209_BIT_ID!=0xc2090900U||c209_resident_epoch_checked(generation,&epoch))return -5;
 return c206_stream_reduce_expected_epoch(j,s,epoch);
}
