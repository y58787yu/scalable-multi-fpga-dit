// Fixed apple / 2 or4 identical TP2 boards / five physical256 INT16 shards.
// All physical MAC state stays in the explicit300MHz RTL island. This top
// consumes only FIVE_FIXED64_V1 phase records; no arbitrary shapes/formats.
#define C209_FIVE_MINIMAL_CSR 1
#include "c209_five_profile.hpp"
#include "c209_five_auxiliary.hpp"
#include "c209_five_gemm.hpp"
#include "c209_five_attention.hpp"
#include "c209_five_mac_rtl.hpp"
#ifndef C209_FIXED_PROFILE
#define C209_FIXED_PROFILE 0
#endif
static_assert(C209_FIXED_PROFILE==1,"Use the explicit fixed-profile build");

namespace c209five {
static void supervisor(bus_t *act,bus_t *weights,bus_t *output,bus_t *aux,
 unsigned stage,unsigned count,unsigned rank,unsigned mode,unsigned flags,uint64_t program,
 CmdStream &commands,C209_FIVE_OPERAND_PARAMS(left),C209_FIVE_OPERAND_PARAMS(right),C209_FIVE_RESULT_PARAMS(results),
 unsigned &error,unsigned &tiles,c2fast::gelu_command_stream &gc,
 c2fast::nonlinear_pair_stream &gi,c2fast::pair_output_stream &go,c2fast::gelu_ack_stream &ga,
 AuxQuantRequests &qreq,AuxQuantResults &qres) {
#pragma HLS INLINE off
 AuxRow workspace;
#pragma HLS ARRAY_PARTITION variable=workspace cyclic factor=8
#pragma HLS BIND_STORAGE variable=workspace type=ram_2p impl=bram
 error=OK;tiles=0;
 // Register the launch envelope before the descriptor/AXI control phase.
 // Host-owned fixed pool binding performs full range/alias admission once.
 FixedLaunch launch=admit_fixed_launch(stage,count,rank,mode,flags,program);
 error=launch.error;
 for(ap_uint<8> index=0;index<launch.jobs&&error==OK;index++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
  FixedJob j; c209fixed::Kind kind;unsigned completed=0;
  error=decode_fixed(aux,launch.program,index.to_uint(),j,kind);
  if(error!=OK)break;
  switch(j.op) {
   case GEMM:case GEMM_GELU_F16:case GEMM_F16_A_PARTIAL:
    error=gemm_ddr(act,weights,output,aux,j,commands,C209_FIVE_STREAM_ARGS(left),C209_FIVE_STREAM_ARGS(right),C209_FIVE_STREAM_ARGS(results),completed,gc,gi,go,ga);break;
   case ATTENTION:error=attention_ddr(act,weights,output,aux,j,commands,C209_FIVE_STREAM_ARGS(left),C209_FIVE_STREAM_ARGS(right),C209_FIVE_STREAM_ARGS(results),completed,qreq,qres);break;
   case QUANTIZE:case NORM_ADALN_FIXED:
    error=quantize_row_service(act,output,aux,j,workspace,completed,qreq,qres);break;
   case RMS_NORM:case RMS_ROPE_FUSED:
    error=rms128_service(act,output,aux,j,completed);break;
   case PREPARE_KV_GLOBAL:error=prepare_global(act,weights,output,aux,j,completed,qreq,qres);break;
   default:error=BAD_STAGE;break;
  }
  tiles+=completed;
 }
 commands.write(make_command(0,0,0,false,false,0,false,true));
 c2fast::GeluCommand stop={0,0,true};gc.write(stop);
 AuxQuantRequest qstop=0;qstop[96]=1;qreq.write(qstop);
}
static void metrics(PerfStream &perf,unsigned &elapsed,unsigned &multiply,unsigned &load_a,
 unsigned &load_b,unsigned &seed,unsigned &store,unsigned &output_blocked,unsigned &other) {
#pragma HLS INLINE off
 ap_uint<256> p=perf.read();
 elapsed=p.range(31,0);multiply=p.range(63,32);load_a=p.range(95,64);load_b=p.range(127,96);
 seed=p.range(159,128);store=p.range(191,160);output_blocked=p.range(223,192);other=p.range(255,224);
}
} // namespace c209five

#if defined(__SYNTHESIS__) && C209_FIVE_MINIMAL_CSR
extern "C" void cosmos2b_tile(bus_t*act,bus_t*weights,bus_t*output,bus_t*aux,
 unsigned rows,uint64_t aux0,unsigned&error,unsigned&completed_tiles,
 unsigned &perf_elapsed_cycles,unsigned &perf_owner_a_cycles,unsigned &perf_owner_b_cycles,
 unsigned &perf_source_empty_cycles,unsigned &perf_result_stall_cycles,
 unsigned &perf_a_commits,unsigned &perf_b_commits,unsigned &perf_no_request_cycles){
#else
extern "C" void cosmos2b_tile(bus_t*act,bus_t*weights,bus_t*output,bus_t*aux,
 unsigned stage,unsigned rows,unsigned cols,unsigned inner,unsigned lda,unsigned ldb,unsigned ldc,
 unsigned heads,unsigned flags,unsigned mode,unsigned k_start,unsigned k_count,
 uint64_t a,uint64_t b,uint64_t c,uint64_t aux0,uint64_t aux1,uint64_t aux2,uint64_t aux3,
 float factor,float epsilon,unsigned&error,unsigned&completed_tiles,
 unsigned &perf_elapsed_cycles,unsigned &perf_owner_a_cycles,unsigned &perf_owner_b_cycles,
 unsigned &perf_source_empty_cycles,unsigned &perf_result_stall_cycles,
 unsigned &perf_a_commits,unsigned &perf_b_commits,unsigned &perf_no_request_cycles){
#endif
#pragma HLS INTERFACE m_axi port=act offset=slave bundle=act depth=32768 max_widen_bitwidth=256 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=weights depth=32768 max_widen_bitwidth=256 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=output depth=32768 max_widen_bitwidth=256 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=aux offset=slave bundle=aux depth=32768 max_widen_bitwidth=256 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=act bundle=control offset=0x010
#pragma HLS INTERFACE s_axilite port=weights bundle=control offset=0x01c
#pragma HLS INTERFACE s_axilite port=output bundle=control offset=0x028
#pragma HLS INTERFACE s_axilite port=aux bundle=control offset=0x034
#pragma HLS INTERFACE s_axilite port=rows bundle=control offset=0x048
#pragma HLS INTERFACE s_axilite port=aux0 bundle=control offset=0x0c4
#pragma HLS INTERFACE s_axilite port=error bundle=control offset=0x104
#pragma HLS INTERFACE s_axilite port=completed_tiles bundle=control offset=0x114
#pragma HLS INTERFACE s_axilite port=perf_elapsed_cycles bundle=control offset=0x124
#pragma HLS INTERFACE s_axilite port=perf_owner_a_cycles bundle=control offset=0x134
#pragma HLS INTERFACE s_axilite port=perf_owner_b_cycles bundle=control offset=0x144
#pragma HLS INTERFACE s_axilite port=perf_source_empty_cycles bundle=control offset=0x154
#pragma HLS INTERFACE s_axilite port=perf_result_stall_cycles bundle=control offset=0x164
#pragma HLS INTERFACE s_axilite port=perf_a_commits bundle=control offset=0x174
#pragma HLS INTERFACE s_axilite port=perf_b_commits bundle=control offset=0x184
#pragma HLS INTERFACE s_axilite port=perf_no_request_cycles bundle=control offset=0x194
 c209five::CmdStream commands;
 // The importer follows aliases to their original objects. These are actual
 // independent scalar FIFOs, never elements or aliases of a stream array.
 c209five::OperandStream left0;
 c209five::OperandStream left1;
 c209five::OperandStream left2;
 c209five::OperandStream left3;
 c209five::OperandStream left4;
 c209five::OperandStream right0;
 c209five::OperandStream right1;
 c209five::OperandStream right2;
 c209five::OperandStream right3;
 c209five::OperandStream right4;
 c209five::ResultStream results0;
 c209five::ResultStream results1;
 c209five::ResultStream results2;
 c209five::ResultStream results3;
 c209five::ResultStream results4;
 c209five::PerfStream perf;
 c209five::AuxQuantRequests qreq;c209five::AuxQuantResults qres;
 c2fast::gelu_command_stream gc;c2fast::nonlinear_pair_stream gi;
 c2fast::pair_output_stream go;c2fast::gelu_ack_stream ga;
#pragma HLS STREAM variable=commands depth=4
#pragma HLS STREAM variable=left0 depth=16
#pragma HLS STREAM variable=left1 depth=16
#pragma HLS STREAM variable=left2 depth=16
#pragma HLS STREAM variable=left3 depth=16
#pragma HLS STREAM variable=left4 depth=16
#pragma HLS STREAM variable=right0 depth=16
#pragma HLS STREAM variable=right1 depth=16
#pragma HLS STREAM variable=right2 depth=16
#pragma HLS STREAM variable=right3 depth=16
#pragma HLS STREAM variable=right4 depth=16
#pragma HLS STREAM variable=results0 depth=4
#pragma HLS STREAM variable=results1 depth=4
#pragma HLS STREAM variable=results2 depth=4
#pragma HLS STREAM variable=results3 depth=4
#pragma HLS STREAM variable=results4 depth=4
#pragma HLS STREAM variable=qreq depth=16
#pragma HLS STREAM variable=qres depth=16
#pragma HLS STREAM variable=perf depth=2
#pragma HLS STREAM variable=gc depth=2
#pragma HLS STREAM variable=gi depth=16
#pragma HLS STREAM variable=go depth=16
#pragma HLS STREAM variable=ga depth=2
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
#ifdef __SYNTHESIS__
 C209_FIXED_TASK(c209five::supervisor(act,weights,output,aux,c2::EXECUTE_PROGRAM,rows,1,c209tp4::HARDWARE_ID,0,aux0,
  commands,C209_FIVE_STREAM_ARGS(left),C209_FIVE_STREAM_ARGS(right),C209_FIVE_STREAM_ARGS(results),error,completed_tiles,gc,gi,go,ga,qreq,qres));
#else
 C209_FIXED_TASK(c209five::supervisor(act,weights,output,aux,stage,rows,heads,mode,flags,aux0,
  commands,C209_FIVE_STREAM_ARGS(left),C209_FIVE_STREAM_ARGS(right),C209_FIVE_STREAM_ARGS(results),error,completed_tiles,gc,gi,go,ga,qreq,qres));
#endif
 C209_FIXED_TASK(c209_five_mac_rtl(commands,left0,right0,results0,left1,right1,results1,
  left2,right2,results2,left3,right3,results3,left4,right4,results4,perf));
 // CSR slot order is stable; semantics are FAST_GROUP_V1, not old A/B-owner
 // counters: elapsed,multiply,loadA,loadB,seed,store,outputBlocked,other.
 C209_FIXED_TASK(c209five::metrics(perf,perf_elapsed_cycles,perf_owner_a_cycles,perf_owner_b_cycles,
  perf_source_empty_cycles,perf_result_stall_cycles,perf_a_commits,perf_b_commits,perf_no_request_cycles));
 C209_FIXED_TASK(c2fast::shared_gelu_service(gc,gi,go,ga));
 C209_FIXED_TASK(c209five::aux_quant4_service(qreq,qres));
}
