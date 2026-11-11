#pragma once
#ifdef COSMOS_C207
// Each attention slot has one owner until its QK and PV users finish.
// This must precede the first hls_streamofblocks.h include.
#ifndef EXPLICIT_ACQUIRE_RELEASE
#define EXPLICIT_ACQUIRE_RELEASE 1
#endif
#endif
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include "c209_precision.hpp"
#include "c209_tp4_config.hpp"
namespace c2 {
using bus_t=ap_uint<256>; using i16=ap_int<16>; using i48=ap_int<48>;
constexpr unsigned BM=64,BN=128,BK=256,PI=16;
#ifdef COSMOS_C205
#ifndef C206_MAC_PO
#define C206_MAC_PO 32
#endif
constexpr unsigned PO=C206_MAC_PO;
static_assert(PO==16 || PO==32,"Isolated resource candidate supports PO16/PO32 only");
#else
constexpr unsigned PO=64;
#endif
constexpr unsigned GEMM_M=256,MAC_ROWS=256;
#ifdef COSMOS_C207
constexpr unsigned GEMM_N=256,C207_MAC_ROWS=32,C207_MAC_COLS=32;
#else
constexpr unsigned GEMM_N=BN;
#endif
// C207 keeps attention's numerical key block separate from GEMM tiling.
constexpr unsigned C207_KV_KEY_BLOCK=128;
// FFN is the largest rank-local working width, not the model-wide FFN8192.
constexpr unsigned DIM=c209tp4::HIDDEN,FFN=c209tp4::LOCAL_FFN,HEAD_DIM=128,LAYERS=c209tp4::LAYERS,MAX_ROWS=c209tp4::TOKENS;
enum Stage : unsigned { GEMM=1,QUANTIZE=2,LAYER_NORM=3,RMS_NORM=4,
 VECTOR=5,ROPE=6,ATTENTION=7,PACK_BLOCK=8,UNPACK_BLOCK=9,NORM_ADALN_QUANTIZE=10,EXECUTE_PROGRAM=11,PREPARE_KV=12 };
enum Flag : unsigned { BIAS=1,GELU=2,ADD_OUTPUT=4,LOCAL_PARTIAL=8,SKIP_K=16,ZERO_KV_TAIL=32,WEIGHTS_INT8=64,PREQUANT_KV=128 };
enum Vector : unsigned { ADD=0,MUL=1,ADALN=2,GATED_RESIDUAL=3,GELU_TANH=4,SILU=5,COPY=6,SUM_GATED_RESIDUAL=7 };
enum Error : unsigned { OK=0,BAD_STAGE=1,BAD_SHAPE=2,BAD_ALIGN=3,BAD_FLAGS=4,BAD_ABI=5,BAD_FORMAT=6,BAD_ALIAS=7 };
enum TensorFormat : unsigned { UNUSED=0,F16=1,F32=2,I16=3,KV_PACKED=4,U32=5,I8=6 };
constexpr unsigned MODULAR16_HARDWARE_ID=c209tp4::HARDWARE_ID;
struct Job {
 unsigned op,m,n,k,la,lb,lc,heads,flags,mode,k0,kc;
 uint64_t a,b,c,s0,s1,s2,s3; float factor,epsilon;
 // Defaults exist only for direct native mathematical fixtures. The production
 // command path requires a V2 descriptor and overwrites every format field.
 unsigned fa=F32,fb=F32,fc=F32,fs0=F32,fs1=F32,fs2=F32,fs3=F32;
 unsigned abi_version=1;
};
// All addresses are absolute physical bytes; base=0 on hardware. Native tests
// pass a common arena and use byte offsets. Row strides are ELEMENTS.
}
#include "c209_tensor_io.hpp"
#if defined(__SYNTHESIS__) && defined(C209_FIVE_MINIMAL_CSR) && C209_FIVE_MINIMAL_CSR
extern "C" void cosmos2b_tile(c2::bus_t* act,c2::bus_t* weights,c2::bus_t* output,c2::bus_t* aux,
 unsigned rows,uint64_t aux0,unsigned &error,unsigned &completed_tiles,
 unsigned &perf_elapsed_cycles,unsigned &perf_owner_a_cycles,unsigned &perf_owner_b_cycles,
 unsigned &perf_source_empty_cycles,unsigned &perf_result_stall_cycles,
 unsigned &perf_a_commits,unsigned &perf_b_commits,unsigned &perf_no_request_cycles);
#else
extern "C" void cosmos2b_tile(c2::bus_t* act,c2::bus_t* weights,c2::bus_t* output,c2::bus_t* aux,
 unsigned stage,unsigned rows,unsigned cols,unsigned inner,unsigned lda,unsigned ldb,unsigned ldc,
 unsigned heads,unsigned flags,unsigned mode,unsigned k_start,unsigned k_count,
 uint64_t a,uint64_t b,uint64_t c,uint64_t aux0,uint64_t aux1,uint64_t aux2,uint64_t aux3,
 float factor,float epsilon,unsigned &error,unsigned &completed_tiles
#if defined(C209_FIXED_PROFILE) && C209_FIXED_PROFILE
 ,unsigned &perf_elapsed_cycles,unsigned &perf_owner_a_cycles,unsigned &perf_owner_b_cycles,
 unsigned &perf_source_empty_cycles,unsigned &perf_result_stall_cycles,
 unsigned &perf_a_commits,unsigned &perf_b_commits,unsigned &perf_no_request_cycles
#endif
 );
#endif
