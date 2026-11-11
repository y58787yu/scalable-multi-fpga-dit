#pragma once
#include "c209_five_protocol.hpp"
#include "c209_five_fixed_job.hpp"
#include "c209_fixed_profile.hpp"
// Two AXI256 beats per fixed task. Shapes, types, scalar constants and strides
// are decoded here; they never travel as arbitrary runtime descriptor fields.
namespace c209five {
using namespace c2;
constexpr unsigned RECORD_BYTES=64,MAX_JOBS=128,RECORD_MAGIC=0xC5;
struct FixedLaunch { uint64_t program;ap_uint<8> jobs;unsigned error; };
static FixedLaunch admit_fixed_launch(unsigned stage,unsigned count,unsigned rank,
 unsigned mode,unsigned flags,uint64_t program) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY min=2
 FixedLaunch result={program,ap_uint<8>(count),OK};
 // FixedProgramCache and Board.program admit this envelope before START.
 // The production PL retains bounded8-bit dispatch and protocol completion,
 // rather than reimplementing the host ABI/range checker in every launch.
#ifndef __SYNTHESIS__
 if(stage!=EXECUTE_PROGRAM||mode!=c209tp4::HARDWARE_ID||rank<1||rank>4)result.error=BAD_ABI;
 else if(flags||count==0||count>MAX_JOBS||(program&31ull))result.error=BAD_SHAPE;
#endif
 return result;
}
static unsigned decode_fixed(bus_t *memory,uint64_t address,unsigned index,FixedJob &j,c209fixed::Kind &kind) {
#pragma HLS INLINE off
 const WordOffset record_word=WordOffset(index)<<1;
 const uint64_t base=word_address(address,record_word);
 bus_t w0=memory[base],w1=memory[base+1];ap_uint<64> header=w0.range(63,0);
#ifndef __SYNTHESIS__
 if(header.range(63,32)!=0 || header.range(31,24)!=RECORD_MAGIC || header.range(23,18)!=0)return BAD_ABI;
#endif
 unsigned op=header.range(7,0),shape=header.range(11,8),rc=header.range(13,12);
#ifndef __SYNTHESIS__
 if(op!=GEMM&&op!=QUANTIZE&&op!=RMS_NORM&&op!=ATTENTION&&op!=PREPARE_KV_GLOBAL&&
    op!=NORM_ADALN_FIXED&&op!=GEMM_GELU_F16&&op!=GEMM_F16_A_PARTIAL&&op!=RMS_ROPE_FUSED)return BAD_STAGE;
#endif
 unsigned group=header.range(15,14);bool negative=header[16];
 const bool packed_b=header[17];
 const bool matrix=op==GEMM||op==GEMM_GELU_F16||op==GEMM_F16_A_PARTIAL;
 // New hardware identity uses one physical packed-I16 layout only.
 if(packed_b!=matrix)return BAD_FLAGS;
 // These legacy combinations are absent from the fixed SP3 schedule: FFN
 // always uses fused15/16, and standalone quantization is PATCH or ATTN1280.
 if((op==GEMM&&(shape==3||shape==4))||(op==QUANTIZE&&shape==2))return BAD_STAGE;
 unsigned rows=rc==0?128u:rc==1?256u:rc==2?1280u:1408u;
#ifndef __SYNTHESIS__
 if(group>2 || (op!=PREPARE_KV_GLOBAL&&group) || (negative&&!(op==ATTENTION&&shape==1)))return BAD_FLAGS;
 if((op==GEMM&&shape>5)||(op==QUANTIZE&&shape>2)||(op==ATTENTION&&shape>1)||
    (op!=GEMM&&op!=QUANTIZE&&op!=ATTENTION&&shape))return BAD_SHAPE;
#endif
 j=FixedJob();kind=c209fixed::INVALID;j.op=op;j.m=rows;j.abi_version=2;j.epsilon=1e-6f;
 j.fa=j.fb=j.fc=j.fs0=j.fs1=j.fs2=j.fs3=UNUSED;
 j.a=w0.range(127,64);j.b=w0.range(191,128);j.c=w0.range(255,192);
 j.s0=w1.range(63,0);j.s1=w1.range(127,64);j.s2=w1.range(191,128);j.s3=w1.range(255,192);
 if(op==GEMM||op==GEMM_GELU_F16||op==GEMM_F16_A_PARTIAL) {
  if(op==GEMM_GELU_F16)shape=3;
  if(op==GEMM_F16_A_PARTIAL)shape=4;
  bool partial=shape==2||shape==4;
#ifndef __SYNTHESIS__
  if(partial?(rc>1):(rc<2))return BAD_SHAPE;
#endif
  j.flags=64u|(partial?unsigned(LOCAL_PARTIAL):0u);j.fa=op==GEMM_F16_A_PARTIAL?F16:I16;j.fb=I16;
  j.fc=partial||shape==0||shape==5?F32:F16;j.fs0=j.fs1=F32;
  j.n=shape==1?1280u:shape==5?64u:5120u;
  j.k=shape==0?80u:shape==2?1280u:5120u;j.la=j.lb=j.k;j.lc=j.n;
  kind=shape==0?c209fixed::PATCH:shape==1?c209fixed::PROJECTION:shape==2?c209fixed::ATTENTION_PARTIAL:
       shape==3?c209fixed::FFN_UP:shape==4?c209fixed::FFN_PARTIAL:c209fixed::FINAL_OUTPUT;
  if(op==GEMM_GELU_F16)j.fs2=F32;
 } else {
#ifndef __SYNTHESIS__
  if(rc<2)return BAD_SHAPE;
#endif
  switch(op) {
   case QUANTIZE:
    j.n=shape==0?68u:1280u;j.la=j.lc=shape==0?80u:j.n.to_uint();
    j.fa=shape==0?F32:F16;j.fc=I16;j.fs0=F32;
    kind=shape==0?c209fixed::QUANT_PATCH:c209fixed::QUANT_PROJECTION;break;
   case RMS_NORM:
    j.m=rows*10;j.n=j.la=j.lc=128;j.fa=j.fc=F16;j.fs0=F32;kind=c209fixed::RMS128;break;
   case RMS_ROPE_FUSED:
    j.n=j.la=j.lc=1280;j.heads=10;j.fa=j.fc=F16;j.fs0=j.fs1=j.fs2=F32;
    kind=c209fixed::ROTARY;break;
   case NORM_ADALN_FIXED:
    j.n=j.la=j.lc=5120;j.fa=F32;j.fc=I16;j.fs0=U32;j.fs1=j.fs2=j.fs3=F32;
    kind=c209fixed::NORM_ADALN;break;
   case PREPARE_KV_GLOBAL:
    j.n=4096u;j.kc=rows;j.heads=10;
#ifndef __SYNTHESIS__
    if(rows!=(group==2?1280u:1408u))return BAD_SHAPE;
#endif
    j.k0=group==0?0u:group==1?1408u:2816u;
    j.la=j.lb=j.lc=1280;j.fa=j.fb=F16;j.fc=j.fs2=KV_PACKED;j.fs0=j.fs1=F32;
    kind=c209fixed::SELF_PREPARE;break;
   case ATTENTION:
    j.n=shape?512u:4096u;j.kc=shape?(negative?11u:16u):4096u;j.heads=10;
    j.la=j.lb=j.lc=1280;j.flags=PREQUANT_KV|(shape?unsigned(ZERO_KV_TAIL):0u);
    j.fa=j.fc=F16;j.fb=j.fs2=KV_PACKED;j.fs0=j.fs1=F32;
    kind=shape?c209fixed::CROSS_ATTENTION:c209fixed::SELF_ATTENTION;break;
   default:return BAD_STAGE;
  }
 }
 // Fixed slots; shape, format and alignment admission resides in the fixed
 // host serializer. The production switch still rejects an unknown opcode.
#ifndef __SYNTHESIS__
 if((j.a|j.b|j.c|j.s0|j.s1|j.s2|j.s3)&31ull)return BAD_ALIGN;
 if((j.fa==UNUSED&&j.a)||(j.fb==UNUSED&&j.b)||(j.fc==UNUSED&&j.c)||
    (j.fs0==UNUSED&&j.s0)||(j.fs1==UNUSED&&j.s1)||(j.fs2==UNUSED&&j.s2)||
    (j.fs3==UNUSED&&j.s3))return BAD_FORMAT;
#endif
 return OK;
}
#ifndef __SYNTHESIS__
static unsigned decode_fixed(bus_t *memory,uint64_t address,unsigned index,Job &j,c209fixed::Kind &kind) {
 FixedJob compact;unsigned error=decode_fixed(memory,address,index,compact,kind);
 j=static_cast<Job>(compact);return error;
}
// Host-reference admission only. Production owns a fixed trusted buffer pool
// and performs range/lifetime checks at binding, not in the PL hot path.
static unsigned admit_ranges(const Job &j,uint64_t commands,unsigned jobs) {
#pragma HLS INLINE off
 uint64_t count[7]={0,0,0,0,0,0,0};unsigned writes=1u<<2;
 if(j.op==GEMM||j.op==GEMM_GELU_F16||j.op==GEMM_F16_A_PARTIAL) {
  count[0]=uint64_t(j.m)*j.k*2;count[1]=uint64_t(j.n)*(j.k==80u?64u:j.k)*2;
  count[2]=uint64_t(j.m)*j.n*(j.fc==F32?4u:2u);count[3]=j.m*4;count[4]=j.n*4;
  if(j.op==GEMM_GELU_F16){count[5]=j.m*4;writes|=1u<<5;}
 } else if(j.op==QUANTIZE) {
  count[0]=j.fa==F32?(uint64_t(j.m)-1)*80*4+288:uint64_t(j.m)*j.n*2;
  count[2]=uint64_t(j.m)*j.lc*2;count[3]=j.m*4;writes|=1u<<3;
 } else if(j.op==NORM_ADALN_QUANTIZE||j.op==NORM_ADALN_FIXED) {
  count[0]=uint64_t(j.m)*5120*4;count[2]=uint64_t(j.m)*5120*2;
  count[3]=j.op==NORM_ADALN_FIXED?32u:5120u*4;count[4]=count[5]=5120u*4;count[6]=j.m*4;writes|=1u<<6;
 } else if(j.op==RMS_NORM) {
  count[0]=count[2]=uint64_t(j.m)*128*2;count[3]=128u*4;
 } else if(j.op==ROPE||j.op==RMS_ROPE_FUSED) {
  count[0]=count[2]=uint64_t(j.m)*1280*2;
  if(j.op==ROPE){count[3]=count[4]=uint64_t(j.m)*128*4;}
  else {count[3]=128u*4;count[4]=count[5]=uint64_t(j.m)*128*4;}
 } else if(j.op==VECTOR) {
  count[0]=count[2]=uint64_t(j.m)*5120*2;
 } else if(j.op==PREPARE_KV||j.op==PREPARE_KV_GLOBAL||j.op==ATTENTION) {
  bool prep=j.op!=ATTENTION;unsigned stored=prep?(j.op==PREPARE_KV_GLOBAL?4096u:j.m):j.kc;
  count[0]=uint64_t(j.m)*1280*2;count[1]=uint64_t(prep?j.m:stored)*1280*2;
  count[2]=uint64_t(prep?stored:j.m)*1280*2;
  count[3]=stored*16u*4;count[4]=10u*((stored+127u)/128u)*128u*4;
  count[5]=10ull*((stored+127u)/128u)*32768u;
  if(prep)writes|=(1u<<3)|(1u<<4)|(1u<<5);
 }
 const uint64_t base[7]={j.a,j.b,j.c,j.s0,j.s1,j.s2,j.s3};
 uint64_t command_end=commands+uint64_t(jobs)*RECORD_BYTES;
 for(unsigned i=0;i<7;i++) {
#pragma HLS PIPELINE II=1
  if(base[i]+count[i]<base[i])return BAD_SHAPE;
  if(count[i]&&(writes&(1u<<i))&&base[i]<command_end&&commands<base[i]+count[i])return BAD_ALIAS;
 }
 return OK;
}
#endif
} // namespace c209five

