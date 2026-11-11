#pragma once
#include "cosmos.hpp"
#include "c209_job_ranges.hpp"

// Fixed production admission, not a promise that an unused generic opcode is
// synthesized away. The production top has a separate, finite call graph.
namespace c209fixed {
using namespace c2;
constexpr unsigned HARDWARE_ID=c209tp4::HARDWARE_ID,UPDATES=c209tp4::UPDATES,QUERY_ROWS=64,CONTEXTS=2;
constexpr unsigned HIDDEN=c209tp4::HIDDEN,LOCAL_WIDTH=c209tp4::LOCAL_WIDTH,LOCAL_FFN=c209tp4::LOCAL_FFN;
constexpr unsigned LOCAL_HEADS=c209tp4::LOCAL_HEADS,KS_STRIDE=c209tp4::SCALE_HEAD_STRIDE;
constexpr unsigned MAX_LOCAL_COLS=LOCAL_FFN,NORM_WIDTH=HIDDEN;
// Only these complete 128-token partitions enter the source-only SP3 profile.
static bool local_rows(unsigned m) { return m==1280u||m==1408u; }
static bool gemm_rows(unsigned m) { return m==128u||m==256u; }
static unsigned canonical_local_rows(unsigned m) { return m==1408u?1408u:1280u; }
enum Kind : unsigned { INVALID=0,PATCH,PROJECTION,ATTENTION_PARTIAL,FFN_UP,
 FFN_PARTIAL,FINAL_OUTPUT,QUANT_PATCH,QUANT_PROJECTION,QUANT_FFN,
 NORM_ADALN,RMS128,ROTARY,FFN_GELU,SELF_PREPARE,SELF_ATTENTION,CROSS_ATTENTION };
struct Spec {
 Kind kind;unsigned m,n,k,la,lb,lc;unsigned format[7];
};
static unsigned bits(float value) {
#pragma HLS INLINE
 union {float f;unsigned u;} p;p.f=value;return p.u;
}
// Condition is bound by the software task, independently of physical TP rank.
// Only the admitted job's actual stored-key extent enters the datapath.
static unsigned stored_cross(const Job &j) {
#pragma HLS INLINE
 return j.kc;
}
static bool shape(const Job &j,unsigned m,unsigned n,unsigned k,unsigned la,unsigned lb,unsigned lc) {
#pragma HLS INLINE
 return j.m==m&&j.n==n&&j.k==k&&j.la==la&&j.lb==lb&&j.lc==lc;
}
static unsigned describe(const Job &j,unsigned node,Spec &s) {
#pragma HLS INLINE off
 s={INVALID,0,0,0,0,0,0,{UNUSED,UNUSED,UNUSED,UNUSED,UNUSED,UNUSED,UNUSED}};
 if(node<1||node>4||j.abi_version!=2)return BAD_ABI;
 if(j.k0||j.factor!=0.f||(bits(j.epsilon)!=0&&bits(j.epsilon)!=bits(1e-6f)))return BAD_FLAGS;
 if(j.mode!=(j.op==VECTOR?unsigned(GELU_TANH):0u))return BAD_FLAGS;
 if(j.op!=GEMM&&j.k)return BAD_SHAPE;
 if(j.op!=ATTENTION&&j.op!=PREPARE_KV&&j.kc)return BAD_FLAGS;
 if(j.heads!=((j.op==ATTENTION||j.op==PREPARE_KV||j.op==ROPE)?LOCAL_HEADS:0u))return BAD_SHAPE;
 switch(j.op) {
 case GEMM:
  if(j.flags!=64u&&j.flags!=(64u|LOCAL_PARTIAL))return BAD_FLAGS;
  if(!(j.flags&LOCAL_PARTIAL)&&local_rows(j.m)&&shape(j,j.m,HIDDEN,80,80,80,HIDDEN))s.kind=PATCH;
  else if(!(j.flags&LOCAL_PARTIAL)&&local_rows(j.m)&&shape(j,j.m,LOCAL_WIDTH,HIDDEN,HIDDEN,HIDDEN,LOCAL_WIDTH))s.kind=PROJECTION;
  else if((j.flags&LOCAL_PARTIAL)&&gemm_rows(j.m)&&shape(j,j.m,HIDDEN,LOCAL_WIDTH,LOCAL_WIDTH,LOCAL_WIDTH,HIDDEN))s.kind=ATTENTION_PARTIAL;
  else if(!(j.flags&LOCAL_PARTIAL)&&local_rows(j.m)&&shape(j,j.m,LOCAL_FFN,HIDDEN,HIDDEN,HIDDEN,LOCAL_FFN))s.kind=FFN_UP;
  else if((j.flags&LOCAL_PARTIAL)&&gemm_rows(j.m)&&shape(j,j.m,HIDDEN,LOCAL_FFN,LOCAL_FFN,LOCAL_FFN,HIDDEN))s.kind=FFN_PARTIAL;
  else if(!(j.flags&LOCAL_PARTIAL)&&local_rows(j.m)&&shape(j,j.m,64,HIDDEN,HIDDEN,HIDDEN,64))s.kind=FINAL_OUTPUT;
  s.format[0]=I16;s.format[1]=I16;
  s.format[2]=(s.kind==PROJECTION||s.kind==FFN_UP)?F16:F32;
  s.format[3]=F32;s.format[4]=F32;
  break;
 case QUANTIZE:
  if(j.flags)return BAD_FLAGS;
  if(local_rows(j.m)&&j.n==68&&j.la==80&&j.lc==80)s.kind=QUANT_PATCH;
  else if(local_rows(j.m)&&j.n==LOCAL_WIDTH&&j.la==LOCAL_WIDTH&&j.lc==LOCAL_WIDTH)s.kind=QUANT_PROJECTION;
  else if(local_rows(j.m)&&j.n==LOCAL_FFN&&j.la==LOCAL_FFN&&j.lc==LOCAL_FFN)s.kind=QUANT_FFN;
  s.format[0]=s.kind==QUANT_PATCH?F32:F16;s.format[2]=I16;s.format[3]=F32;
  break;
 case NORM_ADALN_QUANTIZE:
  if(j.flags)return BAD_FLAGS;
  if(local_rows(j.m)&&j.n==HIDDEN&&j.la==HIDDEN&&j.lc==HIDDEN)s.kind=NORM_ADALN;
  s.format[0]=F32;s.format[2]=I16;s.format[3]=F32;s.format[4]=F32;s.format[5]=F32;s.format[6]=F32;
  break;
 case RMS_NORM:
  if(j.flags)return BAD_FLAGS;
  if((j.m==1280u*LOCAL_HEADS||j.m==1408u*LOCAL_HEADS)&&j.n==128&&j.la==128&&j.lc==128)s.kind=RMS128;
  s.format[0]=F16;s.format[2]=F16;s.format[3]=F32;
  break;
 case ROPE:
  if(j.flags)return BAD_FLAGS;
  if(local_rows(j.m)&&j.n==LOCAL_WIDTH&&j.la==LOCAL_WIDTH&&j.lc==LOCAL_WIDTH)s.kind=ROTARY;
  s.format[0]=F16;s.format[2]=F16;s.format[3]=F32;s.format[4]=F32;
  break;
 case VECTOR:
  if(j.flags)return BAD_FLAGS;
  if(local_rows(j.m)&&j.n==LOCAL_FFN&&j.la==LOCAL_FFN&&j.lc==LOCAL_FFN)s.kind=FFN_GELU;
  s.format[0]=F16;s.format[2]=F16;
  break;
 case PREPARE_KV:
  if(j.flags)return BAD_FLAGS;
  if(local_rows(j.m)&&j.n==j.m&&j.kc==j.m&&j.la==LOCAL_WIDTH&&j.lb==LOCAL_WIDTH&&j.lc==LOCAL_WIDTH)s.kind=SELF_PREPARE;
  s.format[0]=F16;s.format[1]=F16;s.format[2]=KV_PACKED;s.format[3]=F32;s.format[4]=F32;s.format[5]=KV_PACKED;
  break;
 case ATTENTION:
  if(j.flags!=PREQUANT_KV&&j.flags!=(PREQUANT_KV|ZERO_KV_TAIL))return BAD_FLAGS;
  if(local_rows(j.m)&&j.la==LOCAL_WIDTH&&j.lb==LOCAL_WIDTH&&j.lc==LOCAL_WIDTH) {
   if(j.flags==PREQUANT_KV&&j.n==4096&&j.kc==4096)s.kind=SELF_ATTENTION;
   if(j.flags==(PREQUANT_KV|ZERO_KV_TAIL)&&j.n==512&&j.kc>=1&&j.kc<=c209tp4::CROSS_KEYS)s.kind=CROSS_ATTENTION;
  }
  s.format[0]=F16;s.format[1]=KV_PACKED;s.format[2]=F16;s.format[3]=F32;s.format[4]=F32;s.format[5]=KV_PACKED;
  break;
 default:return BAD_STAGE;
 }
 if(s.kind==INVALID)return BAD_SHAPE;
 const unsigned actual[7]={j.fa,j.fb,j.fc,j.fs0,j.fs1,j.fs2,j.fs3};
 const uint64_t address[7]={j.a,j.b,j.c,j.s0,j.s1,j.s2,j.s3};
 for(unsigned i=0;i<7;i++) {
#pragma HLS PIPELINE off
  if(actual[i]!=s.format[i]||(s.format[i]==UNUSED&&address[i]))return BAD_FORMAT;
  if(address[i]&31ull)return BAD_ALIGN;
 }
 // Drive the datapath with a finite constant mux, not unbounded raw descriptor
 // fields whose values merely happened to pass an earlier condition.
 s.m=(s.kind==ATTENTION_PARTIAL||s.kind==FFN_PARTIAL)?(j.m==128u?128u:256u):
     s.kind==RMS128?(j.m==1408u*LOCAL_HEADS?1408u*LOCAL_HEADS:1280u*LOCAL_HEADS):canonical_local_rows(j.m);
 switch(s.kind) {
 case PATCH:s.n=HIDDEN;s.k=80;s.la=s.lb=80;s.lc=HIDDEN;break;
 case PROJECTION:s.n=LOCAL_WIDTH;s.k=HIDDEN;s.la=s.lb=HIDDEN;s.lc=LOCAL_WIDTH;break;
 case ATTENTION_PARTIAL:s.n=HIDDEN;s.k=LOCAL_WIDTH;s.la=s.lb=LOCAL_WIDTH;s.lc=HIDDEN;break;
 case FFN_UP:s.n=LOCAL_FFN;s.k=HIDDEN;s.la=s.lb=HIDDEN;s.lc=LOCAL_FFN;break;
 case FFN_PARTIAL:s.n=HIDDEN;s.k=LOCAL_FFN;s.la=s.lb=LOCAL_FFN;s.lc=HIDDEN;break;
 case FINAL_OUTPUT:s.n=64;s.k=HIDDEN;s.la=s.lb=HIDDEN;s.lc=64;break;
 case QUANT_PATCH:s.n=68;s.la=s.lc=80;break;
 case QUANT_PROJECTION:s.n=s.la=s.lc=LOCAL_WIDTH;break;
 case QUANT_FFN:case FFN_GELU:s.n=s.la=s.lc=LOCAL_FFN;break;
 case NORM_ADALN:s.n=s.la=s.lc=HIDDEN;break;
 case RMS128:s.n=s.la=s.lc=128;break;
 case ROTARY:s.n=s.la=s.lc=LOCAL_WIDTH;break;
 case SELF_PREPARE:s.n=s.m;s.la=s.lb=s.lc=LOCAL_WIDTH;break;
 case SELF_ATTENTION:s.n=4096;s.la=s.lb=s.lc=LOCAL_WIDTH;break;
 case CROSS_ATTENTION:s.n=512;s.la=s.lb=s.lc=LOCAL_WIDTH;break;
 default:return BAD_STAGE;
 }
 return OK;
}
static Job canonical_job(const Job &source,const Spec &s,unsigned node) {
#pragma HLS INLINE
 Job j=source;j.m=s.m;j.n=s.n;j.k=s.k;j.la=s.la;j.lb=s.lb;j.lc=s.lc;
 j.fa=s.format[0];j.fb=s.format[1];j.fc=s.format[2];j.fs0=s.format[3];
 j.fs1=s.format[4];j.fs2=s.format[5];j.fs3=s.format[6];
 j.k0=0;j.kc=0;j.flags=0;j.mode=0;j.heads=0;j.factor=0.f;j.epsilon=1e-6f;
 if(s.kind>=PATCH&&s.kind<=FINAL_OUTPUT) {
  j.op=GEMM;j.flags=64u|((s.kind==ATTENTION_PARTIAL||s.kind==FFN_PARTIAL)?unsigned(LOCAL_PARTIAL):0u);
 } else if(s.kind>=QUANT_PATCH&&s.kind<=QUANT_FFN)j.op=QUANTIZE;
 else if(s.kind==NORM_ADALN)j.op=NORM_ADALN_QUANTIZE;
 else if(s.kind==RMS128)j.op=RMS_NORM;
 else if(s.kind==ROTARY){j.op=ROPE;j.heads=LOCAL_HEADS;}
 else if(s.kind==FFN_GELU){j.op=VECTOR;j.mode=GELU_TANH;}
 else {
  j.heads=LOCAL_HEADS;j.kc=s.kind==CROSS_ATTENTION?stored_cross(source):s.kind==SELF_PREPARE?s.m:4096u;
  j.op=s.kind==SELF_PREPARE?PREPARE_KV:ATTENTION;
  j.flags=s.kind==SELF_PREPARE?0u:s.kind==CROSS_ATTENTION?(PREQUANT_KV|ZERO_KV_TAIL):PREQUANT_KV;
 }
 return j;
}
static unsigned validate_ranges(const Job &j,const Spec &s,uint64_t command_base,uint64_t command_bytes) {
#pragma HLS INLINE off
 using namespace c2::job_ranges;
 // Fixed shapes have constant footprints; cross-KV depends only on bounded stored keys.
 // Keep only
 // base+constant overflow logic here instead of re-synthesizing generic matrix
 // stride multipliers after describe() has already accepted a finite shape.
 unsigned count[7]={0,0,0,0,0,0,0};unsigned writes=1u<<2;
 switch(s.kind) {
 case PATCH:count[0]=s.m*80*2;count[1]=HIDDEN*64*2;count[2]=s.m*HIDDEN*4;count[3]=s.m*4;count[4]=HIDDEN*4;break;
 case PROJECTION:count[0]=s.m*HIDDEN*2;count[1]=LOCAL_WIDTH*HIDDEN*2;count[2]=s.m*LOCAL_WIDTH*2;count[3]=s.m*4;count[4]=LOCAL_WIDTH*4;break;
 case ATTENTION_PARTIAL:count[0]=s.m*LOCAL_WIDTH*2;count[1]=HIDDEN*LOCAL_WIDTH*2;count[2]=s.m*HIDDEN*4;count[3]=s.m*4;count[4]=HIDDEN*4;break;
 case FFN_UP:count[0]=s.m*HIDDEN*2;count[1]=LOCAL_FFN*HIDDEN*2;count[2]=s.m*LOCAL_FFN*2;count[3]=s.m*4;count[4]=LOCAL_FFN*4;break;
 case FFN_PARTIAL:count[0]=s.m*LOCAL_FFN*2;count[1]=HIDDEN*LOCAL_FFN*2;count[2]=s.m*HIDDEN*4;count[3]=s.m*4;count[4]=HIDDEN*4;break;
 case FINAL_OUTPUT:count[0]=s.m*HIDDEN*2;count[1]=64u*HIDDEN*2;count[2]=s.m*64*4;count[3]=s.m*4;count[4]=64u*4;break;
 // The last PATCH input row touches a full AXI beat after element67.
 case QUANT_PATCH:count[0]=(s.m-1u)*80*4+288;count[2]=s.m*80*2;count[3]=s.m*4;writes|=1u<<3;break;
 case QUANT_PROJECTION:count[0]=count[2]=s.m*LOCAL_WIDTH*2;count[3]=s.m*4;writes|=1u<<3;break;
 case QUANT_FFN:count[0]=count[2]=s.m*LOCAL_FFN*2;count[3]=s.m*4;writes|=1u<<3;break;
 case NORM_ADALN:count[0]=s.m*HIDDEN*4;count[2]=s.m*HIDDEN*2;count[3]=count[4]=count[5]=HIDDEN*4;count[6]=s.m*4;writes|=1u<<6;break;
 case RMS128:count[0]=count[2]=s.m*128*2;count[3]=128u*4;break;
 case ROTARY:count[0]=count[2]=s.m*LOCAL_WIDTH*2;count[3]=count[4]=s.m*128*4;break;
 case FFN_GELU:count[0]=count[2]=s.m*LOCAL_FFN*2;break;
 case SELF_PREPARE:
  count[0]=count[1]=count[2]=count[5]=s.m*LOCAL_WIDTH*2;
  count[3]=s.m*KS_STRIDE*4;count[4]=LOCAL_HEADS*(s.m/128u)*128*4;
  writes|=(1u<<3)|(1u<<4)|(1u<<5);break;
 case SELF_ATTENTION:
  count[0]=count[2]=s.m*LOCAL_WIDTH*2;
  count[1]=count[5]=4096u*LOCAL_WIDTH*2;
  count[3]=4096u*KS_STRIDE*4;count[4]=LOCAL_HEADS*32u*128*4;break;
 case CROSS_ATTENTION: {
  const unsigned stored=j.kc,blocks=(stored+127u)/128u;
  count[0]=count[2]=s.m*LOCAL_WIDTH*2;count[1]=stored*LOCAL_WIDTH*2;
  count[3]=stored*KS_STRIDE*4;count[4]=LOCAL_HEADS*blocks*128*4;
  count[5]=LOCAL_HEADS*blocks*32768u;
  break;
 }
 default:return BAD_STAGE;
 }
 Range r[7]={};unsigned e=OK;
 const uint64_t base[7]={j.a,j.b,j.c,j.s0,j.s1,j.s2,j.s3};
 for(unsigned i=0;i<7;i++) {
#pragma HLS PIPELINE off
  e=contiguous(r[i],base[i],count[i],bool((writes>>i)&1u));
  if(e!=OK)return e;
 }
 Range program;e=contiguous(program,command_base,command_bytes,false);if(e!=OK)return e;
 for(unsigned i=0;i<7;i++)if(r[i].write&&overlap(r[i],program))return BAD_ALIAS;
 for(unsigned i=0;i<7;i++)for(unsigned k=i+1;k<7;k++) {
  if(!(r[i].write||r[k].write)||!overlap(r[i],r[k]))continue;
  if(r[i].write&&r[k].write)return BAD_ALIAS;
  const unsigned writer=r[i].write?i:k,reader=r[i].write?k:i;
  bool inplace=(s.kind==RMS128||s.kind==ROTARY||s.kind==FFN_GELU)&&writer==2&&reader==0&&r[0].begin==r[2].begin&&r[0].end==r[2].end;
  if(!inplace)return BAD_ALIAS;
 }
 return OK;
}
static unsigned validate_job(const Job &j,unsigned node,uint64_t commands,uint64_t bytes,Spec &s) {
#pragma HLS INLINE off
 unsigned e=describe(j,node,s);return e==OK?validate_ranges(j,s,commands,bytes):e;
}
} // namespace c209fixed
