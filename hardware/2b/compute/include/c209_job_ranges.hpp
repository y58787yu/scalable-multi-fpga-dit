#pragma once
#include "cosmos.hpp"

// Descriptor-only guard. It touches no DDR and cannot establish that the host
// owns an allocation; host capacity checks remain required. Ranges conservatively
// include row-stride gaps and all complete AXI beats touched by a stage.
namespace c2 { namespace job_ranges {
struct Range {uint64_t begin,end;bool present,write;};
static inline bool add(uint64_t a,uint64_t b,uint64_t &out) {
 if(a>(~uint64_t(0))-b)return false;out=a+b;return true;
}
static inline unsigned ordinary_bytes(unsigned format) {return format==F16?2u:format==F32?4u:0u;}
static inline unsigned contiguous(Range &range,uint64_t base,uint64_t count,bool write) {
 range={base,base,false,write};if(!count)return OK;
 if(base&31u)return BAD_ALIGN;
 uint64_t rounded=0,end=0;
 if(!add(count,31,rounded))return BAD_SHAPE;rounded&=~uint64_t(31);
 if(!add(base,rounded,end))return BAD_SHAPE;
 range={base,end,true,write};return OK;
}
static inline unsigned matrix(Range &range,uint64_t base,unsigned rows,unsigned cols,
 unsigned stride,unsigned bytes,bool write) {
 range={base,base,false,write};if(!rows||!cols)return OK;
 if(bytes!=1&&bytes!=2&&bytes!=4)return BAD_FORMAT;
 if(rows>MAX_ROWS*16)return BAD_SHAPE;
 const unsigned shift=bytes==1?0u:bytes==2?1u:2u;
 const unsigned word_elements=bytes==1?32u:bytes==2?16u:8u;
 if(stride<cols || (stride&(word_elements-1)))return BAD_SHAPE;
 // rows<=65536, stride/cols are uint32, bytes<=4: the complete span is
 // below 2^50. Only base+span needs an overflow guard; no runtime divider.
 uint64_t last=(uint64_t(rows-1)*stride)<<shift;
 uint64_t row=((uint64_t(cols)<<shift)+31u)&~uint64_t(31);
 uint64_t count=last+row;
 return contiguous(range,base,count,write);
}
static inline bool overlap(const Range &a,const Range &b) {
 return a.present&&b.present&&a.begin<b.end&&b.begin<a.end;
}
static inline bool exact_inplace(const Job &j,unsigned read_slot,const Range &out,const Range &in) {
 bool allowed=false;unsigned format=UNUSED,stride=0;
 if(j.op==VECTOR) {
  allowed=read_slot==0||read_slot==1||(read_slot==4&&j.mode==SUM_GATED_RESIDUAL);
 }else if(j.op==LAYER_NORM||j.op==RMS_NORM||j.op==ROPE||
          (j.op==NORM_ADALN_QUANTIZE&&(j.mode&2)))allowed=read_slot==0;
 if(!allowed)return false;
 if(read_slot==0){format=j.fa;stride=j.la;}
 else if(read_slot==1){format=j.fb;stride=j.lb;}
 else if(read_slot==4){format=j.fs1;stride=j.lc;}
 return out.begin==in.begin&&out.end==in.end&&j.fc==format&&j.lc==stride;
}
} // namespace job_ranges

static unsigned validate_job_ranges(const Job &j,uint64_t command_base=0,uint64_t command_bytes=0) {
#pragma HLS INLINE off
 using namespace job_ranges;
 Range r[7]={};unsigned status=OK;
 const uint64_t base[7]={j.a,j.b,j.c,j.s0,j.s1,j.s2,j.s3};
 for(unsigned i=0;i<7;i++)if(base[i]&31u)return BAD_ALIGN;
 // Format/role/flags validation is performed by validate_program_job first.
 // Still reject invalid ordinary formats here before any division or footprint.
 const unsigned ab=ordinary_bytes(j.fa),bb=ordinary_bytes(j.fb),cb=ordinary_bytes(j.fc);
#define JR_MATRIX(slot,rows,cols,stride,bytes,wr) do {status=matrix(r[slot],base[slot],rows,cols,stride,bytes,wr);if(status!=OK)return status;}while(0)
#define JR_BYTES(slot,count,wr) do {status=contiguous(r[slot],base[slot],count,wr);if(status!=OK)return status;}while(0)
 switch(j.op) {
 case GEMM: {
  if(!j.m||j.m>MAX_ROWS||!j.n||j.n>FFN||!j.k||j.k>FFN)return BAD_SHAPE;
  uint64_t end=j.kc?uint64_t(j.k0)+j.kc:j.k;
  if(j.k0%16||end>j.k||end<=j.k0)return BAD_SHAPE;
  if((j.flags&LOCAL_PARTIAL)&&j.fc!=F32)return BAD_FORMAT;
  JR_MATRIX(0,j.m,j.k,j.la,2,false);
#if defined(C209_FIXED_PROFILE) && C209_FIXED_PROFILE
  if(j.flags&64u) {JR_BYTES(1,uint64_t(j.n)*(j.k==80u?64u:j.k)*2u,false);}
  else {JR_MATRIX(1,j.n,j.k,j.lb,2,false);}
#else
  JR_MATRIX(1,j.n,j.k,j.lb,(j.flags&WEIGHTS_INT8)?1:2,false);
#endif
  JR_MATRIX(2,j.m,j.n,j.lc,cb,true);
  JR_BYTES(3,uint64_t(j.m)*4,false);JR_BYTES(4,uint64_t(j.n)*4,false);
  if(j.flags&BIAS)JR_BYTES(5,uint64_t(j.n)*4,false);
  if(j.flags&SKIP_K)JR_BYTES(6,32,false);
  break;
 }
 case QUANTIZE:case PACK_BLOCK:case LAYER_NORM:case RMS_NORM:case NORM_ADALN_QUANTIZE: {
  bool fused=j.op==NORM_ADALN_QUANTIZE;
  bool quant=j.op==QUANTIZE||j.op==PACK_BLOCK;
  bool output_float=!quant&&(!fused||(j.mode&2));
  if(!j.m||j.m>((fused||quant)?MAX_ROWS:MAX_ROWS*16)||!j.n||j.n>FFN)return BAD_SHAPE;
  JR_MATRIX(0,j.m,j.n,j.la,ab,false);
  if(output_float)JR_MATRIX(2,j.m,j.n,j.lc,cb,true);
  else {if(j.lc<j.n)return BAD_SHAPE;JR_MATRIX(2,j.m,j.lc,j.lc,2,true);}
  if(quant)JR_BYTES(3,uint64_t(j.m)*4,true);
  else {
   JR_BYTES(3,uint64_t(j.n)*4,false);
   if(fused||(j.flags&BIAS))JR_BYTES(4,uint64_t(j.n)*4,false);
   if(fused)JR_BYTES(5,uint64_t(j.n)*4,false);
   if(!output_float)JR_BYTES(6,uint64_t(j.m)*4,true);
  }
  break;
 }
 case VECTOR:
  if(!j.m||j.m>MAX_ROWS||!j.n||j.n>FFN||j.mode>SUM_GATED_RESIDUAL)return BAD_SHAPE;
  JR_MATRIX(0,j.m,j.n,j.la,ab,false);JR_MATRIX(2,j.m,j.n,j.lc,cb,true);
  if(j.mode==ADD||j.mode==MUL||j.mode==GATED_RESIDUAL||j.mode==SUM_GATED_RESIDUAL)
   JR_MATRIX(1,j.m,j.n,j.lb,bb,false);
  if(j.mode==ADALN||j.mode==GATED_RESIDUAL||j.mode==SUM_GATED_RESIDUAL)JR_BYTES(3,uint64_t(j.n)*4,false);
  if(j.mode==ADALN)JR_BYTES(4,uint64_t(j.n)*4,false);
  if(j.mode==SUM_GATED_RESIDUAL)JR_MATRIX(4,j.m,j.n,j.lc,ordinary_bytes(j.fs1),false);
  break;
 case ROPE:
  if(!j.m||j.m>MAX_ROWS||!j.heads||j.heads>16||j.n!=j.heads*128)return BAD_SHAPE;
  JR_MATRIX(0,j.m,j.n,j.la,ab,false);JR_MATRIX(2,j.m,j.n,j.lc,cb,true);
  JR_BYTES(3,uint64_t(j.m)*128*4,false);JR_BYTES(4,uint64_t(j.m)*128*4,false);break;
 case UNPACK_BLOCK:
  if(!j.m||j.m>MAX_ROWS||!j.n||j.n>FFN||j.lc<j.n)return BAD_SHAPE;
  JR_MATRIX(0,j.m,j.n,j.la,2,false);
  JR_MATRIX(2,j.m,j.lc<((j.n+15u)&~15u)?j.lc:((j.n+15u)&~15u),j.lc,4,true);
  JR_BYTES(3,uint64_t(j.m)*4,false);break;
 case PREPARE_KV:case ATTENTION: {
  bool prepare=j.op==PREPARE_KV;
  if(!j.n||j.n>MAX_ROWS||!j.heads||j.heads>16)return BAD_SHAPE;
  const unsigned hs=(j.heads+7u)&~7u;
  if(prepare) {
   if(j.m>j.n||((j.flags&ZERO_KV_TAIL)?j.kc!=j.m:j.m!=j.n))return BAD_SHAPE;
  }else if(!j.m||j.m>MAX_ROWS||!(j.flags&PREQUANT_KV))return BAD_SHAPE;
  unsigned stored=prepare?j.m:((j.flags&ZERO_KV_TAIL)?j.kc:j.n);
  if(stored>j.n)return BAD_SHAPE;
  unsigned columns=j.heads*128,blocks=(stored+127)/128;
  if(!ab||(prepare?!bb:!cb))return BAD_FORMAT;
  const unsigned aw=ab==2?16u:8u,bw=prepare?(bb==2?16u:8u):16u,cw=prepare?16u:(cb==2?16u:8u);
  if(j.la<columns||j.lb<columns||j.lc<columns||(j.la&(aw-1))||
     (j.lb&(bw-1))||(j.lc&(cw-1)))return BAD_SHAPE;
  if(!prepare)JR_MATRIX(2,j.m,columns,j.lc,cb,true);
  if(!stored)break; // Prepare no-op; attention writes zero without reading Q/K/V.
  JR_MATRIX(0,prepare?stored:j.m,columns,j.la,ab,false);
  if(prepare) {
   JR_MATRIX(1,stored,columns,j.lb,bb,false);JR_MATRIX(2,stored,columns,j.lc,2,true);
  }else JR_MATRIX(1,stored,columns,j.lb,2,false);
  JR_BYTES(3,uint64_t(stored)*hs*4,prepare);
  JR_BYTES(4,uint64_t(j.heads)*blocks*128*4,prepare);
  JR_BYTES(5,uint64_t(j.heads)*blocks*32768,prepare);
  break;
 }
 default:return BAD_STAGE;
 }
#undef JR_MATRIX
#undef JR_BYTES
 if(command_bytes) {
  Range commands;
  status=contiguous(commands,command_base,command_bytes,false);
  if(status!=OK)return status;
  for(unsigned i=0;i<7;i++)if(r[i].write&&overlap(r[i],commands))return BAD_ALIAS;
 }
 for(unsigned i=0;i<7;i++)for(unsigned k=i+1;k<7;k++) {
  if(!(r[i].write||r[k].write)||!overlap(r[i],r[k]))continue;
  if(r[i].write&&r[k].write)return BAD_ALIAS;
  unsigned writer=r[i].write?i:k,reader=r[i].write?k:i;
  if(writer!=2||!exact_inplace(j,reader,r[writer],r[reader]))return BAD_ALIAS;
 }
 return OK;
}
} // namespace c2
