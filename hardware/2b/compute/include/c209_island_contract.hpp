#pragma once
#include "cosmos.hpp"

// G3 stream-only PROBE ABI. This is not the production DDR/program ABI.
// Every job is three little-endian 64-bit words:
//   0: opcode[7:0], flags[15:8], rows[31:16], local_columns[47:32], reserved[63:48]
//   1: inner_or_stored_keys[31:0], zero_key_count[63:32]
//   2: factor_bits[31:0], job_tag[63:32]
// ATTN: one head, <=64 queries, two contexts, original 128-key blocks.
// GEMM: two independent local column shards, K split into original <=256
// commands. No K reduction/reassociation. GELU is explicitly unsupported here.
// STOP: three zero words, sent only after complete valid input jobs.
// The top-level dispatcher rejects an invalid job without consuming operands;
// callers must not send payload for a rejected descriptor.
namespace c209island {
using word64=ap_uint<64>;
using stream64=hls::stream<word64>;
using stream128=hls::stream<ap_uint<128>>;
using stream256=hls::stream<c2::bus_t>;
enum Opcode : unsigned { STOP=0, ATTN=1, GEMM=2 };
struct Job {
 unsigned op,flags,rows,columns,inner,zero_count,tag;
 float factor;
 bool reserved_zero;
};
static float from_bits(ap_uint<32> bits) {
 union {uint32_t u;float f;} value;value.u=bits.to_uint();return value.f;
}
static ap_uint<32> to_bits(float value) {
 union {uint32_t u;float f;} bits;bits.f=value;return bits.u;
}
static Job decode(word64 a,word64 b,word64 c) {
 Job j;
 j.op=a.range(7,0);j.flags=a.range(15,8);j.rows=a.range(31,16);j.columns=a.range(47,32);
 j.inner=b.range(31,0);j.zero_count=b.range(63,32);j.factor=from_bits(c.range(31,0));j.tag=c.range(63,32);
 j.reserved_zero=a.range(63,48)==0;
 return j;
}
static Job read_job(stream64 &input) {
 word64 a=input.read(),b=input.read(),c=input.read();return decode(a,b,c);
}
static unsigned validate(const Job &j) {
 if(!j.reserved_zero)return c2::BAD_ABI;
 if(j.op==STOP)return (j.flags||j.rows||j.columns||j.inner||j.zero_count||j.tag||to_bits(j.factor)!=0)?c2::BAD_ABI:c2::OK;
 if(j.op==ATTN) {
  if(j.flags)return c2::BAD_FLAGS;
  // Check the encoded exponent before any floating-point comparison: both
  // infinities and every NaN payload must fail closed at the stream boundary.
  if((to_bits(j.factor)&0x7f800000u)==0x7f800000u)return c2::BAD_FORMAT;
  if(!j.rows||j.rows>64||j.columns!=128||!j.inner||j.inner>4096||j.zero_count>4096-j.inner)return c2::BAD_SHAPE;
  return c2::OK;
 }
 if(j.op==GEMM) {
  if(j.flags&~(c2::BIAS|c2::LOCAL_PARTIAL))return c2::BAD_FLAGS; // includes GELU: never a dummy path
  if((j.flags&c2::BIAS)&&(j.flags&c2::LOCAL_PARTIAL))return c2::BAD_FLAGS;
  if(!j.rows||j.rows>256||!j.columns||j.columns>128||!j.inner||j.inner>8192||j.zero_count)return c2::BAD_SHAPE;
  return c2::OK;
 }
 return c2::BAD_STAGE;
}
static unsigned contexts(const Job &j){return (j.rows+31)/32;}
static unsigned blocks(const Job &j){return contexts(j)*((j.inner+127)/128);}
static unsigned padded_columns(const Job &j){return (j.columns+31)&~31u;}
// Output event: tag[63:32], format[31:24], values_per_beat[23:16], beats[15:0].
// GEMM includes zero-padded local columns up to a multiple of 32. ATTN A emits
// a zero-length completion; B emits row-major, 8 FP16 values per 128-bit beat.
static word64 event(unsigned tag,unsigned format,unsigned lanes,unsigned beats) {
 word64 value=0;value.range(63,32)=tag;value.range(31,24)=format;
 value.range(23,16)=lanes;value.range(15,0)=beats;return value;
}
}
