#pragma once
#include "cosmos.hpp"

namespace c209five {
// The descriptor carries a finite fixed-apple phase. Preserve those bounds
// across non-inlined HLS services instead of widening them back to unsigned.
// RMS128 has up to32768 head rows; dimensions/strides still fit13 bits.
// The16-bit m includes the32768 terminal count without wrapping.
struct FixedJob {
 ap_uint<5> op;
 ap_uint<16> m;
 ap_uint<13> n,k,la,lb,lc,k0,kc;
 ap_uint<4> heads;
 ap_uint<8> flags;
 ap_uint<3> mode;
 uint64_t a,b,c,s0,s1,s2,s3;
 float factor,epsilon;
 ap_uint<3> fa,fb,fc,fs0,fs1,fs2,fs3;
 ap_uint<2> abi_version;
 FixedJob():op(0),m(0),n(0),k(0),la(0),lb(0),lc(0),k0(0),kc(0),
  heads(0),flags(0),mode(0),a(0),b(0),c(0),s0(0),s1(0),s2(0),s3(0),
  factor(0),epsilon(0),fa(c2::UNUSED),fb(c2::UNUSED),fc(c2::UNUSED),
  fs0(c2::UNUSED),fs1(c2::UNUSED),fs2(c2::UNUSED),fs3(c2::UNUSED),abi_version(2) {}
 // Compatibility at the old numerical-reference boundary only. Production
 // services take FixedJob directly and do not roundtrip through c2::Job.
 FixedJob(const c2::Job &j):op(j.op),m(j.m),n(j.n),k(j.k),la(j.la),lb(j.lb),
  lc(j.lc),k0(j.k0),kc(j.kc),heads(j.heads),flags(j.flags),mode(j.mode),
  a(j.a),b(j.b),c(j.c),s0(j.s0),s1(j.s1),s2(j.s2),s3(j.s3),
  factor(j.factor),epsilon(j.epsilon),fa(j.fa),fb(j.fb),fc(j.fc),
  fs0(j.fs0),fs1(j.fs1),fs2(j.fs2),fs3(j.fs3),abi_version(j.abi_version) {}
 operator c2::Job() const {
#pragma HLS INLINE
  c2::Job j={};
  j.op=op;j.m=m;j.n=n;j.k=k;j.la=la;j.lb=lb;j.lc=lc;j.k0=k0;j.kc=kc;
  j.heads=heads;j.flags=flags;j.mode=mode;
  j.a=a;j.b=b;j.c=c;j.s0=s0;j.s1=s1;j.s2=s2;j.s3=s3;
  j.factor=factor;j.epsilon=epsilon;j.fa=fa;j.fb=fb;j.fc=fc;
  j.fs0=fs0;j.fs1=fs1;j.fs2=fs2;j.fs3=fs3;j.abi_version=abi_version;
  return j;
 }
};
// Preserve the full physical base, including its high bits. Only the bounded
// offset is narrow; no assumption is made that the CMA arena lies below4GiB.
using WordOffset=ap_uint<27>;
using ByteOffset=ap_uint<32>;
static uint64_t word_address(uint64_t base,WordOffset word_offset) {
#pragma HLS INLINE
 return (base>>5)+uint64_t(word_offset);
}
static uint64_t byte_address(uint64_t base,ByteOffset byte_offset) {
#pragma HLS INLINE
 return base+uint64_t(byte_offset);
}
} // namespace c209five
