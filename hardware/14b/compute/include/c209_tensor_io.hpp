#pragma once
#include "cosmos.hpp"
#ifndef __SYNTHESIS__
#include <cassert>
#endif
// Scalar helpers are for setup/metadata and reference fixtures. Bulk datapaths
// load a complete bus word and use lane/set_lane, never element RMW stores.
namespace c2 { namespace tensor {
static inline bool ordinary(unsigned format) { return format==F16 || format==F32; }
static inline unsigned bytes(unsigned format) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
 assert(ordinary(format));
#endif
 return format==F16?2u:format==F32?4u:0u;
}
static inline unsigned lanes(unsigned format) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
 assert(ordinary(format));
#endif
 return format==F16?16u:format==F32?8u:0u;
}
static inline float lane(bus_t value,unsigned index,unsigned format) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
 assert(ordinary(format) && index<lanes(format));
#endif
 if(format==F32) { union {uint32_t u;float f;} p;p.u=value.range(index*32+31,index*32);return p.f; }
 if(format==F16) {
#if C209_MIXED_PRECISION
  return c209mp::widen(c209mp::from_bits(value.range(index*16+15,index*16)));
#else
#ifndef __SYNTHESIS__
  assert(false && "Packed F16 requires the MP16 build");
#endif
#endif
 }
 return 0;
}
static inline bus_t set_lane(bus_t value,unsigned index,float scalar,unsigned format) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
 assert(ordinary(format) && index<lanes(format));
#endif
 if(format==F32) {union {float f;uint32_t u;} p;p.f=scalar;value.range(index*32+31,index*32)=p.u;}
 else if(format==F16) {
#if C209_MIXED_PRECISION
  value.range(index*16+15,index*16)=c209mp::bits(c209mp::narrow(scalar));
#else
#ifndef __SYNTHESIS__
  assert(false && "Packed F16 requires the MP16 build");
#endif
#endif
 }
 return value;
}
static inline float read(bus_t *p,uint64_t base,uint64_t element,unsigned format) {
#pragma HLS INLINE
 const uint64_t address=base+element*bytes(format);
 return lane(p[address>>5],unsigned(address&31u)/bytes(format),format);
}
static inline void write(bus_t *p,uint64_t base,uint64_t element,float scalar,unsigned format) {
#pragma HLS INLINE
 const uint64_t address=base+element*bytes(format);
 p[address>>5]=set_lane(p[address>>5],unsigned(address&31u)/bytes(format),scalar,format);
}
} }
