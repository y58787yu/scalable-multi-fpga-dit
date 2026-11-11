#pragma once
// C209 MODULAR16: explicitly typed F16/F32 external tensor ABI, FP16 ordinary
// on-chip payloads. Scalar metadata and wide accumulation remain unchanged.
// Scales, statistics, softmax state and INT48 matrix accumulation are NOT half.
// Binary16 uses nearest-even with flush-to-signed-zero (FTZ). At the exact
// min-normal transition midpoint (2^-14 - 2^-25), FTZ wins. Explicitly
// canonicalize both sides: vendor underflow detection near this boundary
// does not always agree with IEEE rounding-before-FTZ.
// Macro 0 is an explicit numerical-regression build, never the MP16 release.
#ifndef C209_MIXED_PRECISION
#define C209_MIXED_PRECISION 1
#endif
#if C209_MIXED_PRECISION != 0 && C209_MIXED_PRECISION != 1
#error C209_MIXED_PRECISION must be 0 or 1
#endif
#include <ap_int.h>
#include <stdint.h>
#if C209_MIXED_PRECISION
#if defined(HLS_NO_XIL_FPO_LIB) && !defined(__SYNTHESIS__)
#error MP16 CPU verification requires the vendor bit-accurate FPO library; half_sim truncates and flushes subnormals
#endif
#include <hls_half.h>
#endif
namespace c209mp {
#if C209_MIXED_PRECISION
using activation=half;
constexpr unsigned BITS=16;
#else
using activation=float;
constexpr unsigned BITS=32;
#endif
using word=ap_uint<BITS>;
static inline word bits(activation value) {
#if C209_MIXED_PRECISION
#ifdef __SYNTHESIS__
 union { half h; uint16_t u; } p; p.h=value; return p.u;
#else
 return value.get_bits();
#endif
#else
 union { float f; uint32_t u; } p; p.f=value; return p.u;
#endif
}
static inline activation from_bits(word value) {
#if C209_MIXED_PRECISION
#ifdef __SYNTHESIS__
 union { half h; uint16_t u; } p; p.u=value.to_uint(); return p.h;
#else
 half result; result.set_bits(value.to_uint()); return result;
#endif
#else
 union { float f; uint32_t u; } p; p.u=value.to_uint(); return p.f;
#endif
}
static inline activation narrow(float value) {
#if C209_MIXED_PRECISION
 union { float f; uint32_t u; } p; p.f=value;
 if ((p.u & 0x7fffffffu) <= 0x387fe000u)
  return from_bits(word((p.u >> 16) & 0x8000u));
 // Rounding above that midpoint must reach the minimum normal. Do not
 // rely on the vendor model's earlier tininess test in this small interval.
 if ((p.u & 0x7fffffffu) < 0x38800000u)
  return from_bits(word(((p.u >> 16) & 0x8000u) | 0x0400u));
 activation result=activation(value);
 word encoded=bits(result);
 if ((encoded & 0x7c00u)==0) return from_bits(encoded & 0x8000u);
 return result;
#else
 return value;
#endif
}
static inline float widen(activation value) {
#if C209_MIXED_PRECISION
 word encoded=bits(value);
 if ((encoded & 0x7c00u)==0) {
  union { float f; uint32_t u; } p;
  p.u=(encoded.to_uint() & 0x8000u) << 16; return p.f;
 }
#endif
 return float(value);
}
static inline float round(float value) { return widen(narrow(value)); }
template<class T> static inline T convert(float value) { return T(value); }
template<> inline activation convert<activation>(float value) { return narrow(value); }
}
