#pragma once
#include "c209_precision.hpp"

// Narrow pointwise arithmetic only. Statistics, epsilon, reciprocals and row
// quantization scales must stay FP32; they must not pass through these helpers.
// Both operands of each operation are half and each result is committed to
// half before the following operation. No mixed half/float expression or FMA.
namespace c209five { namespace aux_mixed {
using scalar=c209mp::activation;
static_assert(c209mp::BITS==16,"Auxiliary pointwise arithmetic requires binary16");

static scalar canonical(scalar value) {
#pragma HLS INLINE
 ap_uint<16> bits=c209mp::bits(value);
 if(bits.range(14,10)==0)bits.range(9,0)=0;
 else if(bits.range(14,10)==31&&bits.range(9,0)!=0)bits=0x7e00u;
 return c209mp::from_bits(bits);
}
static scalar add(scalar a,scalar b) {
#pragma HLS INLINE
 half result=a+b;
 return canonical(result);
}
static scalar multiply(scalar a,scalar b) {
#pragma HLS INLINE
 half result=a*b;
 return canonical(result);
}
static scalar negate(scalar value) {
#pragma HLS INLINE
 return c209mp::from_bits(c209mp::bits(value)^ap_uint<16>(0x8000u));
}
static ap_uint<15> magnitude(scalar value) {
#pragma HLS INLINE
 ap_uint<15> bits=c209mp::bits(value);
 // Match the old max scan: NaNs do not update the maximum, infinity does.
 return bits>0x7c00u?ap_uint<15>(0):bits;
}
static float maximum4(const ap_uint<15> values[4]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=values complete
 ap_uint<15> a=values[0]>values[1]?values[0]:values[1];
 ap_uint<15> b=values[2]>values[3]?values[2]:values[3];
 return c209mp::widen(c209mp::from_bits(c209mp::word(a>b?a:b)));
}
} }
