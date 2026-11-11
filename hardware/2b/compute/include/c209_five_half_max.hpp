#pragma once
#include <ap_int.h>

namespace c209five {
// Nonnegative binary16 encodings have the same order as their magnitudes.
// NaNs must be ignored just as the former floating `absolute > running` did.
// Subnormal encodings may survive in the state; the final widen performs the
// existing FTZ and the scale's zero test is deliberately after that widen.
static ap_uint<16> half_max_magnitude(ap_uint<16> bits) {
#pragma HLS INLINE
 ap_uint<16> magnitude=bits&0x7fffu;
 return magnitude>0x7c00u?ap_uint<16>(0):magnitude;
}
static ap_uint<16> half_max_abs8(ap_uint<128> packed,ap_uint<16> previous) {
#pragma HLS INLINE
 ap_uint<16> values[8],pairs[4],quads[2];
#pragma HLS ARRAY_PARTITION variable=values complete
#pragma HLS ARRAY_PARTITION variable=pairs complete
#pragma HLS ARRAY_PARTITION variable=quads complete
 for(unsigned i=0;i<8;i++) {
#pragma HLS UNROLL
  values[i]=half_max_magnitude(ap_uint<16>(packed.range(16*i+15,16*i)));
 }
 for(unsigned i=0;i<4;i++) {
#pragma HLS UNROLL
  pairs[i]=values[2*i]>values[2*i+1]?values[2*i]:values[2*i+1];
 }
 for(unsigned i=0;i<2;i++) {
#pragma HLS UNROLL
  quads[i]=pairs[2*i]>pairs[2*i+1]?pairs[2*i]:pairs[2*i+1];
 }
 ap_uint<16> current=quads[0]>quads[1]?quads[0]:quads[1];
 return current>previous?current:previous;
}
} // namespace c209five
