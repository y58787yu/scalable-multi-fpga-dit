#pragma once
// Included inside attention_detail. All values are finite bit patterns.
// The argument is a nonnegative log difference in unsigned Q4.12. Values
// at or beyond16 map to zero before this finite16-bit unit is entered.
// exp output uses unsigned Q1.15: 0..32768, including exact1.
namespace softmax16 {
using Half=ap_uint<16>;
using Den=ap_uint<32>; // unsigned Q13.19; global4096-key bound is2^31.
constexpr unsigned DEN_FRACTION=19;
constexpr unsigned EXP_FRACTION=15;

static Half finite_half(float value) {
#pragma HLS INLINE
 unsigned bits=::fu(value),magnitude=bits&0x7fffffffu;
 // Invalid arithmetic cannot feed NaNs into the exponential/denominator.
 // A NaN score is represented as a masked -Inf half and contributes zero.
 if(magnitude>0x7f800000u)return Half(0xfc00u);
 if(magnitude>0x477fe000u)return Half((bits>>16&0x8000u)|0x7bffu);
 return c209mp::bits(c209mp::narrow(value));
}
static bool valid_half(Half value) {
#pragma HLS INLINE
 return (value&0x7c00u)!=0x7c00u;
}
static Half maximum(Half left,Half right) {
#pragma HLS INLINE
 if(!valid_half(right))return left;
 if(!valid_half(left))return right;
 ap_uint<16> lk=left[15]?ap_uint<16>(~left):ap_uint<16>(left^0x8000u);
 ap_uint<16> rk=right[15]?ap_uint<16>(~right):ap_uint<16>(right^0x8000u);
 return rk>lk?right:left;
}
static ap_int<30> half_q12(Half bits) {
#pragma HLS INLINE
 ap_uint<5> exponent=bits.range(14,10);
 if(exponent==0)return 0; // The external activation ABI already uses FTZ.
 ap_uint<11> mantissa=ap_uint<11>(1024u)|bits.range(9,0);
 ap_uint<29> magnitude=0;
 if(exponent>=13)magnitude=ap_uint<29>(mantissa)<<(exponent-13);
 else {
  unsigned shift=13-exponent.to_uint();
  magnitude=(ap_uint<29>(mantissa)+(ap_uint<29>(1)<<(shift-1)))>>shift;
 }
 ap_int<30> signed_value=magnitude;
 return bits[15]?ap_int<30>(-signed_value):signed_value;
}
template<unsigned UNIT> static ap_uint<16> exp_delta(ap_uint<16> difference) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
 // 64 linear segments for2^(-fraction); range reduction is integer.
 // Each word contains Q1.15 base[15:0] and nonnegative delta[24:16].
 static const unsigned segments[64]={
  0x1618000u,0x15d7e9fu,0x15a7d42u,0x1557be8u,0x1527a93u,0x14f7941u,0x14a77f2u,0x14876a8u,0x1437560u,0x140741du,0x13d72ddu,0x13a71a0u,0x1367066u,0x1326f30u,0x12f6dfeu,0x12d6ccfu,0x1286ba2u,0x1266a7au,0x1226954u,0x1206832u,0x11c6712u,0x11965f6u,0x11664ddu,0x11363c7u,0x11162b4u,0x10d61a3u,0x10a6096u,0x1085f8cu,0x1045e84u,0x1025d80u,0xff5c7eu,0xfd5b7fu,0xf95a82u,0xf75989u,0xf45892u,0xf2579eu,0xef56acu,0xec55bdu,0xea54d1u,0xe853e7u,0xe452ffu,0xe3521bu,0xe05138u,0xdd5058u,0xdc4f7bu,0xd84e9fu,0xd74dc7u,0xd44cf0u,0xd24c1cu,0xd04b4au,0xcd4a7au,0xcb49adu,0xc948e2u,0xc74819u,0xc54752u,0xc2468du,0xc145cbu,0xbe450au,0xbc444cu,0xbb4390u,0xb842d5u,0xb6421du,0xb54167u,0xb240b2u
 };
#pragma HLS BIND_STORAGE variable=segments type=rom_1p impl=lutram
 ap_uint<37> product=ap_uint<37>(difference)*ap_uint<21>(1512775u); // log2(e),Q20.
 ap_uint<22> binary=(product+32768u)>>16; // Q16.
 unsigned whole=binary>>16;
 ap_uint<6> segment=binary.range(15,10);
 ap_uint<10> position=binary.range(9,0);
 ap_uint<25> entry=segments[segment];
 ap_uint<16> base=entry.range(15,0);
 ap_uint<9> slope=entry.range(24,16);
 ap_uint<20> interpolation=ap_uint<20>(slope)*position+512u;
 ap_uint<17> mantissa=ap_uint<17>(base)-(interpolation>>10);
 if(whole==0)return ap_uint<16>(mantissa);
 if(whole>16)return 0;
 ap_uint<33> rounded=ap_uint<33>(mantissa)+(ap_uint<33>(1)<<(whole-1));
 return ap_uint<16>(rounded>>whole);
}
template<unsigned UNIT> static ap_uint<16> exponential(Half largest,Half value) {
#pragma HLS INLINE
 if(!valid_half(value)||!valid_half(largest))return 0;
 ap_int<31> difference=ap_int<31>(half_q12(largest))-half_q12(value);
 if(difference<=0)return 32768u;
 if(difference>=65536)return 0;
 return exp_delta<UNIT>(ap_uint<16>(difference));
}
static ap_uint<16> probability_word(ap_uint<16> value) {
#pragma HLS INLINE
 // round(Q1.15 probability *32760), exact shifts instead of a multiplier.
 ap_uint<32> scaled=(ap_uint<32>(value)<<15)-(ap_uint<32>(value)<<3)+16384u;
 return ap_uint<16>(scaled>>15);
}
static Den update_den(Den previous,ap_uint<16> alpha,ap_uint<23> sum) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
 // Four rows are independent transactions. Register the integer product
 // and each addition instead of placing mul48/add48/add34/saturation in
 // the same cycle of a packed-descriptor read/modify/write recurrence.
 ap_uint<48> product=previous*alpha;
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=3
 ap_uint<49> retained=ap_uint<49>(product)+16384u;
#pragma HLS BIND_OP variable=retained op=add impl=fabric latency=1
 ap_uint<34> next=(retained>>15)+(ap_uint<34>(sum)<<4);
#pragma HLS BIND_OP variable=next op=add impl=fabric latency=1
 // This cannot clip an admitted<=4096-key row. Keep explicit saturating
 // behavior if an invalid geometry ever reaches the fixed service.
 return next>0x80000000ull?Den(0x80000000u):Den(next);
}
static float alpha_float(ap_uint<16> alpha) {
#pragma HLS INLINE
 return float(alpha.to_uint())*(1.f/32768.f);
}
static float den_float(Den den) {
#pragma HLS INLINE
 return float(den.to_uint())*(1.f/524288.f);
}
} // namespace softmax16

