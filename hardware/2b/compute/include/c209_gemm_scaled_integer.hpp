#pragma once
#include <ap_int.h>
#include <stdint.h>

// Bounded precision GEMM epilogue. Keep the complete INT48 dot until its
// exponent is known; never cast it, or a small dynamic scale, to binary16.
// Three independent 16-significant-bit RNE mantissas feed integer products.
// The scale exponent retains the full finite binary32 range, including
// subnormals. External partial results remain IEEE binary32 bit patterns.
// For a nonzero finite result away from overflow/underflow, mantissa rounding
// contributes at most approximately 3*2^-16 relative error (<4.6e-5), plus
// final binary32 rounding. This is intentionally not bit-exact X2 FP32 math.
namespace c209gemmscale {
using Scale=ap_uint<26>; // sign, signed9 exponent, unsigned16 significand

// Constant-shift leading-zero networks. Do not replace these with a
// variable49-bit shift/mask/comparison: that caused the first Y scaler's
// 7k-LUT expression cone. The result is the same RNE significand, obtained
// from fixed guard/sticky positions after ONE normalization network.
static ap_uint<48> normalize48(ap_uint<48> value,ap_uint<6> &shift) {
#pragma HLS INLINE
 shift=0;
 if(value.range(47,16)==0){value<<=32;shift+=32;}
 if(value.range(47,32)==0){value<<=16;shift+=16;}
 if(value.range(47,40)==0){value<<=8;shift+=8;}
 if(value.range(47,44)==0){value<<=4;shift+=4;}
 if(value.range(47,46)==0){value<<=2;shift+=2;}
 if(!value[47]){value<<=1;shift+=1;}
 return value;
}
static ap_uint<24> normalize24(ap_uint<24> value,ap_uint<5> &shift) {
#pragma HLS INLINE
 shift=0;
 if(value.range(23,8)==0){value<<=16;shift+=16;}
 if(value.range(23,16)==0){value<<=8;shift+=8;}
 if(value.range(23,20)==0){value<<=4;shift+=4;}
 if(value.range(23,22)==0){value<<=2;shift+=2;}
 if(!value[23]){value<<=1;shift+=1;}
 return value;
}
static ap_uint<16> round16(ap_uint<16> leading,bool guard,bool sticky,
                           ap_int<10> &exponent) {
#pragma HLS INLINE
 ap_uint<17> rounded=ap_uint<17>(leading)+ap_uint<1>(guard&&(sticky||leading[0]));
 if(rounded[16]){exponent++;return ap_uint<16>(0x8000u);}
 return ap_uint<16>(rounded);
}
// A normalized FP32 mantissa plus guard and sticky is only26 bits. Jam
// discarded bits into bit0 at each constant shift, so a subnormal uses
// one bounded network rather than another general49-bit rounding engine.
static ap_uint<26> shift_right_jam26(ap_uint<26> value,ap_uint<5> distance) {
#pragma HLS INLINE
 if(distance[4]){bool lost=value.range(15,0)!=0;value>>=16;value[0]=bool(value[0])||lost;}
 if(distance[3]){bool lost=value.range(7,0)!=0;value>>=8;value[0]=bool(value[0])||lost;}
 if(distance[2]){bool lost=value.range(3,0)!=0;value>>=4;value[0]=bool(value[0])||lost;}
 if(distance[1]){bool lost=value.range(1,0)!=0;value>>=2;value[0]=bool(value[0])||lost;}
 if(distance[0]){bool lost=value[0];value>>=1;value[0]=bool(value[0])||lost;}
 return value;
}

static Scale encode_scale_bits(ap_uint<32> bits) {
#pragma HLS INLINE
 Scale out=0;out[25]=bits[31];
 unsigned exponent=bits.range(30,23).to_uint();
 ap_uint<23> fraction=bits.range(22,0);
 ap_int<10> e=0;ap_uint<16> m=0;
 if(exponent==255) {
  // Reserved exponents preserve exceptional values without floating math.
  e=fraction==0?255:254;
 } else if(exponent!=0||fraction!=0) {
  ap_uint<24> aligned=(ap_uint<24>(1)<<23)|fraction;
  ap_uint<5> shift=0;
  if(exponent==0)aligned=normalize24(ap_uint<24>(fraction),shift);
  e=exponent?ap_int<10>(int(exponent)-127):ap_int<10>(-126-ap_int<10>(shift));
  m=round16(aligned.range(23,8),aligned[7],aligned.range(6,0)!=0,e);
 }
 out.range(15,0)=m;out.range(24,16)=ap_int<9>(e);
 return out;
}

static ap_uint<32> scaled_dot_bits(ap_int<48> dot,Scale row,Scale column) {
#pragma HLS INLINE
 bool sign=dot[47]^row[25]^column[25];
 ap_uint<32> encoded=0;encoded[31]=sign;
 ap_int<10> re=ap_int<9>(row.range(24,16)),ce=ap_int<9>(column.range(24,16));
 ap_uint<16> rm=row.range(15,0),cm=column.range(15,0);
 bool zero=dot==0||((rm==0)&&re<254)||((cm==0)&&ce<254);
 if(re==254||ce==254||((re==255||ce==255)&&zero))return ap_uint<32>(0x7fc00000u);
 if(re==255||ce==255){encoded.range(30,23)=255;return encoded;}
 if(zero)return encoded;
 ap_uint<48> magnitude=dot[47]?ap_uint<48>(-ap_int<49>(dot)):ap_uint<48>(dot);
 ap_uint<6> shift=0;ap_uint<48> normalized=normalize48(magnitude,shift);
 ap_int<10> de=47-ap_int<10>(shift);
 ap_uint<16> dm=round16(normalized.range(47,32),normalized[31],normalized.range(30,0)!=0,de);
 ap_uint<32> first=dm*rm;
#pragma HLS BIND_OP variable=first op=mul impl=dsp
 ap_uint<48> product=first*cm;
#pragma HLS BIND_OP variable=product op=mul impl=dsp
 // All finite nonzero normalized significands have their top bit at15.
 ap_uint<2> increment=0;ap_uint<48> aligned;
 if(product[47]){aligned=product;increment=2;}
 else if(product[46]){aligned=product<<1;increment=1;}
 else aligned=product<<2;
 ap_int<11> result_exponent=ap_int<11>(de)+ap_int<11>(re)+ap_int<11>(ce)+ap_int<11>(increment);
 if(result_exponent>127){encoded.range(30,23)=255;return encoded;}
 if(result_exponent < -150)return encoded;
 ap_uint<26> significant=0;
 significant.range(25,1)=aligned.range(47,23);significant[0]=aligned.range(22,0)!=0;
 bool subnormal=result_exponent < -126;
 if(subnormal)significant=shift_right_jam26(significant,ap_uint<5>(-126-result_exponent));
 ap_uint<24> leading=significant.range(25,2);
 ap_uint<25> rounded=ap_uint<25>(leading)+ap_uint<1>(significant[1]&&(significant[0]||significant[2]));
 if(subnormal) {
  // Rounding a subnormal upward to 0x00800000 is the min-normal encoding.
  encoded.range(30,0)=ap_uint<31>(rounded);return encoded;
 }
 if(rounded[24]){rounded>>=1;result_exponent++;}
 if(result_exponent>127){encoded.range(30,23)=255;return encoded;}
 encoded.range(30,23)=ap_uint<8>(result_exponent+127);
 encoded.range(22,0)=rounded.range(22,0);
 return encoded;
}
} // namespace c209gemmscale
