#pragma once

// C207 short RMS rows only. Adjacent phase pairs are reduced in exactly the
// C205 order, but different rows occupy independent DATAFLOW stages. This is
// not an alternative sum association, reciprocal approximation, or fused FMA.
namespace c204 {
using namespace c2;
using short_norm_words = hls::stream<bus_t>;
using short_norm_scalars = hls::stream<float>;
// Raw rows are retained at activation precision, independently of the FP32
// square/statistics streams. Eight values still occupy each FIFO element.
using short_norm_raw_word = ap_uint<8*c209mp::BITS>;
using short_norm_raw_half = ap_uint<4*c209mp::BITS>;
using short_norm_raw_words = hls::stream<short_norm_raw_word>;

static void short_norm_read_square(bus_t *a, const Job &j,
                                  short_norm_raw_words &raw, short_norm_words &squares) {
#pragma HLS INLINE off
 bus_t original=0;ap_uint<128> low=0;const unsigned al=tensor::lanes(j.fa);
 short_norm_raw_half raw_low=0;
 C209_RMS_READ4:for(unsigned half=0;half<j.m*32;half++) {
#pragma HLS PIPELINE II=1 style=flp
  unsigned r=half/32,first=(half%32)*4;bool high=half&1u;
  if(first%al==0)original=a[(j.a>>5)+((uint64_t)r*j.la+first)/al];
  ap_uint<128> pack=0;
  short_norm_raw_half raw_pack=0;
  for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
   c209mp::activation cached=c209mp::narrow(tensor::lane(original,first%al+l,j.fa));
   raw_pack.range(c209mp::BITS*(l+1)-1,c209mp::BITS*l)=c209mp::bits(cached);
   float value=c209mp::widen(cached)-0.f;
   float square=value*value;
#pragma HLS BIND_OP variable=square op=fmul impl=dsp
   // Each original phase is initialized to +0 and updated once at n=128.
   float phase=0.f+square;
   pack.range(32*l+31,32*l)=fu(phase);
  }
  if(high){
   short_norm_raw_word cached=0;
   cached.range(4*c209mp::BITS-1,0)=raw_low;
   cached.range(8*c209mp::BITS-1,4*c209mp::BITS)=raw_pack;
   raw.write(cached);
   bus_t packed=0;packed.range(127,0)=low;packed.range(255,128)=pack;squares.write(packed);
  } else {low=pack;raw_low=raw_pack;}
 }
}

using short_norm_pairs=hls::stream<ap_uint<512>>;
static void short_norm_read_pairs(unsigned pairs,short_norm_words &input,
                                 short_norm_pairs &output) {
#pragma HLS INLINE off
 C207_RMS_PAIR_READ:for(unsigned i=0;i<pairs;i++) {
#pragma HLS PIPELINE II=2 style=flp
  bus_t even=input.read(),odd=input.read();
  ap_uint<512> pair=0;
  pair.range(255,0)=even;pair.range(511,256)=odd;
  output.write(pair);
 }
}

template<unsigned ADDERS>
static ap_uint<ADDERS*32> short_norm_group(bus_t word,unsigned group) {
#pragma HLS INLINE
 // Index a narrow scalar, not a 256-bit variable shift. All eight source
 // slices are literal, so this is at most an eight-way 32-bit selector.
 ap_uint<32> lanes[8];
#pragma HLS ARRAY_PARTITION variable=lanes complete
 for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
  lanes[l]=word.range(32*l+31,32*l);
 }
 ap_uint<ADDERS*32> result=0;
 for(unsigned l=0;l<ADDERS;l++) {
#pragma HLS UNROLL
  result.range(32*l+31,32*l)=lanes[group*ADDERS+l];
 }
 return result;
}

template<unsigned ADDERS>
static void short_norm_add_groups(unsigned pairs,short_norm_pairs &input,
                         hls::stream<ap_uint<ADDERS*32>> &output) {
#pragma HLS INLINE off
 const unsigned GROUPS=8/ADDERS;
 ap_uint<512> pair=0;unsigned group=0;
 C207_RMS_GROUP_ADD:for(unsigned i=0;i<pairs*GROUPS;i++) {
#pragma HLS PIPELINE II=1 style=flp
  if(group==0)pair=input.read();
  ap_uint<ADDERS*32> even=short_norm_group<ADDERS>(bus_t(pair.range(255,0)),group);
  ap_uint<ADDERS*32> odd=short_norm_group<ADDERS>(bus_t(pair.range(511,256)),group);
  ap_uint<ADDERS*32> sum=0;
  for(unsigned l=0;l<ADDERS;l++) {
#pragma HLS UNROLL
   float value=uf(even.range(32*l+31,32*l))+uf(odd.range(32*l+31,32*l));
   sum.range(32*l+31,32*l)=fu(value);
  }
  output.write(sum);group=(group+1==GROUPS)?0:group+1;
 }
}

template<unsigned ADDERS>
static void short_norm_collect_groups(unsigned pairs,
                         hls::stream<ap_uint<ADDERS*32>> &input,
                         short_norm_words &output) {
#pragma HLS INLINE off
 const unsigned GROUPS=8/ADDERS;
 ap_uint<ADDERS*32> partial[GROUPS];unsigned group=0;
#pragma HLS ARRAY_PARTITION variable=partial complete
 C207_RMS_GROUP_COLLECT:for(unsigned i=0;i<pairs*GROUPS;i++) {
#pragma HLS PIPELINE II=1 style=flp
  ap_uint<ADDERS*32> value=input.read();
  partial[group]=value;
  if(group+1==GROUPS) {
   bus_t pack=0;
   for(unsigned g=0;g<GROUPS;g++) {
#pragma HLS UNROLL
    pack.range((g+1)*ADDERS*32-1,g*ADDERS*32)=g+1==GROUPS?value:partial[g];
   }
   output.write(pack);group=0;
  } else group++;
 }
}

template<unsigned WORDS,unsigned ADDERS>
static void short_norm_phase_pairs(unsigned rows,
                         short_norm_words &input,short_norm_words &output) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 short_norm_pairs pairs;
 hls::stream<ap_uint<ADDERS*32>> groups;
#pragma HLS STREAM variable=pairs depth=2
#pragma HLS STREAM variable=groups depth=2
 short_norm_read_pairs(rows*(WORDS/2),input,pairs);
 short_norm_add_groups<ADDERS>(rows*(WORDS/2),pairs,groups);
 short_norm_collect_groups<ADDERS>(rows*(WORDS/2),groups,output);
}

static void short_norm_inverse(unsigned rows,float epsilon,
                               short_norm_words &lanes,short_norm_scalars &inverse) {
#pragma HLS INLINE off
 C207_RMS_INVERSE:for(unsigned r=0;r<rows;r++) {
#pragma HLS PIPELINE II=32 style=flp
  bus_t pack=lanes.read();
  float v[8], p[4], q[2];
#pragma HLS ARRAY_PARTITION variable=v complete
#pragma HLS ARRAY_PARTITION variable=p complete
#pragma HLS ARRAY_PARTITION variable=q complete
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   v[l]=uf(pack.range(32*l+31,32*l));
  }
  for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
   p[l]=v[2*l]+v[2*l+1];
  }
  for(unsigned l=0;l<2;l++) {
#pragma HLS UNROLL
   q[l]=p[2*l]+p[2*l+1];
  }
  float sum=q[0]+q[1];
  float variance=sum/128.f;
  if(variance<0.f)variance=0.f;
  float inv=1.f/hls::sqrtf(variance+(epsilon>0.f?epsilon:1e-6f));
  inverse.write(inv);
 }
}

static void short_norm_store(bus_t *c,const Job &j,const float gamma[128],
                             const float shift[128],short_norm_raw_words &raw,
                             short_norm_scalars &inverse) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=shift cyclic factor=8
 for(unsigned r=0;r<j.m;r++) {
  float inv=inverse.read();
  short_norm_raw_word original=0;bus_t packed=0;const unsigned cl=tensor::lanes(j.fc);
  C209_RMS_STORE4:for(unsigned half=0;half<32;half++) {
#pragma HLS PIPELINE II=1 style=flp
   bool high=half&1u;if(!high)original=raw.read();
   short_norm_raw_half selected=high?short_norm_raw_half(original.range(8*c209mp::BITS-1,4*c209mp::BITS)):short_norm_raw_half(original.range(4*c209mp::BITS-1,0));
   unsigned first=half*4;if(first%cl==0)packed=0;
   for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
    unsigned i=half*4+l;
    c209mp::word bits=selected.range(c209mp::BITS*(l+1)-1,c209mp::BITS*l);
    float centered=c209mp::widen(c209mp::from_bits(bits))-0.f;
    float scaled=centered*inv;
    float normalized=scaled*gamma[i];
    float value=normalized+((j.flags&BIAS)?shift[i]:0.f);
    packed=tensor::set_lane(packed,first%cl+l,c209mp::round(value),j.fc);
   }
   if(first%cl+4==cl)c[(j.c>>5)+((uint64_t)r*j.lc+first)/cl]=packed;
  }
 }
}

static void short_norm_rms128_dataflow(bus_t *a,bus_t *c,const Job &j,
                                     const float gamma[128],const float shift[128]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=shift cyclic factor=8
 short_norm_raw_words raw;
 short_norm_words squares,pairs8,pairs4,pairs2,lanes;
 short_norm_scalars inverse;
 // A complete row must fit before its inverse exists, including in-place
 // operation. Four rows also absorb startup latency across the phase tree.
#pragma HLS STREAM variable=raw depth=64
#pragma HLS STREAM variable=squares depth=4
#pragma HLS STREAM variable=pairs8 depth=4
#pragma HLS STREAM variable=pairs4 depth=4
#pragma HLS STREAM variable=pairs2 depth=4
#pragma HLS STREAM variable=lanes depth=4
#pragma HLS STREAM variable=inverse depth=4
 short_norm_read_square(a,j,raw,squares);
 // Thirty-two scalar-half cycles/row: the same phase tree needs only
 // 2,1,1,1 adders. Adjacent phase and final lane associations are unchanged.
 short_norm_phase_pairs<16,2>(j.m,squares,pairs8);
 short_norm_phase_pairs<8,1>(j.m,pairs8,pairs4);
 short_norm_phase_pairs<4,1>(j.m,pairs4,pairs2);
 short_norm_phase_pairs<2,1>(j.m,pairs2,lanes);
 short_norm_inverse(j.m,j.epsilon,lanes,inverse);
 short_norm_store(c,j,gamma,shift,raw,inverse);
}

static unsigned short_norm_rms128(bus_t *a,bus_t *c,bus_t *x,
                                 const Job &j,unsigned &tiles) {
#pragma HLS INLINE off
 float gamma[128],shift[128];
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=shift cyclic factor=8
 for(unsigned w=0;w<16;w++) {
#pragma HLS PIPELINE II=1
  bus_t g=x[(j.s0>>5)+w],s=0;
  if(j.flags&BIAS)s=x[(j.s1>>5)+w];
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   gamma[w*8+l]=uf(g.range(32*l+31,32*l));
   shift[w*8+l]=uf(s.range(32*l+31,32*l));
  }
 }
 short_norm_rms128_dataflow(a,c,j,gamma,shift);
 tiles+=j.m;
 return OK;
}
} // namespace c204
