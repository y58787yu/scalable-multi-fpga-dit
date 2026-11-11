#pragma once
#include "cosmos.hpp"
#include <hls_math.h>
#include "shared_gelu.hpp"

// These helpers deliberately keep nonlinear functions out of the short,
// eight-element linear/vector output loops. A job chooses its datapath once;
// the selected datapath then processes the whole matrix as a long pipeline.
namespace c2fast {
using namespace c2;
constexpr unsigned SUM_GATED_RESIDUAL_MODE = 7;

static inline float bits_float(unsigned u) {
 union { unsigned u; float f; } v; v.u=u; return v.f;
}
static inline unsigned float_bits(float f) {
 union { unsigned u; float f; } v; v.f=f; return v.u;
}
static inline float lane_float(const bus_t &p,unsigned lane) {
 return bits_float(p.range(32*lane+31,32*lane));
}
// Already-narrow stream packets need no half->float->half conversion to
// reach F16 DDR. Retain the previous FTZ and vendor canonical-NaN store
// behavior with bit logic; F32 output still widens at the existing boundary.
static inline bus_t store_activation_lane(bus_t word,unsigned lane,
 c209mp::word value,unsigned format) {
#pragma HLS INLINE
#if C209_MIXED_PRECISION
 if(format==F16) {
  ap_uint<16> encoded=value;
  if(encoded.range(14,10)==0)encoded.range(9,0)=0;
  else if(encoded.range(14,10)==31 && encoded.range(9,0)!=0)encoded=0x7e00u;
  word.range(16*lane+15,16*lane)=encoded;return word;
 }
#endif
 return tensor::set_lane(word,lane,c209mp::widen(c209mp::from_bits(value)),format);
}
static inline float gelu_tanh(float v) {
#if C209_MIXED_PRECISION
 return c209mp::widen(c209nl::gelu(c209mp::narrow(v)));
#else
 return .5f*v*(1.f+hls::tanhf(.7978845608f*(v+.044715f*v*v*v)));
#endif
}

// Metadata is small relative to a matrix tile. Cache it once, so the hot
// epilogue has one output AXI write per eight values and no aux AXI reads.
static void load_epilogue_metadata(bus_t *aux,const Job &j,
 unsigned rb,unsigned nb,unsigned mm,unsigned nn,
 float row_scale[MAC_ROWS],float column_scale[GEMM_N],float column_bias[GEMM_N]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
 for(unsigned r=0;r<mm;r++) {
#pragma HLS PIPELINE II=1
  bus_t p=aux[(j.s0>>5)+((rb+r)>>3)];
  row_scale[r]=lane_float(p,(rb+r)&7);
 }
 for(unsigned n=0;n<nn;n+=8) {
#pragma HLS PIPELINE II=1
  bus_t p=aux[(j.s1>>5)+(nb+n)/8];
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   column_scale[n+l]=lane_float(p,l);
  }
 }
 if(j.flags&BIAS)for(unsigned n=0;n<nn;n+=8) {
#pragma HLS PIPELINE II=1
  bus_t p=aux[(j.s2>>5)+(nb+n)/8];
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   column_bias[n+l]=lane_float(p,l);
  }
 }
}

static void linear_epilogue(bus_t *output,const Job &j,
 const i48 ct[BM][BN],unsigned rb,unsigned nb,unsigned mm,unsigned nn,
 const float row_scale[MAC_ROWS],const float column_scale[GEMM_N],
 const float column_bias[GEMM_N]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ct cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
 const unsigned words=(nn+7)>>3;
 unsigned r=0,n=0;const unsigned cl=tensor::lanes(j.fc);bus_t pack=0;
 for(unsigned word=0;word<mm*words;word++) {
#pragma HLS PIPELINE II=1
  if(n%cl==0)pack=0;
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   float f=0;
   if(n+l<nn) {
    // Preserve the existing FP32 association: ((float)acc*sa)*sw + bias.
    float scaled=(float)ct[r][n+l]*row_scale[r];
    f=scaled*column_scale[n+l];
    if(j.flags&BIAS)f+=column_bias[n+l];
   }
   pack=tensor::set_lane(pack,n%cl+l,(j.flags&LOCAL_PARTIAL)?f:c209mp::round(f),j.fc);
  }
  if(n%cl+8==cl||n+8>=nn)
   output[(j.c>>5)+((uint64_t)(rb+r)*j.lc+nb+n)/cl]=pack;
  n+=8;if(n>=nn){n=0;r++;}
 }
}

// This process contains only register/wire packing and AXI writes. In
// particular, pack's recurrence never traverses a floating-point operator.
static void store_pair_words(bus_t *output,const Job &j,
 unsigned rb,unsigned nb,unsigned mm,unsigned nn,unsigned padded_n,
 pair_output_stream &pairs) {
#pragma HLS INLINE off
 unsigned r=0,n=0;bus_t pack=0;const unsigned cl=tensor::lanes(j.fc);
 const unsigned count=mm*(padded_n/2);
 for(unsigned i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
  pair_word pair=pairs.read();
  if(n%cl==0)pack=0;
  for(unsigned lane=0;lane<2;lane++) {
#pragma HLS UNROLL
   c209mp::word value=pair.range(c209mp::BITS*(lane+1)-1,c209mp::BITS*lane);
   pack=store_activation_lane(pack,n%cl+lane,value,j.fc);
  }
  if(n%cl+2==cl||n+2==padded_n) {
   if(n-n%cl<nn)output[(j.c>>5)+((uint64_t)(rb+r)*j.lc+nb+n)/cl]=pack;
  }
  n+=2;if(n>=padded_n){n=0;r++;}
 }
}

// The FP32 chain has no state between iterations: each output pair is a
// fresh 64-bit value. It is allowed to be deep while accepting a new pair
// every clock, with FIFO backpressure separated from partial-word packing.
template<unsigned Mode>
static void nonlinear_scalar_math(nonlinear_scalar_stream &input,
 scalar_output_stream &output,unsigned count) {
#pragma HLS INLINE off
 for(unsigned i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
  NonlinearScalar value=input.read();float f=0;
   if(value.valid) {
    float v=c209mp::widen(c209mp::from_bits(value.value));
    if(Mode==GELU_TANH)f=gelu_tanh(v);
    else {
#if C209_MIXED_PRECISION
     f=c209mp::widen(c209nl::silu(c209mp::narrow(v)));
#else
     f=v/(1.f+hls::expf(-v));
#endif
    }
   }
  output.write(c209mp::bits(c209mp::narrow(f)));
 }
}
template<unsigned Mode>
static void nonlinear_pair_math(nonlinear_pair_stream &input,
 pair_output_stream &output,unsigned count) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 nonlinear_scalar_stream values;scalar_output_stream results;
#pragma HLS STREAM variable=values depth=32
#pragma HLS STREAM variable=results depth=32
 nonlinear_split_pairs(input,values,count);
 nonlinear_scalar_math<Mode>(values,results,2*count);
 nonlinear_join_pairs(results,output,count);
}
static void read_ct_gelu_pairs(const i48 ct[BM][BN],const Job &j,
 unsigned mm,unsigned nn,const float row_scale[MAC_ROWS],
 const float column_scale[GEMM_N],const float column_bias[GEMM_N],
 nonlinear_pair_stream &pairs) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ct cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
 unsigned r=0,n=0;const unsigned padded_n=(nn+7)&~7u;
 for(unsigned i=0;i<mm*(padded_n/2);i++) {
#pragma HLS PIPELINE II=1
  NonlinearPair pair;pair.values=0;
  for(unsigned l=0;l<2;l++) {
#pragma HLS UNROLL
   bool valid=n+l<nn;pair.valid[l]=valid;
   float f=0;
   if(valid) {
    float scaled=(float)ct[r][n+l]*row_scale[r];
    f=scaled*column_scale[n+l];
    if(j.flags&BIAS)f+=column_bias[n+l];
   }
   pair.values.range(c209mp::BITS*l+c209mp::BITS-1,c209mp::BITS*l)=c209mp::bits(c209mp::narrow(f));
  }
  pairs.write(pair);n+=2;if(n>=padded_n){n=0;r++;}
 }
}

// Explicit three-process dataflow, rather than an FP chain in a loop which
// partially updates a pack register. This removes the deep pack Phi path.
static void gelu_epilogue_exchange(bus_t *output,const Job &j,
 const i48 ct[BM][BN],unsigned rb,unsigned nb,unsigned mm,unsigned nn,
 const float row_scale[MAC_ROWS],const float column_scale[GEMM_N],
 const float column_bias[GEMM_N],gelu_command_stream &commands,
 nonlinear_pair_stream &input,pair_output_stream &results) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ct cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
#pragma HLS DATAFLOW
#ifdef __SYNTHESIS__
 send_shared_gelu_command(commands,mm*((nn+7)/8)*4,1);
#endif
 read_ct_gelu_pairs(ct,j,mm,nn,row_scale,column_scale,column_bias,input);
#ifndef __SYNTHESIS__
 shared_gelu_pair_math(input,results,mm*((nn+7)/8)*4);
#endif
 store_pair_words(output,j,rb,nb,mm,nn,(nn+7)&~7u,results);
}
static void gelu_epilogue(bus_t *output,const Job &j,
 const i48 ct[BM][BN],unsigned rb,unsigned nb,unsigned mm,unsigned nn,
 const float row_scale[MAC_ROWS],const float column_scale[GEMM_N],
 const float column_bias[GEMM_N],gelu_command_stream &commands,
 nonlinear_pair_stream &input,pair_output_stream &results,gelu_ack_stream &ack) {
#pragma HLS INLINE off
 gelu_epilogue_exchange(output,j,ct,rb,nb,mm,nn,row_scale,column_scale,column_bias,commands,input,results);
 end_shared_gelu(ack);
}

// Replace the old gemm_stage r/n/g/l output loop with this call. ct retains
// its existing 16-way column banking. The helper does not alter tile count.
static void fast_gemm_epilogue(bus_t *output,bus_t *aux,const Job &j,
 const i48 ct[BM][BN],unsigned rb,unsigned nb,unsigned mm,unsigned nn,
 gelu_command_stream &commands,nonlinear_pair_stream &input,
 pair_output_stream &results,gelu_ack_stream &ack) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ct cyclic factor=16 dim=2
 float row_scale[MAC_ROWS],column_scale[GEMM_N],column_bias[GEMM_N];
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
 load_epilogue_metadata(aux,j,rb,nb,mm,nn,row_scale,column_scale,column_bias);
 if(j.flags&GELU)
  gelu_epilogue(output,j,ct,rb,nb,mm,nn,row_scale,column_scale,column_bias,commands,input,results,ack);
 else linear_epilogue(output,j,ct,rb,nb,mm,nn,row_scale,column_scale,column_bias);
}

// Fast modes each have a separate statically selected arithmetic path.
// Only the two nonlinear modes below instantiate transcendental functions.
// Metadata is reused across rows, and all complete input words are read
// before writing their output word, preserving exact in-place operation.
template<unsigned Mode>
static void vector_words(bus_t *a,bus_t *b,bus_t *output,bus_t *aux,
 const Job &j,unsigned &tiles) {
#pragma HLS INLINE off
 float scale[FFN],shift[FFN];
#pragma HLS ARRAY_PARTITION variable=scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=shift cyclic factor=8
#pragma HLS BIND_STORAGE variable=scale type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=shift type=ram_2p impl=bram
 if(Mode==ADALN||Mode==GATED_RESIDUAL||Mode==SUM_GATED_RESIDUAL_MODE) {
  for(unsigned n=0;n<j.n;n+=8) {
#pragma HLS PIPELINE II=1
   bus_t p=aux[(j.s0>>5)+n/8];
   for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
    float f=lane_float(p,l);
    scale[n+l]=Mode==ADALN?1.f+f:f;
   }
  }
 }
 if(Mode==ADALN)for(unsigned n=0;n<j.n;n+=8) {
#pragma HLS PIPELINE II=1
  bus_t p=aux[(j.s1>>5)+n/8];
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   shift[n+l]=lane_float(p,l);
  }
 }
 unsigned r=0,n=0;const unsigned words=(j.n+7)>>3;
 const unsigned al=tensor::lanes(j.fa),cl=tensor::lanes(j.fc);
 const bool two_inputs=Mode==ADD||Mode==MUL||Mode==GATED_RESIDUAL||Mode==SUM_GATED_RESIDUAL_MODE;
 const unsigned bl=two_inputs?tensor::lanes(j.fb):8;
 const unsigned rl=Mode==SUM_GATED_RESIDUAL_MODE?tensor::lanes(j.fs1):8;
 bus_t av=0,bv=0,rv=0,pack=0;
 for(unsigned word=0;word<j.m*words;word++) {
#pragma HLS PIPELINE II=1
  if(n%al==0)av=a[(j.a>>5)+((uint64_t)r*j.la+n)/al];
  if(n%cl==0)pack=0;
  if(two_inputs&&n%bl==0)bv=b[(j.b>>5)+((uint64_t)r*j.lb+n)/bl];
  // Mode 7: aux0 = per-column gate, aux1 = FP32 residual matrix, with ldc stride.
  if(Mode==SUM_GATED_RESIDUAL_MODE&&n%rl==0)
   rv=aux[(j.s1>>5)+((uint64_t)r*j.lc+n)/rl];
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   float f=0;
   if(n+l<j.n) {
    float v=tensor::lane(av,n%al+l,j.fa),z=two_inputs?tensor::lane(bv,n%bl+l,j.fb):0.f;
    if(Mode==ADD)f=v+z;
    else if(Mode==MUL)f=v*z;
    else if(Mode==ADALN)f=v*scale[n+l]+shift[n+l];
    else if(Mode==GATED_RESIDUAL)f=z+v*scale[n+l];
    else if(Mode==SUM_GATED_RESIDUAL_MODE) {
     // Each intermediate is FP32, matching ADD followed by GATED_RESIDUAL.
     float summed=v+z;
     float weighted=summed*scale[n+l];
     f=tensor::lane(rv,n%rl+l,j.fs1)+weighted;
    }else f=v;
   }
   pack=tensor::set_lane(pack,n%cl+l,f,j.fc);
  }
  if(n%cl+8==cl||n+8>=j.n)output[(j.c>>5)+((uint64_t)r*j.lc+n)/cl]=pack;
  n+=8;if(n>=j.n){n=0;r++;}
 }
 tiles+=j.m*words;
}

static void read_nonlinear_pairs(bus_t *a,const Job &j,nonlinear_pair_stream &pairs) {
#pragma HLS INLINE off
 unsigned r=0,n=0;const unsigned words=(j.n+7)>>3,al=tensor::lanes(j.fa);bus_t av=0;
 for(unsigned i=0;i<j.m*words*4;i++) {
#pragma HLS PIPELINE II=1
  if(n%al==0)av=a[(j.a>>5)+((uint64_t)r*j.la+n)/al];
  NonlinearPair pair;
  pair.values=pack_nonlinear_pair(tensor::lane(av,n%al,j.fa),tensor::lane(av,n%al+1,j.fa));
  for(unsigned l=0;l<2;l++) {
#pragma HLS UNROLL
   pair.valid[l]=n+l<j.n;
  }
  pairs.write(pair);
  n+=2;if(n>=words*8){n=0;r++;}
 }
}
template<unsigned Mode>
static void nonlinear_dataflow(bus_t *a,bus_t *output,const Job &j) {
#pragma HLS INLINE off
 nonlinear_pair_stream input;pair_output_stream results;
#pragma HLS STREAM variable=input depth=32
#pragma HLS STREAM variable=results depth=32
#pragma HLS DATAFLOW
 read_nonlinear_pairs(a,j,input);
 nonlinear_pair_math<Mode>(input,results,j.m*((j.n+7)/8)*4);
 store_pair_words(output,j,0,0,j.m,j.n,(j.n+7)&~7u,results);
}
template<unsigned Mode>
static void nonlinear_words(bus_t *a,bus_t *output,const Job &j,unsigned &tiles) {
#pragma HLS INLINE off
 nonlinear_dataflow<Mode>(a,output,j);
 tiles+=j.m*((j.n+7)/8);
}

static void shared_vector_gelu_exchange(bus_t *a,bus_t *output,const Job &j,
 gelu_command_stream &commands,nonlinear_pair_stream &input,pair_output_stream &results) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#ifdef __SYNTHESIS__
 send_registered_vector_gelu_command(commands,j.m,j.n,0);
#endif
 read_nonlinear_pairs(a,j,input);
#ifndef __SYNTHESIS__
 shared_gelu_pair_math(input,results,j.m*((j.n+7)/8)*4);
#endif
 store_pair_words(output,j,0,0,j.m,j.n,(j.n+7)&~7u,results);
}
static void shared_vector_gelu_words(bus_t *a,bus_t *output,const Job &j,unsigned &tiles,
 gelu_command_stream &commands,nonlinear_pair_stream &input,
 pair_output_stream &results,gelu_ack_stream &ack) {
#pragma HLS INLINE off
 shared_vector_gelu_exchange(a,output,j,commands,input,results);
 end_shared_gelu(ack);
 tiles+=j.m*((j.n+7)/8);
}

// Direct replacement for vector_stage. Existing ABI and modes 0..6 remain
// compatible. The parent should add SUM_GATED_RESIDUAL=7 to its public enum.
static unsigned fast_vector_stage(bus_t *a,bus_t *b,bus_t *output,bus_t *aux,
 const Job &j,unsigned &tiles,gelu_command_stream &commands,
 nonlinear_pair_stream &input,pair_output_stream &results,gelu_ack_stream &ack) {
#pragma HLS INLINE off
 const bool two_inputs=j.mode==ADD||j.mode==MUL||j.mode==GATED_RESIDUAL||j.mode==SUM_GATED_RESIDUAL_MODE;
 if(!j.m||j.m>MAX_ROWS||!j.n||j.n>FFN||j.la<j.n||j.lc<j.n||
    j.la%tensor::lanes(j.fa)||j.lc%tensor::lanes(j.fc)||
    (two_inputs&&(j.lb<j.n||j.lb%tensor::lanes(j.fb)))||j.mode>SUM_GATED_RESIDUAL_MODE)return BAD_SHAPE;
 switch(j.mode) {
  case ADD:vector_words<ADD>(a,b,output,aux,j,tiles);break;
  case MUL:vector_words<MUL>(a,b,output,aux,j,tiles);break;
  case ADALN:vector_words<ADALN>(a,b,output,aux,j,tiles);break;
  case GATED_RESIDUAL:vector_words<GATED_RESIDUAL>(a,b,output,aux,j,tiles);break;
  case COPY:vector_words<COPY>(a,b,output,aux,j,tiles);break;
  case SUM_GATED_RESIDUAL_MODE:vector_words<SUM_GATED_RESIDUAL_MODE>(a,b,output,aux,j,tiles);break;
  case GELU_TANH:shared_vector_gelu_words(a,output,j,tiles,commands,input,results,ack);break;
  case SILU:nonlinear_words<SILU>(a,output,j,tiles);break;
 }
 return OK;
}
}

#ifdef COSMOS_FAST_GEMM_STREAM_API
namespace c2fast {
// The producer emits every padded 16-accumulator packet. Consume padding as
// well, but issue stores only for valid output columns. The production entry
// remains four lanes; the modular owner uses two lanes with identical math.
using linear_quad_stream=hls::stream<ap_uint<128>>;
using linear_pair_stream=hls::stream<ap_uint<64>>;
static inline float linear_epilogue_scalar(i48 value,float row_scale,
 float column_scale,float bias,unsigned flags) {
#pragma HLS INLINE
 // Preserve the production associations and rounding point. In particular,
 // do not combine the two scales or fuse the optional bias addition.
 float scaled=(float)value*row_scale;
 float f=scaled*column_scale;
 if(flags&BIAS)f+=bias;
 return (flags&LOCAL_PARTIAL)?f:c209mp::round(f);
}
static void stream_linear_quads(const Job &j,
 const FastGemmTileMeta &d,result_stream &co,
 const float row_scale[MAC_ROWS],const float column_scale[GEMM_N],
 const float column_bias[GEMM_N],linear_quad_stream &out) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
 unsigned r=0,n=0,quarter=0;ap_uint<768> accum=0;
 const unsigned words=d.padded_n/4;
 for(unsigned word=0;word<d.m*words;word++) {
#pragma HLS PIPELINE II=1
  if(quarter==0)accum=co.read();
  ap_uint<192> selected=0;
  switch(quarter){case 0:selected=accum.range(191,0);break;
   case 1:selected=accum.range(383,192);break;case 2:selected=accum.range(575,384);break;
   default:selected=accum.range(767,576);break;}
  ap_uint<128> pack=0;
  for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
   float f=0;
   if(n+l<d.n) {
    i48 value=selected.range(48*l+47,48*l);
    f=linear_epilogue_scalar(value,row_scale[r],column_scale[n+l],
                            (j.flags&BIAS)?column_bias[n+l]:0.f,j.flags);
   }
   // TP partials stay FP32 through the retained SFP arithmetic reducer.
   // Ordinary activations round only AFTER INT48->FP32 scale application.
   pack.range(32*l+31,32*l)=float_bits(f);
  }
  out.write(pack);
  quarter=(quarter+1)&3u;n+=4;if(n>=d.padded_n){n=0;r++;}
 }
}
static void stream_linear_pairs(const Job &j,
 const FastGemmTileMeta &d,result_stream &co,
 const float row_scale[MAC_ROWS],const float column_scale[GEMM_N],
 const float column_bias[GEMM_N],linear_pair_stream &out) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=2
 unsigned r=0,n=0,pair=0;ap_uint<768> accum=0;
 const unsigned words=d.padded_n/2;
 for(unsigned word=0;word<d.m*words;word++) {
#pragma HLS PIPELINE II=1
  if(pair==0)accum=co.read();
  ap_uint<96> selected=0;
  switch(pair) {
   case 0:selected=accum.range(95,0);break;
   case 1:selected=accum.range(191,96);break;
   case 2:selected=accum.range(287,192);break;
   case 3:selected=accum.range(383,288);break;
   case 4:selected=accum.range(479,384);break;
   case 5:selected=accum.range(575,480);break;
   case 6:selected=accum.range(671,576);break;
   default:selected=accum.range(767,672);break;
  }
  ap_uint<64> pack=0;
  for(unsigned lane=0;lane<2;lane++) {
#pragma HLS UNROLL
   float f=0;
   if(n+lane<d.n) {
    i48 value=selected.range(48*lane+47,48*lane);
    f=linear_epilogue_scalar(value,row_scale[r],column_scale[n+lane],
                            (j.flags&BIAS)?column_bias[n+lane]:0.f,j.flags);
   }
   pack.range(32*lane+31,32*lane)=float_bits(f);
  }
  out.write(pack);
  pair=(pair+1)&7u;n+=2;if(n>=d.padded_n){n=0;r++;}
 }
}
static void store_linear_quads(bus_t *output,const Job &j,const FastGemmTileMeta &d,
 linear_quad_stream &input) {
#pragma HLS INLINE off
 unsigned r=0,n=0;const unsigned cl=tensor::lanes(j.fc);bus_t packed=0;
 for(unsigned quad=0;quad<d.m*(d.padded_n/4);quad++) {
#pragma HLS PIPELINE II=1
  ap_uint<128> values=input.read();if(n%cl==0)packed=0;
  for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
   packed=tensor::set_lane(packed,n%cl+l,bits_float(values.range(32*l+31,32*l)),j.fc);
  }
  if(n%cl+4==cl||n+4==d.padded_n)
   if(n-n%cl<d.n)output[(j.c>>5)+((uint64_t)(d.rb+r)*j.lc+d.nb+n)/cl]=packed;
  n+=4;if(n>=d.padded_n){n=0;r++;}
 }
}
static void stream_linear_tile(bus_t *output,const Job &j,
 const FastGemmTileMeta &d,result_stream &co,const float row_scale[MAC_ROWS],
 const float column_scale[GEMM_N],const float column_bias[GEMM_N]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
 linear_quad_stream quads;
#pragma HLS STREAM variable=quads depth=8
 stream_linear_quads(j,d,co,row_scale,column_scale,column_bias,quads);
 store_linear_quads(output,j,d,quads);
}
static void read_stream_gelu_pairs(const Job &j,
 const FastGemmTileMeta &d,result_stream &co,
 const float row_scale[MAC_ROWS],const float column_scale[GEMM_N],
 const float column_bias[GEMM_N],nonlinear_pair_stream &pairs) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
 unsigned r=0,n=0,group=0;ap_uint<768> accum=0;
 const unsigned count=d.m*(d.padded_n/2);
 for(unsigned i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
  if(group==0)accum=co.read();
#if C206_CONSTANT_UNPACK
  ap_uint<96> selected=0;
  switch(group) {
   case 0:selected=accum.range(95,0);break;
   case 1:selected=accum.range(191,96);break;
   case 2:selected=accum.range(287,192);break;
   case 3:selected=accum.range(383,288);break;
   case 4:selected=accum.range(479,384);break;
   case 5:selected=accum.range(575,480);break;
   case 6:selected=accum.range(671,576);break;
   default:selected=accum.range(767,672);break;
  }
#endif
  NonlinearPair pair;pair.values=0;
  for(unsigned l=0;l<2;l++) {
#pragma HLS UNROLL
   unsigned pos=2*group+l;bool valid=n+l<d.n;pair.valid[l]=valid;
#if C206_CONSTANT_UNPACK
   i48 value=selected.range(48*l+47,48*l);
#else
   i48 value=accum.range(48*pos+47,48*pos);
#endif
   float f=0;
   if(valid) {
    float scaled=(float)value*row_scale[r];
    f=scaled*column_scale[n+l];
    if(j.flags&BIAS)f+=column_bias[n+l];
   }
   pair.values.range(c209mp::BITS*l+c209mp::BITS-1,c209mp::BITS*l)=c209mp::bits(c209mp::narrow(f));
  }
  pairs.write(pair);
  group=(group+1)&7;n+=2;if(n>=d.padded_n){n=0;r++;}
 }
}
static void stream_gelu_tile_exchange(bus_t *output,const Job &j,
 const FastGemmTileMeta &d,result_stream &co,
 const float row_scale[MAC_ROWS],const float column_scale[GEMM_N],
 const float column_bias[GEMM_N],gelu_command_stream &commands,
 nonlinear_pair_stream &input,pair_output_stream &results) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
#pragma HLS DATAFLOW
#ifdef __SYNTHESIS__
 send_shared_gelu_command(commands,d.m*(d.padded_n/2),1);
#endif
 read_stream_gelu_pairs(j,d,co,row_scale,column_scale,column_bias,input);
#ifndef __SYNTHESIS__
 shared_gelu_pair_math(input,results,d.m*(d.padded_n/2));
#endif
 store_pair_words(output,j,d.rb,d.nb,d.m,d.n,d.padded_n,results);
}
static void stream_gelu_tile(bus_t *output,const Job &j,
 const FastGemmTileMeta &d,result_stream &co,
 const float row_scale[MAC_ROWS],const float column_scale[GEMM_N],const float column_bias[GEMM_N],
 gelu_command_stream &commands,nonlinear_pair_stream &input,
 pair_output_stream &results,gelu_ack_stream &ack) {
#pragma HLS INLINE off
 stream_gelu_tile_exchange(output,j,d,co,row_scale,column_scale,column_bias,commands,input,results);
 end_shared_gelu(ack);
}
}

// Global overload required by the streamed GEMM dataflow wrapper. Header
// include order: fast_gemm.hpp (meta definitions), then fast_vector.hpp.
static void fast_gemm_epilogue(c2::bus_t *output,c2::bus_t *aux,
 const c2::Job &j,fast_gemm_meta_stream &meta,result_stream &co,
 c2fast::gelu_command_stream &commands,c2fast::nonlinear_pair_stream &input,
 c2fast::pair_output_stream &results,c2fast::gelu_ack_stream &ack) {
#pragma HLS INLINE off
 using namespace c2;
 float row_scale[MAC_ROWS],column_scale[GEMM_N],column_bias[GEMM_N];
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=column_bias cyclic factor=8
 bool done=false;
 while(!done) {
  FastGemmTileMeta d=meta.read();done=d.stop;
  if(!done) {
   c2fast::load_epilogue_metadata(aux,j,d.rb,d.nb,d.m,d.n,
    row_scale,column_scale,column_bias);
   if(j.flags&GELU)c2fast::stream_gelu_tile(output,j,d,co,row_scale,column_scale,column_bias,commands,input,results,ack);
   else c2fast::stream_linear_tile(output,j,d,co,row_scale,column_scale,column_bias);
  }
 }
}
#endif
