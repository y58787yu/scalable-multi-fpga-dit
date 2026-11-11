#pragma once
#include "cosmos.hpp"
#include <hls_math.h>
#include "c207_norm_short.hpp"

// Included after the common bit conversion, quantization and row-load helpers.
// C205 shares the offset and squared-deviation passes in one row_sum core.
// Ordinary cached values use the selected activation precision. Statistics
// retain FP32 centered variance; the overlap profile explicitly permits a
// smaller bank-local reduction association, never a lower scalar bitwidth.
namespace c204 {
using namespace c2;
constexpr unsigned NORM_LANES=8, NORM_PHASES=16;

template<unsigned N> static float balanced_sum(const float values[N]) {
#pragma HLS INLINE
 float tree[2*N];
#pragma HLS ARRAY_PARTITION variable=tree complete
 for(unsigned i=0;i<N;i++) {
#pragma HLS UNROLL
  tree[N+i]=values[i];
 }
 for(unsigned level=N;level>1;level>>=1) {
#pragma HLS UNROLL
  for(unsigned i=0;i<N/2;i++) {
#pragma HLS UNROLL
   if(i<level/2)tree[level/2+i]=tree[level+2*i]+tree[level+2*i+1];
  }
 }
 return tree[1];
}

#if C206_NORM_SHARED
// Resource candidate: identical FP32 tree nodes, scheduled in time. This is
// deliberately NOT a serial accumulation or a reassociated reduction.
static float shared_balanced_tree(float values[NORM_PHASES],unsigned count) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=values complete
 C206_TREE_LEVEL:for(unsigned level=count;level>1;level>>=1) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
  C206_TREE_PAIR:for(unsigned pair=0;pair<level/2;pair++) {
#pragma HLS PIPELINE II=1
   float value=values[2*pair]+values[2*pair+1];
   values[pair]=value;
  }
 }
 return values[0];
}
static float reduce_phases(const float acc[NORM_PHASES][NORM_LANES]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
 float lanes[NORM_LANES],values[NORM_PHASES];
#pragma HLS ARRAY_PARTITION variable=lanes complete
#pragma HLS ARRAY_PARTITION variable=values complete
 // One runtime call site shares the same adder across all eight phase trees
 // and the final lane tree. Each child node is rounded to float unchanged.
 C206_TREE_LANE:for(unsigned lane=0;lane<=NORM_LANES;lane++) {
#pragma HLS PIPELINE off
  for(unsigned p=0;p<NORM_PHASES;p++) {
#pragma HLS UNROLL
   values[p]=lane<NORM_LANES?acc[p][lane]:(p<NORM_LANES?lanes[p]:0.f);
  }
  float value=shared_balanced_tree(values,lane<NORM_LANES?NORM_PHASES:NORM_LANES);
  if(lane<NORM_LANES)lanes[lane]=value;
  else return value;
 }
 return 0.f;
}
#else
static float reduce_phases(const float acc[NORM_PHASES][NORM_LANES]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
 float lanes[NORM_LANES];
#pragma HLS ARRAY_PARTITION variable=lanes complete
 for(unsigned l=0;l<NORM_LANES;l++) {
#pragma HLS UNROLL
  float phases[NORM_PHASES];
#pragma HLS ARRAY_PARTITION variable=phases complete
  for(unsigned p=0;p<NORM_PHASES;p++) {
#pragma HLS UNROLL
   phases[p]=acc[p][l];
  }
  lanes[l]=balanced_sum<NORM_PHASES>(phases);
 }
 return balanced_sum<NORM_LANES>(lanes);
}
#endif

// The overlap image accepts FP32 reassociation: four physical lanes retain
// sixteen recurrence phases each, rather than reconstructing eight legacy
// lane trees from 128 independently materialized phase outputs. Sixteen
// cycles exceeds the Y3 eight-cycle FP32-adder feedback latency; the real
// dependency is retained and must still be scheduled by HLS.
constexpr unsigned OVERLAP_PHASES=16;
static float reduce_overlap_phases(float ring[4][OVERLAP_PHASES]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ring complete dim=1
 float lanes[4],values[OVERLAP_PHASES];
#pragma HLS ARRAY_PARTITION variable=lanes complete
#pragma HLS ARRAY_PARTITION variable=values complete
 C209_OVERLAP_LOCAL_REDUCE:for(unsigned lane=0;lane<4;lane++) {
#pragma HLS LOOP_FLATTEN off
  for(unsigned phase=0;phase<OVERLAP_PHASES;phase++) {
#pragma HLS PIPELINE II=1
   values[phase]=ring[lane][phase];
  }
#if C206_NORM_SHARED
  lanes[lane]=shared_balanced_tree(values,OVERLAP_PHASES);
#else
  lanes[lane]=balanced_sum<OVERLAP_PHASES>(values);
#endif
 }
 return (lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
}

static float row_sum(const c209mp::activation row[FFN],unsigned n,float origin,bool square) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 float ring[4][OVERLAP_PHASES];
#pragma HLS ARRAY_PARTITION variable=ring complete dim=1
#pragma HLS BIND_STORAGE variable=ring type=ram_2p impl=lutram
 // Address a stationary phase instead of shifting 31 FP32 values per lane.
 // The first lap supplies +0 without resetting RAM or reading stale data.
 // This is an explicitly different FP32 association, never a smaller scalar
 // format or the cancellation-prone E[x*x]-E[x]^2 variance formula.
 const unsigned halves=((n+63)/64)*OVERLAP_PHASES;
 C209_ROTATING_ROW_SUM4:for(unsigned half=0;half<halves;half++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=ring type=inter direction=RAW distance=16 true
  for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
   unsigned i=half*4+l;
   float v=i<n?c209mp::widen(row[i])-origin:0.f;
   unsigned phase=half&(OVERLAP_PHASES-1);
   float previous=half<OVERLAP_PHASES?0.f:ring[l][phase];
   ring[l][phase]=previous+(square?v*v:v);
  }
 }
 return reduce_overlap_phases(ring);
}

static void row_statistics(const c209mp::activation row[FFN],unsigned n,bool rms,float epsilon,
 float &mean,float &inverse_std) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 // Summing offsets improves the mean for large DC offsets. The second pass
 // remains sum((x-mean)^2), never E[x^2]-E[x]^2.
#pragma HLS ALLOCATION function instances=row_sum limit=1
 float origin=c209mp::widen(row[0]),variance=0.f;
 mean=0.f;
 // A runtime pass loop provides one syntactic call site to the non-inlined
 // core. RMSNorm skips the offset pass. LayerNorm completes it before the
 // centered-square pass, retaining the FP32 centered-variance formula.
 for(unsigned pass=rms?1u:0u;pass<2;pass++) {
  bool square=pass==1;
  float sum=row_sum(row,n,square?mean:origin,square);
  if(square)variance=sum/float(n);
  else mean=origin+sum/float(n);
 }
 if(variance<0.f)variance=0.f;
 inverse_std=1.f/hls::sqrtf(variance+(epsilon>0.f?epsilon:1e-6f));
}

static float row_max_abs(const c209mp::activation row[FFN],unsigned n) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 // Nonnegative IEEE754 magnitudes have unsigned encoding order. This avoids
 // the large 128-leaf floating comparison tree used in the first C204 pass.
 ap_uint<31> maxima[NORM_LANES];
#pragma HLS ARRAY_PARTITION variable=maxima complete
 for(unsigned l=0;l<NORM_LANES;l++) {
#pragma HLS UNROLL
  maxima[l]=0;
 }
 for(unsigned word=0;word<(n+7)/8;word++) {
#pragma HLS PIPELINE II=1
  for(unsigned l=0;l<NORM_LANES;l++) {
#pragma HLS UNROLL
   unsigned i=word*8+l;
   ap_uint<31> magnitude=i<n?ap_uint<31>(fu(c209mp::widen(row[i]))):ap_uint<31>(0);
   // Existing float `>` ignores NaNs. Preserve this behavior, infinities,
   // subnormal magnitudes, and a +0 result for a row of either signed zero.
   if(magnitude>0x7f800000u)magnitude=0;
   if(magnitude>maxima[l])maxima[l]=magnitude;
  }
 }
 ap_uint<31> pairs[4],quads[2];
#pragma HLS ARRAY_PARTITION variable=pairs complete
#pragma HLS ARRAY_PARTITION variable=quads complete
 for(unsigned i=0;i<4;i++) {
#pragma HLS UNROLL
  pairs[i]=maxima[2*i]>maxima[2*i+1]?maxima[2*i]:maxima[2*i+1];
 }
 for(unsigned i=0;i<2;i++) {
#pragma HLS UNROLL
  quads[i]=pairs[2*i]>pairs[2*i+1]?pairs[2*i]:pairs[2*i+1];
 }
 return uf(quads[0]>quads[1]?unsigned(quads[0]):unsigned(quads[1]));
}

using norm_quant_half=ap_uint<64>;
using norm_quant_stream=hls::stream<norm_quant_half>;
static void quantize_halves(const c209mp::activation row[FFN],unsigned n,float inverse_scale,
 norm_quant_stream &halves) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=16
 for(unsigned half=0;half<4*((n+15)/16);half++) {
#pragma HLS PIPELINE II=1
  norm_quant_half pack=0;
  for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
   unsigned i=half*4+l;
   i16 v=i<n?quant(c209mp::widen(row[i]),inverse_scale):i16(0);
   pack.range(16*l+15,16*l)=ap_uint<16>(v);
  }
  halves.write(pack);
 }
}
static void store_quantized_halves(bus_t *out,uint64_t address,unsigned n,unsigned stride,
 norm_quant_stream &halves) {
#pragma HLS INLINE off
 for(unsigned word=0;word<(n+15)/16;word++) {
#pragma HLS PIPELINE II=4
  bus_t pack;pack.range(63,0)=halves.read();pack.range(127,64)=halves.read();
  pack.range(191,128)=halves.read();pack.range(255,192)=halves.read();
  out[(address>>5)+word]=pack;
 }
 // A later W8 GEMM may use a larger K padding than the FP32 input stride.
 // Clear that integer-only gap without reading beyond the logical input.
 for(unsigned word=(n+15)/16;word<stride/16;word++) {
#pragma HLS PIPELINE II=1
  out[(address>>5)+word]=0;
 }
}
static void store_quantized_row(bus_t *out,uint64_t address,const c209mp::activation row[FFN],
 unsigned n,unsigned stride,float inverse_scale) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=16
 norm_quant_stream halves;
#pragma HLS STREAM variable=halves depth=4
 quantize_halves(row,n,inverse_scale,halves);
 store_quantized_halves(out,address,n,stride,halves);
}

// Stage 10: modes 0/1 = LayerNorm/RMSNorm -> AdaLN -> INT16;
// modes 2/3 = LayerNorm/RMSNorm -> AdaLN -> FP32. s0 = affine gamma,
// s1 = AdaLN shift, s2 = AdaLN modulation scale, s3 = row quant scales.
// No affine beta is used by this fused interface (flags must be zero).
// The complete normalized row is formed before choosing its quant scale;
// quantization therefore remains per row, not per hardware tile.
// Ordinary quantization, block encoding and both normalization modes enter
// this one non-inlined implementation through one supervisor call site.
// Coefficient RAM, row load, statistics, max and quantized store are shared.
static unsigned fast_norm_dispatch(bus_t *a,bus_t *c,bus_t *x,
 const Job &j,c209mp::activation row[FFN],unsigned &tiles) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=16
 bool fused=j.op==NORM_ADALN_QUANTIZE;
 bool quant_only=j.op==QUANTIZE||j.op==PACK_BLOCK;
 if(!fused&&!quant_only&&j.op!=LAYER_NORM&&j.op!=RMS_NORM)return BAD_STAGE;
 if(fused&&(j.flags||j.mode>3))return BAD_FLAGS;
 bool output_float=!quant_only&&(!fused||(j.mode&2)!=0);
 bool rms=fused?(j.mode&1)!=0:j.op==RMS_NORM;
 const unsigned cl=output_float?tensor::lanes(j.fc):16;
 if(!j.m||j.m>((fused||quant_only)?MAX_ROWS:MAX_ROWS*16)||!j.n||j.n>FFN||j.la<j.n||j.lc<j.n||j.la%tensor::lanes(j.fa)||j.lc%cl)return BAD_SHAPE;
 // The 128-wide Q/K RMS rows have a dedicated inter-row pipeline. All other
 // stages, lengths, flags and the complete long-row implementation stay here.
 if(j.op==RMS_NORM&&j.n==128)return short_norm_rms128(a,c,x,j,tiles);
 float gamma[FFN],shift[FFN],modulation[FFN];
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=shift cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=modulation cyclic factor=8
#pragma HLS BIND_STORAGE variable=gamma type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=shift type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=modulation type=ram_2p impl=bram
 if(!quant_only) {
  frow(x,j.s0,j.n,gamma);
  if(fused||(j.flags&BIAS))frow(x,j.s1,j.n,shift);
  if(fused)frow(x,j.s2,j.n,modulation);
 }
 for(unsigned r=0;r<j.m;r++) {
  frow(a,j.a+(uint64_t)tensor::bytes(j.fa)*r*j.la,j.n,row,j.fa);float mean,inverse_std;
  if(!quant_only) {
   row_statistics(row,j.n,rms,j.epsilon,mean,inverse_std);
   bus_t pack=0;
   for(unsigned half=0;half<2*((j.n+7)/8);half++) {
#pragma HLS PIPELINE II=1
   unsigned first=half*4;if(first%cl==0)pack=0;
   for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
    unsigned i=half*4+l;float value=0;
    if(i<j.n) {
     float normalized=(c209mp::widen(row[i])-mean)*inverse_std*gamma[i];
     if(fused)value=normalized*(1.f+modulation[i])+shift[i];
     else value=normalized+((j.flags&BIAS)?shift[i]:0.f);
     if(output_float)value=c209mp::round(value);
     else row[i]=c209mp::narrow(value);
    }
    if(output_float)pack=tensor::set_lane(pack,first%cl+l,value,j.fc);
   }
    if(output_float&&(first%cl+4==cl||first+4==8*((j.n+7)/8)))
     c[(j.c>>5)+((uint64_t)r*j.lc+first)/cl]=pack;
   }
  }
  if(!output_float) {
   float mx=row_max_abs(row,j.n),scale=mx==0.f?1.f:mx/32760.f;
   writef(x,quant_only?j.s0:j.s3,r,scale);
   store_quantized_row(c,j.c+2ull*r*j.lc,row,j.n,j.lc,1.f/scale);
  }
  tiles++;
 }
 return OK;
}

// Native compatibility wrappers; production groups stages 2/3/4/8/10
// and calls fast_norm_dispatch once to prevent specialization into two cores.
static unsigned fast_quantize_stage(bus_t *a,bus_t *c,bus_t *x,const Job &j,
 c209mp::activation row[FFN],unsigned &tiles) {
#pragma HLS INLINE
 return fast_norm_dispatch(a,c,x,j,row,tiles);
}
static unsigned fast_norm_stage(bus_t *a,bus_t *c,bus_t *x,const Job &j,
 c209mp::activation row[FFN],unsigned &tiles) {
#pragma HLS INLINE
 return fast_norm_dispatch(a,c,x,j,row,tiles);
}
static unsigned fast_norm_adaln_quantize_stage(bus_t *a,bus_t *c,bus_t *x,
 const Job &j,c209mp::activation row[FFN],unsigned &tiles) {
#pragma HLS INLINE
 return fast_norm_dispatch(a,c,x,j,row,tiles);
}
} // namespace c204
