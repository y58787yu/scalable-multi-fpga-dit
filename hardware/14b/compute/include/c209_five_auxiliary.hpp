#pragma once
// Fixed five-shard peripheral entry points. Typed ranges and lifetimes are
// bound by the fixed runtime; PL validates only its finite record contract.
#include "c209_fixed_auxiliary.hpp"
#include "c209_five_protocol.hpp"
#include "c209_five_fixed_job.hpp"
#include "c209_five_rms_service.hpp"
#include "c209_aux_quant_service.hpp"
#include "c209_five_prepare_service.hpp"
#include "c209_five_aux_row_pipeline.hpp"
#include "c209_overlap_scale.hpp"

namespace c209five {
using namespace c2;
// One supervisor-owned scratch shared by the sequential stage2/14 entries.
// Production callers supply it explicitly: neither fixed entry allocates a
// private long-row cache. The legacy generic auxiliary switch is not called.
using AuxRow=c209mp::activation[c209fixed::MAX_LOCAL_COLS];

// One production call site owns row maximum/scale and the shared quantizer.
// AdaLN is only the optional preprocessing phase of this same row loop.
static unsigned quantize_row_service(bus_t *a,bus_t *c,bus_t *x,const FixedJob &j,AuxRow &row,
 unsigned &tiles,AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 bool adaln=j.op==NORM_ADALN_FIXED,patch=!adaln&&j.n==68;
 ap_uint<13> n=adaln?5120u:j.n.to_uint(),stride=adaln?5120u:j.lc.to_uint();
#ifndef __SYNTHESIS__
 if(!adaln&&((patch?(j.fa!=F32||j.la!=80||j.lc!=80):
   ((j.n!=1280)||j.fa!=F16||j.la!=j.n||j.lc!=j.n))||j.fc!=I16))return BAD_SHAPE;
#endif
 aux_row_detail::Coefficients coefficients;
#pragma HLS ARRAY_PARTITION variable=coefficients cyclic factor=4
#pragma HLS BIND_STORAGE variable=coefficients type=ram_2p impl=bram
 float fixed_scale=1.f,fixed_inverse=1.f;
 if(adaln) {
  // Reject a wrong record before entering the finite request/response phase.
  if(!load_overlap_scale(x,j.s0,fixed_scale,fixed_inverse))return BAD_ABI;
  aux_row_detail::load_coefficients(x,j.s1,j.s2,coefficients);
 }
 tiles=0;
 ap_uint<15> input_stride=adaln?20480u:patch?320u:2560u;
 ap_uint<14> output_stride=patch?160u:adaln?10240u:2560u;
 aux_row_detail::pipeline(a,c,x,j,row,coefficients,n,stride,input_stride,output_stride,
                          adaln,adaln||patch,requests,results,fixed_scale,fixed_inverse);
 tiles=j.m;
 return OK;
}

#ifndef __SYNTHESIS__
// Historical host oracle only. Production stage4/17 both use the one shared
// rms128_service call site; these bodies cannot instantiate hardware.
static unsigned rms_fixed(bus_t *a,bus_t *c,bus_t *x,const Job &j,unsigned &tiles) {
#pragma HLS INLINE off
 tiles=0;
 if(j.op!=RMS_NORM)return BAD_STAGE;
 if(j.n!=128||j.la!=128||j.lc!=128||j.fa!=F16||j.fc!=F16)return BAD_SHAPE;
 c209fixed::auxiliary_detail::rms_rows<c209tp4::MAX_LOCAL_ROWS*c209tp4::LOCAL_HEADS>(a,c,x,j,tiles,j.m);
 return OK;
}
#endif

static unsigned prepare_global(bus_t *a,bus_t *b,bus_t *c,bus_t *x,
 const FixedJob &source,unsigned &tiles,AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
 return prepare_self_quant_service(a,b,c,x,source,tiles,source.m,requests,results);
}

#ifndef __SYNTHESIS__
// Historical Stage17 host oracle: token-major Q or K, F16 [rows,10,128]. Keep the RMS gamma,
// both original F16 rounding boundaries, and the split-half RoPE mapping.
// s0=gamma[128], s1=cos[rows,128], s2=sin[rows,128].
// Each activation word crosses DDR once in each direction; the intermediate
// normalized vector is held in the local 128-value scratch, never DDR.
static unsigned rms_rope(bus_t *a,bus_t *c,bus_t *x,const Job &j,unsigned &tiles) {
#pragma HLS INLINE off
 float gamma[128],cosine[128],sine[128];
 c209mp::activation row[128],normalized[128];
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=cosine cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=sine cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=normalized cyclic factor=8
 for(unsigned w=0;w<16;w++) {
#pragma HLS PIPELINE II=1
  bus_t p=x[(j.s0>>5)+w];
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   gamma[w*8+l]=uf(p.range(32*l+31,32*l));
  }
 }
 tiles=0;
 for(unsigned r=0;r<j.m;r++) {
  for(unsigned w=0;w<16;w++) {
#pragma HLS PIPELINE II=1
   bus_t cv=x[(j.s1>>5)+r*16+w],sv=x[(j.s2>>5)+r*16+w];
   for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
    cosine[w*8+l]=uf(cv.range(32*l+31,32*l));
    sine[w*8+l]=uf(sv.range(32*l+31,32*l));
   }
  }
  for(unsigned h=0;h<c209tp4::LOCAL_HEADS;h++) {
   for(unsigned w=0;w<8;w++) {
#pragma HLS PIPELINE II=1
    bus_t p=a[(j.a>>5)+(uint64_t(r)*j.la+h*128)/16+w];
    for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
     row[w*16+l]=c209fixed::auxiliary_detail::input_half(ap_uint<16>(p.range(16*l+15,16*l)));
    }
   }
   float mean,inverse;
   // row_statistics(rms=true,n=128) uses the same16-phase then8-lane
   // adjacent-pair FP32 tree as short_norm_rms128; no reassociation.
   c204::row_statistics(row,128,true,1e-6f,mean,inverse);
   for(unsigned d=0;d<128;d+=4) {
#pragma HLS PIPELINE II=1
    for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
     float centered=c209mp::widen(row[d+l])-0.f;
     float scaled=centered*inverse;
     float value=scaled*gamma[d+l]+0.f;
     normalized[d+l]=c209mp::narrow(c209mp::round(value));
    }
   }
   ap_uint<128> low=0;
   for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
    ap_uint<128> half=0;
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     unsigned z=d+l;
     float value=c209mp::widen(normalized[z]);
     float rotated=z<64?-c209mp::widen(normalized[z+64]):c209mp::widen(normalized[z-64]);
     float rounded=c209mp::round(value*cosine[z]+rotated*sine[z]);
     half.range(16*l+15,16*l)=c209mp::bits(c209mp::narrow(rounded));
    }
    if(!(d&8u))low=half;
    else {bus_t p=0;p.range(127,0)=low;p.range(255,128)=half;
     c[(j.c>>5)+(uint64_t(r)*j.lc+h*128+d)/16]=p;}
   }
  }
  tiles++;
 }
 return OK;
}
#endif

// Stage15 uses the existing shared single GELU service. This adapter owns
// only a tile's linear scale/round/packetization and final F16 store; it does
// not instantiate a private GELU lane for each matrix shard.
static void linear_gelu_epilogue(bus_t *c,const Job &j,unsigned rb,unsigned nb,
 unsigned rows,unsigned n,result_stream &dots,const float row_scale[256],
 const float column_scale[128],c2fast::gelu_command_stream &commands,
 c2fast::nonlinear_pair_stream &input,c2fast::pair_output_stream &results,
 c2fast::gelu_ack_stream &ack) {
#pragma HLS INLINE off
 float unused_bias[128];
 Job ordinary=j;ordinary.flags=0;ordinary.fc=F16;
 FastGemmTileMeta tile={rb,nb,rows,n,n,false};
 c2fast::stream_gelu_tile(c,ordinary,tile,dots,row_scale,column_scale,unused_bias,commands,input,results,ack);
}

static void linear_gelu_input(unsigned rows,unsigned n,result_stream &dots,
 const float row_scale[256],const float column_scale[128],
 c2fast::gelu_command_stream &commands,c2fast::nonlinear_pair_stream &input) {
#pragma HLS INLINE off
 Job j={};FastGemmTileMeta tile={0,0,rows,n,n,false};float unused_bias[128];
#ifdef __SYNTHESIS__
 // Keep the registered command boundary used by the production blackbox.
 // Fixed n is divisible by8: rows*ceil(n/8)*4 == rows*(n/2).
 c2fast::send_registered_vector_gelu_command(commands,rows,n,1);
#endif
 c2fast::read_stream_gelu_pairs(j,tile,dots,row_scale,column_scale,unused_bias,input);
}
static void linear_gelu_output(unsigned rows,unsigned n,c2fast::pair_output_stream &results,
 hls::stream<ap_uint<128>> &output) {
#pragma HLS INLINE off
 ap_uint<128> pack=0;
 for(unsigned p=0;p<rows*(n/2);p++) {
#pragma HLS PIPELINE II=1
  c2fast::pair_word value=results.read();ap_uint<32> pair=0;
  pair.range(15,0)=c209fixed::auxiliary_detail::stored_half(c209mp::word(value.range(15,0)));
  pair.range(31,16)=c209fixed::auxiliary_detail::stored_half(c209mp::word(value.range(31,16)));
  pack>>=32;pack.range(127,96)=pair;
  if((p&3u)==3)output.write(pack);
 }
}
static void linear_gelu_exchange(unsigned rows,unsigned n,result_stream &dots,
 const float row_scale[256],const float column_scale[128],hls::stream<ap_uint<128>> &output,
 c2fast::gelu_command_stream &commands,c2fast::nonlinear_pair_stream &input,
 c2fast::pair_output_stream &results) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 linear_gelu_input(rows,n,dots,row_scale,column_scale,commands,input);
#ifndef __SYNTHESIS__
 c2fast::shared_gelu_pair_math(input,results,rows*(n/2));
#endif
 linear_gelu_output(rows,n,results,output);
}
static void linear_gelu_stream(unsigned rows,unsigned n,result_stream &dots,
 const float row_scale[256],const float column_scale[128],hls::stream<ap_uint<128>> &output,
 c2fast::gelu_command_stream &commands,c2fast::nonlinear_pair_stream &input,
 c2fast::pair_output_stream &results,c2fast::gelu_ack_stream &ack) {
#pragma HLS INLINE off
 linear_gelu_exchange(rows,n,dots,row_scale,column_scale,output,commands,input,results);
 c2fast::end_shared_gelu(ack);
}
} // namespace c209five
