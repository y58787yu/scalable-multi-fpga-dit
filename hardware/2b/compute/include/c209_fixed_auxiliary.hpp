#pragma once
// Fixed production auxiliary call graph. The old generic implementations stay
// available as numerical references, but no generic dispatcher is called here.
#include "c209_compute_support.hpp"
#include "c209_fixed_profile.hpp"
#include "../src/c207_mac_stream.hpp"
#include "../src/shared_gelu.hpp"
#include "../src/fast_gemm.hpp"
#include "../src/fast_vector.hpp"
#include "../src/c205_norm.hpp"
#include "../src/c205_mac.hpp"
#include "../src/c204_attention.hpp"

namespace c209fixed {
static_assert(c209mp::BITS==16,"Fixed production ordinary storage is 16 bits");
namespace auxiliary_detail {

// Retain the existing FTZ and canonical-NaN store semantics for packets that
// have already passed through the one global narrow GELU arithmetic service.
static ap_uint<16> stored_half(c209mp::word value) {
#pragma HLS INLINE
 ap_uint<16> bits=value;
 if(bits.range(14,10)==0)bits.range(9,0)=0;
 else if(bits.range(14,10)==31&&bits.range(9,0)!=0)bits=0x7e00u;
 return bits;
}

// Exact half -> existing widen -> existing narrow roundtrip, verified against
// all 65536 half encodings by the native fixture. Avoid instantiating two
// floating conversion cores merely to retain a packed F16 input value.
static c209mp::activation input_half(ap_uint<16> bits) {
#pragma HLS INLINE
 return c209mp::from_bits(stored_half(bits));
}
static ap_uint<128> half_word(bus_t packed,bool high) {
#pragma HLS INLINE
 return high?ap_uint<128>(packed.range(255,128)):ap_uint<128>(packed.range(127,0));
}
static ap_uint<64> quarter_word(bus_t packed,unsigned quarter) {
#pragma HLS INLINE
 switch(quarter) {
  case 0:return packed.range(63,0);case 1:return packed.range(127,64);
  case 2:return packed.range(191,128);default:return packed.range(255,192);
 }
}
static void load_activation_row(bus_t *a,uint64_t address,ap_uint<13> count,bool input_f32,
 c209mp::activation row[MAX_LOCAL_COLS]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 bus_t packed=0;
 C209_FIXED_LOAD_ROW8:for(ap_uint<13> first=0;first<count;first+=8) {
#pragma HLS PIPELINE II=1
  if(input_f32||!(first&8u))packed=a[(address>>5)+(input_f32?(first>>3):(first>>4))];
  ap_uint<128> selected=half_word(packed,bool(first&8u));
  for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
   if(first+lane<count) {
    // Both extraction widths/offsets are literal after lane unrolling. Only
    // the complete 128-bit half has a two-way selector, never eight 256-bit
    // barrel shifters driven by a runtime tensor::lane index.
    c209mp::activation value=input_f32?c209mp::narrow(uf(packed.range(32*lane+31,32*lane))):
      input_half(ap_uint<16>(selected.range(16*lane+15,16*lane)));
    row[first+lane]=value;
   }
  }
 }
}

static Job addresses(const Job &source) {
#pragma HLS INLINE
 Job j={};j.a=source.a;j.b=source.b;j.c=source.c;
 j.s0=source.s0;j.s1=source.s1;j.s2=source.s2;j.s3=source.s3;
 j.fa=j.fb=j.fc=j.fs0=j.fs1=j.fs2=j.fs3=UNUSED;
 j.abi_version=2;return j;
}

// The array objects are rank-local4096 / norm2048 entries; physical lane counts are unchanged. The reused numerical helpers'
// array arguments decay to pointers and allocate no separate FFN-sized array.
// Their maximum accessed index here is the fixed operation's canonical n.
template<unsigned ROWS>
static void long_rows(bus_t *a,bus_t *c,bus_t *x,const Job &j,
 c209mp::activation row[MAX_LOCAL_COLS],unsigned &tiles,unsigned rows=ROWS) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 // Legacy stage10 numerical reference. The five-shard production stage14
 // has its own fixed body and never calls this dynamic dispatcher/helper.
 const bool fused=j.op==NORM_ADALN_QUANTIZE;
 float gamma[NORM_WIDTH],shift[NORM_WIDTH],modulation[NORM_WIDTH];
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=shift cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=modulation cyclic factor=8
#pragma HLS BIND_STORAGE variable=gamma type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=shift type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=modulation type=ram_2p impl=bram
 if(fused) {
  frow(x,j.s0,NORM_WIDTH,gamma,F32);
  frow(x,j.s1,NORM_WIDTH,shift,F32);
  frow(x,j.s2,NORM_WIDTH,modulation,F32);
 }
 for(unsigned r=0;r<rows;r++) {
  const bool input_f32=j.fa==F32;
  uint64_t offset=uint64_t(r)*j.la;
  load_activation_row(a,j.a+(input_f32?(offset<<2):(offset<<1)),j.n,input_f32,row);
  if(fused) {
   float mean,inverse_std;
   c204::row_statistics(row,NORM_WIDTH,false,1e-6f,mean,inverse_std);
   C209_FIXED_LN_ADALN4:for(unsigned half=0;half<NORM_WIDTH/4;half++) {
#pragma HLS PIPELINE II=1
    for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
     unsigned i=half*4+lane;
     float normalized=(c209mp::widen(row[i])-mean)*inverse_std*gamma[i];
     float value=normalized*(1.f+modulation[i])+shift[i];
     row[i]=c209mp::narrow(value);
    }
   }
  }
  float maximum=c204::row_max_abs(row,j.n);
  float scale=maximum==0.f?1.f:maximum/32760.f;
  writef(x,fused?j.s3:j.s0,r,scale);
  c204::store_quantized_row(c,j.c+2ull*r*j.lc,row,j.n,j.lc,1.f/scale);
  tiles++;
 }
}

template<unsigned ROWS>
static void rms_store(bus_t *c,uint64_t address,const float gamma[128],
 c204::short_norm_raw_words &raw,c204::short_norm_scalars &inverse,unsigned rows=ROWS) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
 ap_uint<64> quarters[4];
#pragma HLS ARRAY_PARTITION variable=quarters complete
 for(unsigned r=0;r<rows;r++) {
  float inv=inverse.read();c204::short_norm_raw_word original=0;
  C209_FIXED_RMS_STORE4:for(unsigned half=0;half<32;half++) {
#pragma HLS PIPELINE II=1 style=flp
   bool high=half&1u;if(!high)original=raw.read();
   ap_uint<64> selected=high?ap_uint<64>(original.range(127,64)):ap_uint<64>(original.range(63,0));
   ap_uint<64> quarter=0;
   for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
    c209mp::word bits=selected.range(16*lane+15,16*lane);
    float centered=c209mp::widen(c209mp::from_bits(bits))-0.f;
    float scaled=centered*inv;
    float normalized=scaled*gamma[half*4+lane];
    // Preserve the old no-BIAS addition and both existing rounding boundaries.
    float value=normalized+0.f;
    quarter.range(16*lane+15,16*lane)=c209mp::bits(c209mp::narrow(c209mp::round(value)));
   }
   unsigned slot=half&3u;quarters[slot]=quarter;
   if(slot==3) {
    bus_t packed;
    packed.range(63,0)=quarters[0];packed.range(127,64)=quarters[1];
    packed.range(191,128)=quarters[2];packed.range(255,192)=quarter;
    c[(address>>5)+uint64_t(r)*8+half/4]=packed;
   }
  }
 }
}

template<unsigned ROWS>
static void rms_read_square(bus_t *a,uint64_t address,
 c204::short_norm_raw_words &raw,c204::short_norm_words &squares,unsigned rows=ROWS) {
#pragma HLS INLINE off
 bus_t packed=0;ap_uint<64> raw_low=0;ap_uint<128> square_low=0;
 C209_FIXED_RMS_READ4:for(unsigned half=0;half<rows*32;half++) {
#pragma HLS PIPELINE II=1 style=flp
  if(!(half&3u))packed=a[(address>>5)+(half>>2)];
  ap_uint<64> selected=quarter_word(packed,half&3u),raw_quarter=0;
  ap_uint<128> square_quarter=0;
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   c209mp::activation cached=input_half(ap_uint<16>(selected.range(16*lane+15,16*lane)));
   raw_quarter.range(16*lane+15,16*lane)=c209mp::bits(cached);
   float value=c209mp::widen(cached)-0.f;
   float square=value*value;
#pragma HLS BIND_OP variable=square op=fmul impl=dsp
   float phase=0.f+square;
   square_quarter.range(32*lane+31,32*lane)=fu(phase);
  }
  if(half&1u) {
   c204::short_norm_raw_word saved;
   saved.range(63,0)=raw_low;saved.range(127,64)=raw_quarter;raw.write(saved);
   bus_t sum;sum.range(127,0)=square_low;sum.range(255,128)=square_quarter;squares.write(sum);
  }else {raw_low=raw_quarter;square_low=square_quarter;}
 }
}

template<unsigned ROWS>
static void rms_dataflow(bus_t *a,bus_t *c,const Job &j,const float gamma[128],unsigned rows=ROWS) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
 c204::short_norm_raw_words raw;
 c204::short_norm_words squares,pairs8,pairs4,pairs2,lanes;
 c204::short_norm_scalars inverse;
#pragma HLS STREAM variable=raw depth=64
#pragma HLS STREAM variable=squares depth=4
#pragma HLS STREAM variable=pairs8 depth=4
#pragma HLS STREAM variable=pairs4 depth=4
#pragma HLS STREAM variable=pairs2 depth=4
#pragma HLS STREAM variable=lanes depth=4
#pragma HLS STREAM variable=inverse depth=4
 rms_read_square<ROWS>(a,j.a,raw,squares,rows);
 c204::short_norm_phase_pairs<16,2>(rows,squares,pairs8);
 c204::short_norm_phase_pairs<8,1>(rows,pairs8,pairs4);
 c204::short_norm_phase_pairs<4,1>(rows,pairs4,pairs2);
 c204::short_norm_phase_pairs<2,1>(rows,pairs2,lanes);
 c204::short_norm_inverse(rows,1e-6f,lanes,inverse);
 rms_store<ROWS>(c,j.c,gamma,raw,inverse,rows);
}

template<unsigned ROWS>
static void rms_rows(bus_t *a,bus_t *c,bus_t *x,const Job &source,unsigned &tiles,unsigned rows=ROWS) {
#pragma HLS INLINE off
 Job j=addresses(source);j.op=RMS_NORM;j.m=rows;j.n=j.la=j.lc=128;
 j.fa=j.fc=F16;j.fs0=F32;j.epsilon=1e-6f;
 float gamma[128];
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
 for(unsigned word=0;word<16;word++) {
#pragma HLS PIPELINE II=1
  bus_t packed=x[(j.s0>>5)+word];
  for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
   gamma[word*8+lane]=uf(packed.range(32*lane+31,32*lane));
  }
 }
 rms_dataflow<ROWS>(a,c,j,gamma,rows);tiles+=rows;
}

template<unsigned ROWS>
static void rope_rows(bus_t *a,bus_t *c,bus_t *x,const Job &j,
 c209mp::activation row[MAX_LOCAL_COLS],unsigned &tiles,unsigned rows=ROWS) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 for(unsigned r=0;r<rows;r++) {
  load_activation_row(a,j.a+2ull*r*LOCAL_WIDTH,LOCAL_WIDTH,false,row);
  ap_uint<128> low_half=0;
  for(unsigned head=0;head<LOCAL_HEADS;head++)for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
   bus_t cosine=x[(j.s0>>5)+(uint64_t(r)*128+d)/8];
   bus_t sine=x[(j.s1>>5)+(uint64_t(r)*128+d)/8];
   ap_uint<128> half=0;
   for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
    unsigned pos=d+lane;
    float value=c209mp::widen(row[head*128+pos]);
    float rotated=pos<64?-c209mp::widen(row[head*128+pos+64]):c209mp::widen(row[head*128+pos-64]);
    float cv=uf(cosine.range(32*lane+31,32*lane)),sv=uf(sine.range(32*lane+31,32*lane));
    float rounded=c209mp::round(value*cv+rotated*sv);
    half.range(16*lane+15,16*lane)=c209mp::bits(c209mp::narrow(rounded));
   }
   if(!(d&8u))low_half=half;
   else {
    bus_t packed;packed.range(127,0)=low_half;packed.range(255,128)=half;
    c[(j.c>>5)+(uint64_t(r)*LOCAL_WIDTH+head*128+d)/16]=packed;
   }
  }
  tiles++;
 }
}

template<unsigned ROWS>
static void gelu_store(bus_t *c,uint64_t address,c2fast::pair_output_stream &results,unsigned rows=ROWS) {
#pragma HLS INLINE off
 ap_uint<32> pairs[8];
#pragma HLS ARRAY_PARTITION variable=pairs complete
 C209_FIXED_GELU_STORE_PAIRS:for(unsigned pair=0;pair<rows*(LOCAL_FFN/2);pair++) {
#pragma HLS PIPELINE II=1
  c2fast::pair_word value=results.read();ap_uint<32> normalized;
  normalized.range(15,0)=stored_half(c209mp::word(value.range(15,0)));
  normalized.range(31,16)=stored_half(c209mp::word(value.range(31,16)));
  unsigned slot=pair&7u;pairs[slot]=normalized;
  if(slot==7) {
   bus_t packed;
   for(unsigned i=0;i<8;i++) {
#pragma HLS UNROLL
    packed.range(32*i+31,32*i)=i==7?normalized:pairs[i];
   }
   c[(address>>5)+pair/8]=packed;
  }
 }
}

template<unsigned ROWS>
static void gelu_read(bus_t *a,uint64_t address,c2fast::nonlinear_pair_stream &input,unsigned rows=ROWS) {
#pragma HLS INLINE off
 bus_t packed=0;
 C209_FIXED_GELU_READ2:for(unsigned pair=0;pair<rows*(LOCAL_FFN/2);pair++) {
#pragma HLS PIPELINE II=1
  if(!(pair&7u))packed=a[(address>>5)+(pair>>3)];
  ap_uint<64> quarter=quarter_word(packed,(pair>>1)&3u);
  ap_uint<32> selected=(pair&1u)?ap_uint<32>(quarter.range(63,32)):ap_uint<32>(quarter.range(31,0));
  c2fast::NonlinearPair value;
  value.values.range(15,0)=c209mp::bits(input_half(ap_uint<16>(selected.range(15,0))));
  value.values.range(31,16)=c209mp::bits(input_half(ap_uint<16>(selected.range(31,16))));
  value.valid=3;input.write(value);
 }
}

template<unsigned ROWS>
static void gelu_exchange(bus_t *a,bus_t *c,const Job &j,
 c2fast::gelu_command_stream &commands,c2fast::nonlinear_pair_stream &input,
 c2fast::pair_output_stream &results,unsigned rows=ROWS) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#ifdef __SYNTHESIS__
 c2fast::send_registered_vector_gelu_command(commands,rows,LOCAL_FFN,0);
#endif
 gelu_read<ROWS>(a,j.a,input,rows);
#ifndef __SYNTHESIS__
 c2fast::shared_gelu_pair_math(input,results,rows*(LOCAL_FFN/2));
#endif
 gelu_store<ROWS>(c,j.c,results,rows);
}

template<unsigned ROWS>
static void gelu_rows(bus_t *a,bus_t *c,const Job &source,unsigned &tiles,
 c2fast::gelu_command_stream &commands,c2fast::nonlinear_pair_stream &input,
 c2fast::pair_output_stream &results,c2fast::gelu_ack_stream &ack,unsigned rows=ROWS) {
#pragma HLS INLINE off
 Job j=addresses(source);j.op=VECTOR;j.mode=GELU_TANH;
 j.m=rows;j.n=j.la=j.lc=LOCAL_FFN;j.fa=j.fc=F16;
 gelu_exchange<ROWS>(a,c,j,commands,input,results,rows);
 c2fast::end_shared_gelu(ack);tiles+=rows*(LOCAL_FFN/8);
}

template<unsigned ROWS>
static unsigned prepare_self_rows(bus_t *a,bus_t *b,bus_t *c,bus_t *x,
 const Job &source,unsigned &tiles,unsigned rows=ROWS) {
#pragma HLS INLINE off
 static_assert(ROWS>0&&ROWS<=4096&&ROWS%128==0,"Self KV contains complete 128-key blocks");
 const unsigned BLOCKS=rows/128;
 // Stage13 writes the local token shard directly into the global packed
 // workspace. Quantization still uses each original128-key local block.
 const bool global=source.op==13;
 const unsigned token_offset=global?source.k0:0;
 const unsigned stored_blocks=global?source.n/128:BLOCKS;
 c209mp::activation row[128],vf[128][128];ap_uint<31> vmax[128];
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vf cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=vf cyclic factor=8 dim=2
#pragma HLS BIND_STORAGE variable=vf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=vmax cyclic factor=8
 tiles=0;
 for(unsigned head=0;head<LOCAL_HEADS;head++)for(unsigned block=0;block<BLOCKS;block++) {
  for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
   for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
    vmax[d+lane]=0;
   }
  }
  for(unsigned n=0;n<128;n++) {
   ap_uint<31> maxima[8];
#pragma HLS ARRAY_PARTITION variable=maxima complete
   for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
    maxima[lane]=0;
   }
   bus_t pk=0,pv=0;
   C209_FIXED_PREPARE_LOAD8:for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
    if(!(d&8u)) {
     uint64_t offset=(uint64_t(block*128+n)*LOCAL_WIDTH+head*128+d)>>4;
     pk=a[(source.a>>5)+offset];pv=b[(source.b>>5)+offset];
    }
    ap_uint<128> kh=half_word(pk,bool(d&8u)),vh=half_word(pv,bool(d&8u));
    for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
     c209mp::activation raw_k=input_half(ap_uint<16>(kh.range(16*lane+15,16*lane)));
     c209mp::activation raw_v=input_half(ap_uint<16>(vh.range(16*lane+15,16*lane)));
     row[d+lane]=raw_k;vf[n][d+lane]=raw_v;
     float k=c209mp::widen(raw_k),v=c209mp::widen(raw_v);
     ap_uint<31> ak=c204::attention_absbits(k),av=c204::attention_absbits(v);
     maxima[lane]=ak>maxima[lane]?ak:maxima[lane];vmax[d+lane]=av>vmax[d+lane]?av:vmax[d+lane];
    }
   }
   float mx=c204::attention_absmax8(maxima),km=mx==0?1.f:mx/32760.f;
   writef(x,source.s0,(token_offset+block*128+n)*KS_STRIDE+head,km);float ki=1.f/km;
   bus_t packed_k=0;
   C209_FIXED_PREPARE_K_QUANT4:for(unsigned d=0;d<128;d+=4) {
#pragma HLS PIPELINE II=1
    ap_uint<64> group=0;
    for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
     group.range(16*lane+15,16*lane)=ap_uint<16>(quant(c209mp::widen(row[d+lane]),ki));
    }
    packed_k>>=64;packed_k.range(255,192)=group;
    if((d&15u)==12)c[(source.c>>5)+(uint64_t(token_offset+block*128+n)*LOCAL_WIDTH+head*128+d)/16]=packed_k;
   }
  }
  float vscale[128],vinverse[128];
#pragma HLS ARRAY_PARTITION variable=vscale cyclic factor=8
#pragma HLS BIND_STORAGE variable=vscale type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=vinverse type=ram_2p impl=lutram
  C209_FIXED_PREPARE_V_SCALE:for(unsigned d=0;d<128;d++) {
#pragma HLS PIPELINE off
   float mx=uf((unsigned)vmax[d]),vm=mx==0?1.f:mx/32760.f,vi=1.f/vm;
   vscale[d]=vm;vinverse[d]=vi;
  }
  C209_FIXED_WRITE_V_SCALES:for(unsigned word=0;word<16;word++) {
#pragma HLS PIPELINE II=1
   bus_t packed=0;
   for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
    packed.range(32*lane+31,32*lane)=fu(vscale[word*8+lane]);
   }
   x[(source.s1>>5)+uint64_t(head*stored_blocks+token_offset/128+block)*16+word]=packed;
  }
  bus_t packed_v=0;
  C209_FIXED_PREPARE_V_QUANT4:for(unsigned group_id=0;group_id<c207::VW*4;group_id++) {
#pragma HLS PIPELINE II=1
   unsigned d=group_id/32,n=(group_id&31u)*4;float vi=vinverse[d];ap_uint<64> group=0;
   for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
    group.range(16*lane+15,16*lane)=ap_uint<16>(quant(c209mp::widen(vf[n+lane][d]),vi));
   }
   packed_v>>=64;packed_v.range(255,192)=group;
   if((group_id&3u)==3)x[(source.s2>>5)+uint64_t(head*stored_blocks+token_offset/128+block)*c207::VW+group_id/4]=packed_v;
  }
  tiles++;
 }
 return OK;
}
} // namespace auxiliary_detail

static unsigned auxiliary(bus_t *a,bus_t *b,bus_t *c,bus_t *x,const Job &source,
 Kind kind,unsigned &tiles,c2fast::gelu_command_stream &commands,c2fast::nonlinear_pair_stream &input,
 c2fast::pair_output_stream &results,c2fast::gelu_ack_stream &ack) {
#pragma HLS INLINE off
 // The parent has performed profile/range admission once. Its finite Kind
 // selects only canonical dimensions/formats; no generic dispatcher is used.
 c209mp::activation row[MAX_LOCAL_COLS];
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS BIND_STORAGE variable=row type=ram_2p impl=bram
 // The validated descriptor picks a finite count. All auxiliary kinds share
 // one MAX_LOCAL_ROWS arithmetic body rather than duplicate2048/4096 engines.
 const unsigned rows=canonical_local_rows(kind==RMS128?source.m/LOCAL_HEADS:source.m);
 Job j=auxiliary_detail::addresses(source);
 switch(kind) {
  case QUANT_PATCH:j.op=QUANTIZE;j.m=rows;j.n=68;j.la=j.lc=80;j.fa=F32;j.fc=I16;j.fs0=F32;break;
  case QUANT_PROJECTION:j.op=QUANTIZE;j.m=rows;j.n=j.la=j.lc=LOCAL_WIDTH;j.fa=F16;j.fc=I16;j.fs0=F32;break;
  case QUANT_FFN:j.op=QUANTIZE;j.m=rows;j.n=j.la=j.lc=LOCAL_FFN;j.fa=F16;j.fc=I16;j.fs0=F32;break;
  case NORM_ADALN:j.op=NORM_ADALN_QUANTIZE;j.m=rows;j.n=j.la=j.lc=HIDDEN;j.fa=F32;j.fc=I16;j.fs0=j.fs1=j.fs2=j.fs3=F32;j.epsilon=1e-6f;break;
  case RMS128:auxiliary_detail::rms_rows<c209tp4::MAX_LOCAL_ROWS*LOCAL_HEADS>(a,c,x,source,tiles,rows*LOCAL_HEADS);return OK;
  case ROTARY:auxiliary_detail::rope_rows<c209tp4::MAX_LOCAL_ROWS>(a,c,x,source,row,tiles,rows);return OK;
  case FFN_GELU:auxiliary_detail::gelu_rows<c209tp4::MAX_LOCAL_ROWS>(a,c,source,tiles,commands,input,results,ack,rows);return OK;
  case SELF_PREPARE:return auxiliary_detail::prepare_self_rows<c209tp4::MAX_LOCAL_ROWS>(a,b,c,x,source,tiles,rows);
  default:return BAD_STAGE;
 }
 auxiliary_detail::long_rows<c209tp4::MAX_LOCAL_ROWS>(a,c,x,j,row,tiles,rows);return OK;
}
} // namespace c209fixed
