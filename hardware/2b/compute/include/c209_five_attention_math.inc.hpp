#pragma once
// Shared fixed16 softmax pipeline; outer transactions and FP32 PV remain.
#include "c209_five_softmax_pipeline.inc.hpp"

template<class Sink> static void update_five_to(Sink &sink,unsigned count,
 C209UpdateStream &results,MetaStream &metadata) {
#pragma HLS INLINE off
 using namespace c2;
 // Eight local memory banks. Within one block the half-beat index is
 // strictly increasing, so every FP32 state cell is touched exactly once.
 // A later key block may revisit it only after this finite loop drains.
 float u[ATTENTION_CONTEXTS][QR*16][8];
 float alpha[QR],den[QR],inverse[QR],scale[128];
#pragma HLS ARRAY_PARTITION variable=u complete dim=3
#pragma HLS BIND_STORAGE variable=u type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=scale cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=alpha cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=den cyclic factor=8
#pragma HLS BIND_STORAGE variable=alpha type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=den type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=inverse type=ram_2p impl=lutram
 PV_UPDATE_BLOCKS:for(unsigned block=0;block<count;block++) {
  Meta m=metadata.read();
  ap_uint<768> packed_header=0;ap_uint<2> header_part=0;
  PV_UPDATE_HEADER_EIGHT:for(unsigned chunk=0;chunk<(2*QR+128)/8;chunk++) {
#pragma HLS PIPELINE II=1
    if(header_part==0)packed_header=results.read();
    ap_uint<256> fields=0;
    switch(header_part) {
     case 0:fields=packed_header.range(255,0);break;
     case 1:fields=packed_header.range(511,256);break;
     default:fields=packed_header.range(767,512);break;
    }
    for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
     unsigned index=chunk*8+lane;
     float value=uf(unsigned(fields.range(lane*32+31,lane*32)));
     if(index<QR)alpha[index]=value;
     else if(index<2*QR)den[index-QR]=value;
     else if(index<2*QR+128)scale[index-2*QR]=value;
    }
    header_part=header_part==2?ap_uint<2>(0):ap_uint<2>(header_part+1);
  }
  // One inverse per valid row, before the eight-lane pipeline. Non-final
  // blocks need no normalization; den==0 keeps the exact old zero result.
  if(m.last) {
   PV_UPDATE_ROW_INVERSES:for(unsigned row=0;row<m.mm;row++) {
#pragma HLS PIPELINE II=1
    inverse[row]=den[row]==0?0.f:1.f/den[row];
   }
  }
  ap_uint<384> pending_high=0;
  const unsigned half_beats=m.mm*16u;
  PV_UPDATE_HALF_BEATS_II1:for(unsigned beat=0;beat<half_beats;beat++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=512
   const unsigned row=beat>>4,half=beat&1u,word=(beat&15u)>>1;
   const unsigned first_dimension=(beat&15u)*8u;
   ap_uint<384> dots;
   if(half==0) {
    ap_uint<768> dotword=results.read();
    dots=dotword.range(383,0);
    pending_high=dotword.range(767,384);
   } else dots=pending_high;
   float al=alpha[row],inv=m.last?inverse[row]:0.f;
   ap_uint<128> half_pack=0;
   bus_t float_pack=0;
   PV_UPDATE_EIGHT_LANES:for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
    unsigned dimension=first_dimension+lane;
    i48 dot=i48(dots.range(lane*48+47,lane*48));
    float previous=m.first?0.f:u[m.context][beat][lane];
    // Scale is folded once at its producer, not recomputed for every PV
    // block/header. The eight-word header and complete data drain stay fixed.
    float next=al*previous+(float)dot*scale[dimension];
    u[m.context][beat][lane]=next;
    float rounded=m.last?c209mp::round(next*inv):0.f;
    half_pack.range(lane*16+15,lane*16)=c209mp::bits(c209mp::narrow(rounded));
    float_pack.range(lane*32+31,lane*32)=fu(rounded);
   }
   // Exactly two adjacent half-beats form each output word. The sink's
   // existing low-half register and complete-word F16 stores are retained.
   if(m.last)sink.write(m,row,word,half,half_pack,float_pack);
  }
 }
}

