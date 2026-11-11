#pragma once
#include "c209_aux_quant_service.hpp"
#include "c209_five_fixed_job.hpp"
namespace c209five {
using namespace c2;
using namespace c209fixed::auxiliary_detail;
using namespace c209tp4;
// Full128x128 INT16 V block, with bank=(row+column)&7 and
// address=row*16+column/8. Eight adjacent columns and four adjacent rows
// each touch distinct banks. Loading and transposed reading are separate
// phases, so each physical bank needs only one access per cycle.
using PrepareVBanked=c209mp::activation[8][2048];
static void prepare_v_requests(const PrepareVBanked &vf,const float inverse[128],
 AuxQuantRequests &requests) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=vf complete dim=1
 for(ap_uint<13> group=0;group<c207::VW*4;group++) {
#pragma HLS PIPELINE II=1
  ap_uint<7> d=group>>5,n=(group&31u)*4;ap_uint<64> halves=0;
  c209mp::word bank_values[8];
#pragma HLS ARRAY_PARTITION variable=bank_values complete
  // Static bank accesses prevent HLS from constructing four read ports on
  // every bank. Only the four requested banks read; the rest output zero.
  for(unsigned bank=0;bank<8;bank++) {
#pragma HLS UNROLL
   ap_uint<3> lane=ap_uint<3>(bank)-ap_uint<3>(d)-ap_uint<3>(n);
   bank_values[bank]=0;
   if(lane<4) {
    ap_uint<7> source_row=n+lane;
    ap_uint<11> address=ap_uint<11>(source_row)*16+(d>>3);
    bank_values[bank]=c209mp::bits(vf[bank][address]);
   }
  }
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   ap_uint<3> bank=ap_uint<3>(n)+ap_uint<3>(d)+lane;
   halves.range(16*lane+15,16*lane)=bank_values[bank];
  }
  requests.write(aux_quant_request(halves,inverse[d]));
 }
}
static void prepare_v_quant(const PrepareVBanked &vf,const float inverse[128],
 bus_t *output,uint64_t address,AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(prepare_v_requests(vf,inverse,requests));
 C209_FIXED_TASK(aux_quant_row_store(output,address,16384,16384,results));
}
static unsigned prepare_self_quant_service(bus_t *a,bus_t *b,bus_t *c,bus_t *x,
 const FixedJob &source,unsigned &tiles,unsigned rows,AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
 const unsigned BLOCKS=rows/128;
 // Only fixed stage13 is reachable: ten local heads into the complete
 // global4096-key workspace, with the original128-key quantization blocks.
 const unsigned token_offset=source.k0;
 const unsigned stored_blocks=32;
 c209mp::activation row[128];PrepareVBanked vf;ap_uint<31> vmax[128];
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vf complete dim=1
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
     WordOffset offset=(WordOffset(block*128+n)*LOCAL_WIDTH+head*128u+d)>>4;
     pk=a[word_address(source.a,offset)];pv=b[word_address(source.b,offset)];
    }
    ap_uint<128> kh=half_word(pk,bool(d&8u)),vh=half_word(pv,bool(d&8u));
    c209mp::activation v_lanes[8];
#pragma HLS ARRAY_PARTITION variable=v_lanes complete
    for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
     c209mp::activation raw_k=input_half(ap_uint<16>(kh.range(16*lane+15,16*lane)));
     c209mp::activation raw_v=input_half(ap_uint<16>(vh.range(16*lane+15,16*lane)));
     row[d+lane]=raw_k;v_lanes[lane]=raw_v;
     float k=c209mp::widen(raw_k),v=c209mp::widen(raw_v);
     ap_uint<31> ak=c204::attention_absbits(k),av=c204::attention_absbits(v);
     maxima[lane]=ak>maxima[lane]?ak:maxima[lane];vmax[d+lane]=av>vmax[d+lane]?av:vmax[d+lane];
    }
    const ap_uint<11> v_address=n*16+(d>>3);
    for(unsigned bank=0;bank<8;bank++) {
#pragma HLS UNROLL
     // d is an eight-column boundary; rotate values locally so every
     // statically named bank receives exactly one write.
     ap_uint<3> lane=ap_uint<3>(bank)-ap_uint<3>(n);
     vf[bank][v_address]=v_lanes[lane];
    }
   }
   float mx=c204::attention_absmax8(maxima),km=mx==0?1.f:mx/32760.f;
   writef(x,source.s0,(token_offset+block*128+n)*c209fixed::KS_STRIDE+head,km);float ki=1.f/km;
   // SAMEHEAD_K_HEAD_TOKEN_V1: each head owns a continuous4096-key region.
   // Write directly into the final global layout; no ARM transpose/copy.
   ByteOffset key_bytes=(ByteOffset(head)*4096u+token_offset+block*128+n)*256u;
   aux_quant_row(row,c,byte_address(source.c,key_bytes),
     128,128,ki,requests,results);
  }
  float vscale[128],vinverse[128];
#pragma HLS ARRAY_PARTITION variable=vscale cyclic factor=8
#pragma HLS BIND_STORAGE variable=vscale type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=vinverse type=ram_2p impl=lutram
  C209_FIXED_PREPARE_V_SCALE:for(unsigned d=0;d<128;d++) {
#pragma HLS PIPELINE off
   float mx=uf((unsigned)vmax[d]),vm=mx==0?1.f:mx/32760.f,vi=1.f/vm;
   // PV_PROB_SCALE_FOLDED_V1: V quantization still uses original vm. Only
   // the exported FP32 reconstruction scale includes probability /32760.
   // Cross-attention cache scale records must carry the same new semantics.
   vscale[d]=vm*(1.f/32760.f);vinverse[d]=vi;
  }
  C209_FIXED_WRITE_V_SCALES:for(unsigned word=0;word<16;word++) {
#pragma HLS PIPELINE II=1
   bus_t packed=0;
   for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
    packed.range(32*lane+31,32*lane)=fu(vscale[word*8+lane]);
   }
   WordOffset offset=WordOffset(head*stored_blocks+token_offset/128+block)*16u+word;
   x[word_address(source.s1,offset)]=packed;
  }
  ByteOffset value_bytes=ByteOffset(head*stored_blocks+token_offset/128+block)*c207::VW*32u;
  prepare_v_quant(vf,vinverse,x,byte_address(source.s2,value_bytes),
    requests,results);
  tiles++;
 }
 return OK;
}
} // namespace c209five
