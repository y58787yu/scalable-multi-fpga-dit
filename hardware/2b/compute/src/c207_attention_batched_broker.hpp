#pragma once
#if C207_ATTENTION_DIRECT_BANKS
#include "c207_attention_direct_io.hpp"
#else
// Included inside namespace c207 after shared channel types. One Q64/PV64
// command covers the same pair of Q32 contexts; score/prob/update PIPOs and
// each row's arithmetic remain unchanged. Only this task owns physical MAC.
static void mac_broker(const c2::Job&j,unsigned stored,KVBlocks&operands,MetaStream&kin,
 ScoreBlocks&scores,MetaStream&sout,ProbBlocks&probability,MetaStream&pin,
 UpdateBlocks&updates,MetaStream&uout,c2::i16 at[c2::BM][c2::BK],
 c2::i16 bt[c2::BN][c2::BK],c2::i48 ct[c2::BM][c2::BN],
 cmd_stream&cmd,input_stream&ai,input_stream&bi,result_stream&co,unsigned&tiles) {
#pragma HLS INLINE off
 using namespace c2;
 static_assert(2*QR<=BM,"Two query contexts must fit the existing MAC scratch");
 // One-keyblock lookahead only within the current head/query pair. The two
 // retained V/metadata slots survive QK(k+1), while all Q/P scratch and ct
 // remain shared. The trailing tick drains PV before reusing om/context for
 // another query pair or head.
 bus_t retained_v[2][VW],scale_words[2][(QR+128)/8],v_scale_words[2][128/8],prob_scales[2][2*QR/8];
 Meta issued[2][2];unsigned pair_rows[2];
#pragma HLS BIND_STORAGE variable=retained_v type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=scale_words type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=prob_scales type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=v_scale_words type=ram_2p impl=bram
 tiles=0;
 for(unsigned h=0;h<j.heads;h++)for(unsigned base=0;base<j.m;base+=2*QR) {
  unsigned contexts=minv(2,(j.m-base+QR-1)/QR);
  unsigned keyblocks=blocks(stored);
  for(unsigned tick=0;tick<keyblocks+1;tick++) {
   if(tick<keyblocks) {
   unsigned slot=tick&1u;
   unsigned total_rows=0;
   // Aggregate both Q operands, but decode/retain shared K/V only from ctx0.
   for(unsigned ctx=0;ctx<contexts;ctx++) {
    Meta m=kin.read();issued[slot][ctx]=m;total_rows+=m.mm;
    hls::read_lock<KVBlock> block(operands);
    for(unsigned r=0;r<m.mm;r++)for(unsigned d=0;d<128;d+=16) {
#pragma HLS PIPELINE II=1
     bus_t p=block[r*8+d/16];
     for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
      at[ctx*QR+r][d+l]=i16(p.range(16*l+15,16*l));
     }
    }
    for(unsigned w=0;w<(QR+128)/8;w++) {
#pragma HLS PIPELINE II=1
     scale_words[ctx][w]=block[KV_SCALE+w];
    }
    if(ctx==0) {
     unsigned padded=((m.nn+PO-1)/PO)*PO;
     for(unsigned n=0;n<padded;n++)for(unsigned d=0;d<128;d+=16) {
#pragma HLS PIPELINE II=1
      bus_t p=block[QW+n*8+d/16];
      for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
       bt[n][d+l]=i16(p.range(16*l+15,16*l));
      }
     }
     for(unsigned i=0;i<VW;i++) {
#pragma HLS PIPELINE II=1
      retained_v[slot][i]=block[QW+KW+i];
     }
     for(unsigned w=0;w<128/8;w++)v_scale_words[slot][w]=block[KV_SCALE+(QR+128)/8+w];
    }
   }
   pair_rows[slot]=total_rows;
   unsigned padded=((issued[slot][0].nn+PO-1)/PO)*PO;
   mac_tile(at,bt,ct,total_rows,padded,128,true,true,cmd,ai,bi,co);tiles++;
   // The physical row result is split back into the original two Q32 contexts.
   for(unsigned ctx=0;ctx<contexts;ctx++) {
    Meta m=issued[slot][ctx];
    {
     hls::write_lock<ScoreBlock> output(scores);
     for(unsigned r=0;r<m.mm;r++)for(unsigned n=0;n<128;n+=16) {
#pragma HLS PIPELINE II=1
      ap_uint<768> p=0;
      for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
       p.range(48*l+47,48*l)=n+l<m.nn?ap_uint<48>(ct[ctx*QR+r][n+l]):ap_uint<48>(0);
      }
      output[r*8+n/16]=p;
     }
     for(unsigned w=0;w<(QR+128+23)/24;w++)output[DOT_WORDS+w]=0;
     for(unsigned i=0;i<QR+128;i++)put_float(output[DOT_WORDS+i/24],i%24,get_float(scale_words[ctx][i/8],i%8));
    }
    sout.write(m);
   }
   } // Both QK score blocks and metadata are committed before shared ct reuse.
   // Bootstrap QK0; each subsequent tick issues QK(k+1) above before PV(k).
   // Depth2 score/prob blocks admit precisely this one-block window: if P(k)
   // fills its two slots, softmax can hold Score0(k+1) and the other score slot
   // admits Score1(k+1). We then consume P(k), never request QK(k+2) first.
   if(tick>0) {
   unsigned slot=(tick-1)&1u,total_rows=pair_rows[slot];
   for(unsigned ctx=0;ctx<contexts;ctx++) {
    Meta m=pin.read();
    hls::read_lock<ProbBlock> block(probability);
    unsigned inner=(m.nn+15)&~15u;
    for(unsigned r=0;r<m.mm;r++)for(unsigned n=0;n<inner;n+=16) {
#pragma HLS PIPELINE II=1
     bus_t p=block[r*8+n/16];
     for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
      at[ctx*QR+r][n+l]=i16(p.range(16*l+15,16*l));
     }
    }
    for(unsigned w=0;w<2*QR/8;w++)prob_scales[ctx][w]=block[PROB_SCALE+w];
   }
   unsigned inner=(issued[slot][0].nn+15)&~15u;
   for(unsigned d=0;d<128;d++)for(unsigned n=0;n<inner;n+=16) {
#pragma HLS PIPELINE II=1
    bus_t p=retained_v[slot][d*8+n/16];
    for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
     bt[d][n+l]=i16(p.range(16*l+15,16*l));
    }
   }
   mac_tile(at,bt,ct,total_rows,128,inner,true,true,cmd,ai,bi,co);tiles++;
   for(unsigned ctx=0;ctx<contexts;ctx++) {
    Meta m=issued[slot][ctx];
    {
     hls::write_lock<UpdateBlock> output(updates);
     for(unsigned r=0;r<m.mm;r++)for(unsigned d=0;d<128;d+=16) {
#pragma HLS PIPELINE II=1
      ap_uint<768> p=0;
      for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
       p.range(48*l+47,48*l)=ap_uint<48>(ct[ctx*QR+r][d+l]);
      }
      output[r*8+d/16]=p;
     }
     for(unsigned w=0;w<(2*QR+128+23)/24;w++)output[DOT_WORDS+w]=0;
     for(unsigned i=0;i<2*QR;i++)put_float(output[DOT_WORDS+i/24],i%24,get_float(prob_scales[ctx][i/8],i%8));
     for(unsigned d=0;d<128;d++)put_float(output[DOT_WORDS+(2*QR+d)/24],(2*QR+d)%24,get_float(v_scale_words[slot][d/8],d%8));
    }
    uout.write(m);
   }
   } // Final tick has no QK; it drains the last PV and closes the window.
  }
 }
}
#endif
