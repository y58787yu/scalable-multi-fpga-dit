#pragma once
#include "c209_five_softmax_fixed16.inc.hpp"

// One finite SM8 score/exp-denominator/pack pipeline. Each128-bit FIFO has128 words:
// exactly two complete four-row groups, the SAME bit capacity as SM4. A group
// descriptor is published after its64 data words; the final group drains alone.
// These bounded scratch FIFOs replace the old score/probability scratch arrays;
// the outer score2/probability5 transaction slots remain owned by the caller.
using SoftmaxOctets=hls::stream<ap_uint<128>>;
using SoftmaxGroupMeta=hls::stream<ap_uint<256>>;
using SoftmaxRowState=ap_uint<48>; // max half16, denominator unsignedQ13.19.

// Metadata is a packet only at the FIFO boundary. Fixed slices below are
// wires; selecting a row afterwards is a four-way16/32-bit mux. Never use
// variable range() on the256-bit packet inside an arithmetic pipeline.
static void softmax_unpack_descriptor(ap_uint<256> packet,
 softmax16::Half maximum[4],ap_uint<16> alpha[4],softmax16::Den denominator[4]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=maximum complete
#pragma HLS ARRAY_PARTITION variable=alpha complete
#pragma HLS ARRAY_PARTITION variable=denominator complete
 maximum[0]=packet.range(15,0);maximum[1]=packet.range(31,16);
 maximum[2]=packet.range(47,32);maximum[3]=packet.range(63,48);
 alpha[0]=packet.range(79,64);alpha[1]=packet.range(95,80);
 alpha[2]=packet.range(111,96);alpha[3]=packet.range(127,112);
 denominator[0]=packet.range(159,128);denominator[1]=packet.range(191,160);
 denominator[2]=packet.range(223,192);denominator[3]=packet.range(255,224);
}
static ap_uint<256> softmax_pack_descriptor(const softmax16::Half maximum[4],
 const ap_uint<16> alpha[4],const softmax16::Den denominator[4]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=maximum complete
#pragma HLS ARRAY_PARTITION variable=alpha complete
#pragma HLS ARRAY_PARTITION variable=denominator complete
 ap_uint<256> packet=0;
 packet.range(15,0)=maximum[0];packet.range(31,16)=maximum[1];
 packet.range(47,32)=maximum[2];packet.range(63,48)=maximum[3];
 packet.range(79,64)=alpha[0];packet.range(95,80)=alpha[1];
 packet.range(111,96)=alpha[2];packet.range(127,112)=alpha[3];
 packet.range(159,128)=denominator[0];packet.range(191,160)=denominator[1];
 packet.range(223,192)=denominator[2];packet.range(255,224)=denominator[3];
 return packet;
}

static void softmax_score_groups(const FiveScoreBlock &input,const Meta &m,
 const SoftmaxRowState previous[QR],SoftmaxOctets &scores,SoftmaxGroupMeta &descriptors) {
#pragma HLS INLINE off
 softmax16::Half maxima[8],group_max[4];ap_uint<16> group_alpha[4];ap_uint<32> group_den[4];
#pragma HLS ARRAY_PARTITION variable=maxima complete
#pragma HLS ARRAY_PARTITION variable=group_max complete
#pragma HLS ARRAY_PARTITION variable=group_alpha complete
#pragma HLS ARRAY_PARTITION variable=group_den complete
 const unsigned groups=(m.mm+3u)>>2;bus_t packed=0;SoftmaxRowState old=0;
 SOFTMAX16_SCORE_OCTETS:for(unsigned index=0;index<groups*64u;index++) {
#pragma HLS PIPELINE II=1
  const unsigned row=index>>4,octet=index&15u,context=row&3u;
  const bool row_valid=row<m.mm;
  if(octet==0)old=row_valid?previous[row]:SoftmaxRowState(0);
  if(!(octet&1u))packed=row_valid?input[row*8u+(octet>>1)]:bus_t(0);
  ap_uint<128> encoded=octet&1u?ap_uint<128>(packed.range(255,128)):ap_uint<128>(packed.range(127,0));
  softmax16::Half current[8];
#pragma HLS ARRAY_PARTITION variable=current complete
  for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
   softmax16::Half score=encoded.range(lane*16u+15,lane*16u);
   if(!row_valid||octet*8u+lane>=m.nn)score=0xfc00u;
   encoded.range(lane*16u+15,lane*16u)=score;
   softmax16::Half preceding=octet==0?softmax16::Half(old.range(15,0)):maxima[lane];
   current[lane]=softmax16::maximum(preceding,score);maxima[lane]=current[lane];
  }
  scores.write(encoded);
  if(octet==15) {
   softmax16::Half lower=softmax16::maximum(softmax16::maximum(current[0],current[1]),softmax16::maximum(current[2],current[3]));
   softmax16::Half upper=softmax16::maximum(softmax16::maximum(current[4],current[5]),softmax16::maximum(current[6],current[7]));
   softmax16::Half maximum=softmax16::maximum(lower,upper);
   ap_uint<32> denominator=old.range(47,16);
   group_max[context]=maximum;
   group_alpha[context]=denominator==0?ap_uint<16>(0):softmax16::exponential<8>(maximum,softmax16::Half(old.range(15,0)));
   group_den[context]=denominator;
   if(context==3)descriptors.write(softmax_pack_descriptor(group_max,group_alpha,group_den));
  }
 }
}

using SoftmaxExpMeta=hls::stream<ap_uint<384>>;
static void softmax_exp_values(unsigned groups,unsigned valid_keys,
 SoftmaxOctets &scores,SoftmaxGroupMeta &input_meta,
 SoftmaxOctets &probability,SoftmaxExpMeta &summaries) {
#pragma HLS INLINE off
 for(unsigned group=0;group<groups;group++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
  ap_uint<256> descriptor=input_meta.read();
  softmax16::Half largest_by_row[4];
  ap_uint<16> alpha_by_row[4];
  softmax16::Den previous_by_row[4];
#pragma HLS ARRAY_PARTITION variable=largest_by_row complete
#pragma HLS ARRAY_PARTITION variable=alpha_by_row complete
#pragma HLS ARRAY_PARTITION variable=previous_by_row complete
  softmax_unpack_descriptor(descriptor,largest_by_row,alpha_by_row,previous_by_row);
  ap_uint<23> sums[4];
#pragma HLS ARRAY_PARTITION variable=sums complete
  for(unsigned row=0;row<4;row++) {
#pragma HLS UNROLL
   sums[row]=0;
  }
  SOFTMAX16_EXP_OCTETS:for(unsigned index=0;index<64;index++) {
#pragma HLS PIPELINE II=1
   ap_uint<2> row=ap_uint<6>(index)>>4;
   unsigned key=(index&15u)*8u;
   softmax16::Half largest=largest_by_row[row];
   ap_uint<128> input=scores.read(),output=0;
   ap_uint<16> p[8];
#pragma HLS ARRAY_PARTITION variable=p complete
   p[0]=key<valid_keys?softmax16::exponential<0>(largest,softmax16::Half(input.range(15,0))):ap_uint<16>(0);
   p[1]=key+1<valid_keys?softmax16::exponential<1>(largest,softmax16::Half(input.range(31,16))):ap_uint<16>(0);
   p[2]=key+2<valid_keys?softmax16::exponential<2>(largest,softmax16::Half(input.range(47,32))):ap_uint<16>(0);
   p[3]=key+3<valid_keys?softmax16::exponential<3>(largest,softmax16::Half(input.range(63,48))):ap_uint<16>(0);
   p[4]=key+4<valid_keys?softmax16::exponential<4>(largest,softmax16::Half(input.range(79,64))):ap_uint<16>(0);
   p[5]=key+5<valid_keys?softmax16::exponential<5>(largest,softmax16::Half(input.range(95,80))):ap_uint<16>(0);
   p[6]=key+6<valid_keys?softmax16::exponential<6>(largest,softmax16::Half(input.range(111,96))):ap_uint<16>(0);
   p[7]=key+7<valid_keys?softmax16::exponential<7>(largest,softmax16::Half(input.range(127,112))):ap_uint<16>(0);
   ap_uint<19> total=ap_uint<19>(p[0])+p[1]+p[2]+p[3]+p[4]+p[5]+p[6]+p[7];
   sums[row]+=total;
   for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
    output.range(16u*lane+15,16u*lane)=softmax16::probability_word(p[lane]);
   }
   probability.write(output);
  }
  // The finite group drains before publishing its summary. A separate
  // actor updates denominators while the next group runs its exponentials;
  // no previous tail depends on the next input descriptor becoming valid.
  ap_uint<384> summary=0;summary.range(255,0)=descriptor;
  for(unsigned row=0;row<4;row++) {
#pragma HLS UNROLL
   summary.range(256+row*23+22,256+row*23)=sums[row];
  }
  summaries.write(summary);
 }
}
static void softmax_den_groups(unsigned groups,SoftmaxExpMeta &summaries,
 SoftmaxGroupMeta &output_meta) {
#pragma HLS INLINE off
 for(unsigned group=0;group<groups;group++) {
#pragma HLS LOOP_FLATTEN off
  ap_uint<384> summary=summaries.read();
  softmax16::Half maximum[4];ap_uint<16> alpha[4];
  softmax16::Den previous[4],next[4];ap_uint<23> sums[4];
#pragma HLS ARRAY_PARTITION variable=maximum complete
#pragma HLS ARRAY_PARTITION variable=alpha complete
#pragma HLS ARRAY_PARTITION variable=previous complete
#pragma HLS ARRAY_PARTITION variable=next complete
#pragma HLS ARRAY_PARTITION variable=sums complete
  softmax_unpack_descriptor(ap_uint<256>(summary.range(255,0)),maximum,alpha,previous);
  for(unsigned row=0;row<4;row++) {
#pragma HLS UNROLL
   sums[row]=summary.range(256+row*23+22,256+row*23);
  }
  SOFTMAX16_DENOMINATORS:for(unsigned row=0;row<4;row++) {
#pragma HLS PIPELINE II=1
   ap_uint<2> selected=row;
   next[selected]=softmax16::update_den(previous[selected],alpha[selected],sums[selected]);
  }
  output_meta.write(softmax_pack_descriptor(maximum,alpha,next));
 }
}
static void softmax_exp_groups(unsigned groups,unsigned valid_keys,
 SoftmaxOctets &scores,SoftmaxGroupMeta &input_meta,
 SoftmaxOctets &probability,SoftmaxGroupMeta &output_meta) {
#pragma HLS INLINE off
 SoftmaxExpMeta summaries;
#pragma HLS STREAM variable=summaries depth=2
#pragma HLS BIND_STORAGE variable=summaries type=fifo impl=lutram
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(softmax_exp_values(groups,valid_keys,scores,input_meta,probability,summaries));
 C209_FIXED_TASK(softmax_den_groups(groups,summaries,output_meta));
}

static void softmax_pack_groups(const Meta &m,SoftmaxOctets &probability,
 SoftmaxGroupMeta &descriptors,ProbBlock &output,SoftmaxRowState next[QR]) {
#pragma HLS INLINE off
 ap_uint<16> alpha[QR];
#pragma HLS BIND_STORAGE variable=alpha type=ram_2p impl=lutram
 const unsigned groups=(m.mm+3u)>>2;
 softmax16::Half maximum_by_row[4];
 ap_uint<16> alpha_by_row[4];
 softmax16::Den denominator_by_row[4];
#pragma HLS ARRAY_PARTITION variable=maximum_by_row complete
#pragma HLS ARRAY_PARTITION variable=alpha_by_row complete
#pragma HLS ARRAY_PARTITION variable=denominator_by_row complete
 bus_t packed=0;
 // The128-word probability FIFO holds two complete groups. The
 // exp actor can therefore finish a group's64 words and publish its meta
 // even if this actor is waiting at the next descriptor boundary.
 SOFTMAX16_PROBABILITY_PACK:for(unsigned index=0;index<groups*64u;index++) {
#pragma HLS PIPELINE II=1
   unsigned local_index=index&63u,row=index>>4;
   if(local_index==0) {
    ap_uint<256> descriptor=descriptors.read();
    softmax_unpack_descriptor(descriptor,maximum_by_row,alpha_by_row,denominator_by_row);
   }
   ap_uint<2> selected=ap_uint<6>(local_index)>>4;
   if((index&15u)==0&&row<m.mm) {
    SoftmaxRowState state=0;
    state.range(15,0)=maximum_by_row[selected];
    state.range(47,16)=denominator_by_row[selected];
    next[row]=state;
    alpha[row]=alpha_by_row[selected];
  }
   ap_uint<128> value=probability.read();
   packed>>=128;packed.range(255,128)=value;
   if((index&1u)==1&&row<m.mm)output[row*8u+((index&15u)>>1)]=packed;
 }
 // The public ABI remains eight FP32 header words. Conversion occurs only
 // at this boundary; alpha/den arithmetic above is bounded integer math.
 // One complete word per write avoids packed-memory read/modify/write.
 bus_t header=0;
 SOFTMAX16_HEADER_FIELDS:for(unsigned field=0;field<2u*QR;field++) {
#pragma HLS PIPELINE II=1
  unsigned row=field&(QR-1u);
  float value=0.f;
  if(row<m.mm)value=field<QR?softmax16::alpha_float(alpha[row]):
   softmax16::den_float(softmax16::Den(next[row].range(47,16)));
  header>>=32;header.range(255,224)=::fu(value);
  if((field&7u)==7)output[PROB_SCALE+(field>>3)]=header;
 }
}

static void softmax16_block(const FiveScoreBlock &input,ProbBlock &output,const Meta &m,
 const SoftmaxRowState previous[QR],SoftmaxRowState next[QR]) {
#pragma HLS INLINE off
 SoftmaxOctets scores,probability;
 SoftmaxGroupMeta score_meta,probability_meta;
#pragma HLS STREAM variable=scores depth=128
#pragma HLS STREAM variable=probability depth=128
#pragma HLS STREAM variable=score_meta depth=2
#pragma HLS STREAM variable=probability_meta depth=2
#pragma HLS BIND_STORAGE variable=scores type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=probability type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=score_meta type=fifo impl=lutram
#pragma HLS BIND_STORAGE variable=probability_meta type=fifo impl=lutram
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(softmax_score_groups(input,m,previous,scores,score_meta));
 C209_FIXED_TASK(softmax_exp_groups((m.mm+3u)>>2,m.nn,scores,score_meta,probability,probability_meta));
 C209_FIXED_TASK(softmax_pack_groups(m,probability,probability_meta,output,next));
}

static void softmax_five(unsigned count,unsigned zero_count,float factor,
 FiveScoreBlocks &scores,MetaStream &sin,FiveProbBlocks &probability,MetaStream &pout) {
#pragma HLS INLINE off
 static_assert(ATTENTION_CONTEXTS>=5,"All physical head lanes retain their state");
 softmax16::Half om[ATTENTION_CONTEXTS*QR];
 softmax16::Den ol[ATTENTION_CONTEXTS*QR];
 SoftmaxRowState previous[QR],next[QR];
#pragma HLS BIND_STORAGE variable=om type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ol type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=previous type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=next type=ram_2p impl=lutram
 SOFTMAX16_BLOCKS:for(unsigned block=0;block<count;block++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
  Meta m=sin.read();
  const unsigned state_base=m.context*QR;
  {
   hls::read_lock<FiveScoreBlock> input(scores);
   hls::write_lock<ProbBlock> output(probability);
   SOFTMAX16_LOAD_STATE:for(unsigned row=0;row<m.mm;row++) {
#pragma HLS PIPELINE II=1
    SoftmaxRowState state=0;
    state.range(15,0)=m.first?softmax16::Half(zero_count?0u:0xfbffu):om[state_base+row];
    state.range(47,16)=m.first?softmax16::Den(ap_uint<32>(zero_count)<<19):ol[state_base+row];
    previous[row]=state;
   }
   softmax16_block(input,output,m,previous,next);
   SOFTMAX16_SAVE_STATE:for(unsigned row=0;row<m.mm;row++) {
#pragma HLS PIPELINE II=1
    SoftmaxRowState state=next[row];
    om[state_base+row]=state.range(15,0);ol[state_base+row]=state.range(47,16);
   }
  }
  // Only after every probability word/header and state update has drained
  // is the existing whole-block lock released and metadata made visible.
  pout.write(m);
 }
}

