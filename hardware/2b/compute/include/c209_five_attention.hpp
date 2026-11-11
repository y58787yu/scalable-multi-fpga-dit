#pragma once
// Five query lanes of ONE head share ONE softmax and ONE FP32 PV updater. The
// mixed softmax uses FP16 scores, fixed16 exponentials and a wide integer
// denominator. The overlap profile folds invariant Q/PV factors before the
// hot loops; FP32 association differs, scalar widths and INT48 dots do not.
#include "c209_fixed_compute_islands.hpp"
#include "c209_five_protocol.hpp"
#include "c209_aux_quant_service.hpp"
#include "c209_five_sum32.hpp"
#include "c209_five_fixed_job.hpp"
#include "../src/c209_kv_broadcast_rtl.hpp"
namespace c209five {
namespace attention_detail {
using namespace c2;
using namespace c207;
static_assert(QR==32,"Same-head fixed attention retains32 physical rows per tile");
// Two32-query tiles share the existing K/V physical B banks. Their online
// softmax/PV state remains distinct until every key block completes.
constexpr unsigned ATTENTION_CONTEXTS=10;
// A whole five-head QK wave must be able to commit before its PV wave starts.
// Five probability slots break that bounded group dependency; reducing this
// to the legacy two slots creates a score -> probability -> QK-return cycle.
using FiveProbBlocks=hls::stream_of_blocks<ProbBlock,5>;
// INT48 -> FP32*qscale -> FP32*kscale -> finite-half is performed at
// collector ingress. This replaces, rather than duplicates, the old raw
// 768-bit score PIPO plus scale header. Arithmetic association is unchanged.
using FiveScoreBlock=bus_t[QR*128/16];
using FiveScoreBlocks=hls::stream_of_blocks<FiveScoreBlock,2>;
// The five heads are retained in full, but have one sequential reader/writer.
// A flat depth lets BRAM pack across head boundaries instead of allocating
// five separately rounded shallow/wide memories. Physical MAC A stays local.
using QueryGroup=bus_t[5*QW];
using QueryScales=float[5][QR];
using ScoreScales=bus_t[(5*QR+128)/8];
// Production attention has one fixed F16 destination format. Keep the same
// final rounding in update_five_to, but do not carry the legacy F32 store
// branch or a full generic Job through this memory sink.
struct FixedAttentionDDRSink {
 bus_t *c;const FixedJob &j;ap_uint<128> low_half;
 FixedAttentionDDRSink(bus_t *memory,const FixedJob &job):c(memory),j(job),low_half(0){}
 void write(const Meta &m,unsigned r,unsigned word,unsigned half,
            ap_uint<128> half_pack,bus_t) {
#pragma HLS INLINE
  if(!half)low_half=half_pack;
  else {
   // Dummy lanes still retire full physical QR32 results, but never address
   // an external query/output row. m.mm remains32 for all stream consumers.
   if(m.rb+r>=j.m)return;
   bus_t out=0;out.range(127,0)=low_half;out.range(255,128)=half_pack;
   // Expand before multiplying/shifting. The fixed tensor element offset
   // is bounded, while its externally supplied allocation base stays64b.
   ap_uint<27> elements=ap_uint<27>(m.rb+r)*ap_uint<27>(j.lc)+
                        ap_uint<27>(m.h)*128u+word*16u;
   c[word_address(j.c,elements>>4)]=out;
  }
 }
};
#include "c209_five_attention_math.inc.hpp"

// Query packing uses the one shared auxiliary quantizer. The request and
// response sides are separate processes; FIFO backpressure cannot form a
// write-then-read pipeline stall loop through the service.
static void store_query_quantized(QueryGroup &q,unsigned head,unsigned row,AuxQuantResults &results) {
#pragma HLS INLINE off
 const ap_uint<11> first=head*QW+row*8;
 bus_t packed=0;
 for(unsigned group=0;group<32;group++) {
#pragma HLS PIPELINE II=1
  packed>>=64;packed.range(255,192)=results.read();
  if((group&3u)==3)q[first+(group>>2)]=packed;
 }
}
static void quantize_query_row(const c209mp::activation row[128],float inverse,
 QueryGroup &q,unsigned head,unsigned r,AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(aux_quant_row_requests(row,128,inverse,requests));
 C209_FIXED_TASK(store_query_quantized(q,head,r,results));
}
static void load_queries(bus_t *a,const FixedJob &j,unsigned hb,unsigned rb,
 QueryGroup &q,QueryScales &qm,AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
 c209mp::activation row[128];
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 for(unsigned lane=0;lane<5;lane++)for(unsigned r=0;r<QR;r++) {
  ap_uint<31> maxima[8];
#pragma HLS ARRAY_PARTITION variable=maxima complete
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   maxima[l]=0;
  }
  bus_t packed=0;
  for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
   if(!(d&8u)) {
    const unsigned query_row=rb+lane*QR+r;
    WordOffset elements=WordOffset(query_row)*WordOffset(j.la)+WordOffset(hb)*128u+d;
    packed=query_row<j.m?a[word_address(j.a,elements>>4)]:bus_t(0);
   }
   for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
    c209mp::activation raw=c209mp::narrow(tensor::lane(packed,(d&8u)+l,F16));
    row[d+l]=raw;ap_uint<31> bits=c204::attention_absbits(c209mp::widen(raw));
    maxima[l]=bits>maxima[l]?bits:maxima[l];
   }
  }
  float mx=c204::attention_absmax8(maxima),scale=mx==0?1.f:mx/32760.f;
  // Quantize with the original row scale; only the score metadata includes
  // 1/sqrt(128). One row multiply replaces four per-score-lane multipliers.
  qm[lane][r]=scale*0.08838834764831845f;
  quantize_query_row(row,1.f/scale,q,lane,r,requests,results);
 }
}
using ValueScales=bus_t[16];
using ProbabilityScaleStream=hls::stream<bus_t>;
static void load_scales(bus_t *x,const FixedJob &j,unsigned hb,unsigned kb,
 const QueryScales &qm,ScoreScales &scale,ValueScales &values) {
#pragma HLS INLINE off
 unsigned nn=minv(128u,unsigned(j.kc)-kb),nb=blocks(j.kc),hs=head_stride(j.heads);
 for(unsigned lane=0;lane<5;lane++) {
  for(unsigned w=0;w<QR/8;w++) {
#pragma HLS PIPELINE II=1
   bus_t p=0;
   for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
    put_float(p,l,qm[lane][w*8+l]);
   }
   scale[lane*(QR/8)+w]=p;
  }
 }
  for(unsigned n=0;n<128;n++) {
#pragma HLS PIPELINE II=1
   ByteOffset bytes=(ByteOffset(kb+n)*hs+hb)*4u;
   uint64_t address=byte_address(j.s0,bytes);
   float value=n<nn?get_float(x[address>>5],unsigned((address>>2)&7u)):1.f;
   unsigned index=5*QR+n;put_float(scale[index/8],index%8,value);
  }
  for(unsigned w=0;w<16;w++) {
#pragma HLS PIPELINE II=1
   WordOffset offset=WordOffset(hb*nb+kb/128)*16u+w;
   values[w]=x[word_address(j.s1,offset)];
  }
}
static Meta group_meta(const FixedJob &j,unsigned hb,unsigned rb,unsigned kb,unsigned lane,unsigned context_base) {
 Meta m={hb,rb+lane*QR,kb,QR,minv(128u,unsigned(j.kc)-kb),context_base+lane,kb==0,kb+128>=j.kc};return m;
}
static void send_operand(unsigned lane,bus_t word,C209_FIVE_OPERAND_PARAMS(output)) {
#pragma HLS INLINE
 switch(lane) {
  case 0:output0.write(word);break;case 1:output1.write(word);break;
  case 2:output2.write(word);break;case 3:output3.write(word);break;
  default:output4.write(word);break;
 }
}
static void publish_pair(unsigned columns,unsigned inner,bool reuse_b,CmdStream &commands) {
#pragma HLS INLINE off
#pragma HLS LATENCY min=1
 // QK is followed by its one pending PV command. V can enter the other
 // physical B bank during QK result drain; private A remains protected in RTL.
 commands.write(make_command(QR,columns,128,true,true,31,true,false,reuse_b));
 commands.write(make_command(QR,128,inner,true,true,31,true,false,reuse_b));
}
static void emit_private_a(const QueryGroup &q,unsigned inner,
 FiveProbBlocks &probability,MetaStream &metadata,ProbabilityScaleStream &scales,
 C209_FIVE_OPERAND_PARAMS(output)) {
#pragma HLS INLINE off
 // The same producer owns Q then P on all five physical A channels.
 for(unsigned lane=0;lane<5;lane++)for(unsigned r=0;r<QR;r++)for(unsigned word=0;word<8;word++) {
#pragma HLS PIPELINE II=1
  send_operand(lane,q[lane*QW+r*8+word],C209_FIVE_STREAM_ARGS(output));
 }
 for(unsigned lane=0;lane<5;lane++) {
  Meta m=metadata.read();
  hls::read_lock<ProbBlock> input(probability);
  for(unsigned word=0;word<2*QR/8;word++) {
#pragma HLS PIPELINE II=1
   scales.write(input[PROB_SCALE+word]);
  }
  for(unsigned r=0;r<QR;r++)for(unsigned word=0;word<(inner>>4);word++) {
#pragma HLS PIPELINE II=1
   send_operand(lane,input[r*8+word],C209_FIVE_STREAM_ARGS(output));
  }
 }
}
// Read each same-head K/V panel once. The narrow local FIFO decouples DDR
// burst inference from the five independently backpressured physical ports.
static void read_shared_b(bus_t *b,const FixedJob &j,unsigned head,unsigned kb,
 unsigned columns,unsigned inner,OperandStream &words) {
#pragma HLS INLINE off
 const unsigned valid=minv(128u,unsigned(j.kc)-kb),nb=blocks(j.kc);
 const unsigned key_rows=columns==16?32u:128u;
 // K ABI: compact [head,stored_key,dimension128]. Small cross K remains
 // compact11/16 keys; physical32 padding is generated here, never read.
 const WordOffset key_base=(WordOffset(head)*WordOffset(j.kc)+kb)*8u;
 SAMEHEAD_K_CONTIGUOUS:for(unsigned word=0;word<key_rows*8u;word++) {
#pragma HLS PIPELINE II=1
  words.write((word>>3)<valid?b[word_address(j.b,key_base+word)]:bus_t(0));
 }
 // V remains [head,key_block,column,key], unchanged from the working ABI.
 const unsigned value_words=128u*(inner>>4);
 const WordOffset value_base=WordOffset(head*nb+kb/128)*VW;
 SAMEHEAD_V_CONTIGUOUS:for(unsigned word=0;word<value_words;word++) {
#pragma HLS PIPELINE II=1
  // Self K128 is fully contiguous. Fixed cross K16 takes one original
  // eight-word-stride column fragment, preserving the V padding contract.
  WordOffset offset=value_base+(inner==128?word:word*8u);
  words.write(b[word_address(j.s2,offset)]);
 }
}
template<class Port> static ap_uint<5> multicast_accept_pending(bus_t held,ap_uint<5> pending,
 Port &output0,Port &output1,Port &output2,Port &output3,Port &output4) {
#pragma HLS INLINE
  // A word is released only after EVERY lane accepted exactly once. Never
  // retry a successful lane because another lane is applying backpressure.
 // Every lane tests the SAME immutable old mask. In-place bit updates made
 // HLS chain five whole-vector phi muxes (G1:15.595ns), even though the bits
 // are independent. Merge five accepted bits only once after all attempts.
 bool accepted0=false,accepted1=false,accepted2=false,accepted3=false,accepted4=false;
 if(pending[0])accepted0=output0.write_nb(held);
 if(pending[1])accepted1=output1.write_nb(held);
 if(pending[2])accepted2=output2.write_nb(held);
 if(pending[3])accepted3=output3.write_nb(held);
 if(pending[4])accepted4=output4.write_nb(held);
 ap_uint<5> accepted=(ap_uint<5>(accepted4)<<4)|(ap_uint<5>(accepted3)<<3)|
                     (ap_uint<5>(accepted2)<<2)|(ap_uint<5>(accepted1)<<1)|ap_uint<5>(accepted0);
 return pending&~accepted;
}
static void multicast_shared_b(ap_uint<12> words,OperandStream &input,
 C209_FIVE_OPERAND_PARAMS(output)) {
#pragma HLS INLINE off
 // A FIFO token, not a DATAFLOW scalar input, configures the sequential RTL
 // actor. This wrapper publishes exactly one count before calling it.
 hls::stream<ap_uint<12>> requests;
#pragma HLS STREAM variable=requests depth=2
 requests.write(words);
 c209_kv_broadcast_rtl(requests,input,C209_FIVE_STREAM_ARGS(output));
}
static void load_and_broadcast_b(bus_t *b,const FixedJob &j,unsigned hb,unsigned kb,
 unsigned columns,unsigned inner,C209_FIVE_OPERAND_PARAMS(output)) {
#pragma HLS INLINE off
 OperandStream shared;
#pragma HLS STREAM variable=shared depth=8
#pragma HLS BIND_STORAGE variable=shared type=fifo impl=srl
 const unsigned words=(columns==16?32u:128u)*8u+128u*(inner>>4);
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(read_shared_b(b,j,hb,kb,columns,inner,shared));
 C209_FIXED_TASK(multicast_shared_b(words,shared,C209_FIVE_STREAM_ARGS(output)));
}
static void emit_ddr_b(bus_t *b,const FixedJob &j,unsigned hb,unsigned kb,
 unsigned columns,unsigned inner,bool reuse_b,C209_FIVE_OPERAND_PARAMS(output)) {
#pragma HLS INLINE off
 // Keep the conditional reuse gate outside the finite canonical DATAFLOW
 // region: either both reader/fork actors run, or neither actor starts.
 if(!reuse_b)load_and_broadcast_b(b,j,hb,kb,columns,inner,C209_FIVE_STREAM_ARGS(output));
}

static void receive_half_scores(const Meta &m,unsigned query_lane,const ScoreScales &scale,
 C209_FIVE_RESULT_PARAMS(co),FiveScoreBlocks &scores,MetaStream &metadata,unsigned physical_columns) {
#pragma HLS INLINE off
 {
  hls::write_lock<FiveScoreBlock> result(scores);
  float qm=0.f;bus_t packed=0;
  // One384-bit result beat is eight INT48 scores. Dequantization has the
  // EXACT original evaluation order and finite-half conversion. Removed
  // raw-score/header PIPO storage is not retained beside this half PIPO.
  SAMEHEAD_SCORE_INGRESS8:for(unsigned index=0;index<QR*16u;index++) {
#pragma HLS PIPELINE II=1
   const unsigned row=index>>4,octet=index&15u;
   if(octet==0)qm=get_float(scale[query_lane*(QR/8)+row/8],row%8);
   bus_t ks=scale[5*QR/8+octet];
   ap_uint<384> dots=octet<(physical_columns>>3)?read_result_beat(query_lane,C209_FIVE_STREAM_ARGS(co)):ap_uint<384>(0);
   ap_uint<128> halves=0;
   for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
    const unsigned key=octet*8u+lane;
    softmax16::Half encoded=0xfc00u;
    if(key<m.nn) {
     i48 dot=i48(dots.range(lane*48u+47,lane*48u));
     float dequantized=float(dot)*qm;
     dequantized=dequantized*get_float(ks,lane);
     encoded=softmax16::finite_half(dequantized);
    }
    halves.range(lane*16u+15,lane*16u)=encoded;
   }
   if(!(octet&1u))packed.range(127,0)=halves;
   else {packed.range(255,128)=halves;result[row*8u+(octet>>1)]=packed;}
  }
 }
 metadata.write(m);
}
static void collect_pair(const FixedJob &j,unsigned hb,unsigned rb,unsigned kb,
 unsigned columns,unsigned context_base,const ScoreScales &qkscale,const ValueScales &vscale,
 ProbabilityScaleStream &pscale,C209_FIVE_RESULT_PARAMS(results),FiveScoreBlocks &scores,
 MetaStream &sm,C209UpdateStream &updates,MetaStream &um) {
#pragma HLS INLINE off
 // Drain all five QK returns before waiting for any PV scale or result.
 // Five probability PIPO slots let softmax commit this whole wave even while
 // the first P writer waits for the physical core to release private A.
 for(unsigned lane=0;lane<5;lane++) {
  Meta m=group_meta(j,hb,rb,kb,lane,context_base);
  // One static ingress instance owns all five scalar result ports: do not
  // instantiate five independent eight-lane floating conversion engines.
  receive_half_scores(m,lane,qkscale,C209_FIVE_STREAM_ARGS(results),scores,sm,columns);
 }
 for(unsigned lane=0;lane<5;lane++) {
  bus_t scale[(2*QR+128)/8];
#pragma HLS BIND_STORAGE variable=scale type=ram_2p impl=lutram
  for(unsigned word=0;word<2*QR/8;word++) {
#pragma HLS PIPELINE II=1
   scale[word]=pscale.read();
  }
  for(unsigned word=0;word<16;word++) {
#pragma HLS PIPELINE II=1
   scale[2*QR/8+word]=vscale[word];
  }
  Meta m=group_meta(j,hb,rb,kb,lane,context_base);
  switch(lane) {
   case 0:c209_pv_receive(m,scale,results0,updates,um);break;
   case 1:c209_pv_receive(m,scale,results1,updates,um);break;
   case 2:c209_pv_receive(m,scale,results2,updates,um);break;
   case 3:c209_pv_receive(m,scale,results3,updates,um);break;
   default:c209_pv_receive(m,scale,results4,updates,um);break;
  }
 }
}
static void exchange_pair(bus_t *b,const FixedJob &j,unsigned hb,unsigned rb,unsigned kb,
 unsigned context_base,bool reuse_b,const QueryGroup &q,const ScoreScales &qkscale,const ValueScales &vscale,
 CmdStream &commands,C209_FIVE_OPERAND_PARAMS(ai),C209_FIVE_OPERAND_PARAMS(bi),
 C209_FIVE_RESULT_PARAMS(co),FiveScoreBlocks &scores,MetaStream &sm,
 FiveProbBlocks &probability,MetaStream &pm,C209UpdateStream &updates,MetaStream &um) {
#pragma HLS INLINE off
 unsigned columns=j.kc<=16?16u:128u,inner=(minv(128u,unsigned(j.kc)-kb)+15u)&~15u;
 ProbabilityScaleStream pscale;
 // All five 8-word scale headers must fit while PV waits for all A banks.
 // A smaller FIFO would create header -> A-load -> PV-output -> header cycle.
#pragma HLS STREAM variable=pscale depth=64
#pragma HLS BIND_STORAGE variable=pscale type=fifo impl=bram
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(publish_pair(columns,inner,reuse_b,commands));
 C209_FIXED_TASK(emit_private_a(q,inner,probability,pm,pscale,C209_FIVE_STREAM_ARGS(ai)));
 C209_FIXED_TASK(emit_ddr_b(b,j,hb,kb,columns,inner,reuse_b,C209_FIVE_STREAM_ARGS(bi)));
 C209_FIXED_TASK(collect_pair(j,hb,rb,kb,columns,context_base,qkscale,vscale,pscale,C209_FIVE_STREAM_ARGS(co),scores,sm,updates,um));
}
static void replace_query_scales(const QueryScales &qm,ScoreScales &scale) {
#pragma HLS INLINE off
 for(unsigned lane=0;lane<5;lane++)for(unsigned word=0;word<QR/8;word++) {
#pragma HLS PIPELINE II=1
  bus_t packed=0;
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   put_float(packed,l,qm[lane][word*8+l]);
  }
  scale[lane*(QR/8)+word]=packed;
 }
}
static void sequence(bus_t *a,bus_t *b,bus_t *x,const FixedJob &j,CmdStream &commands,
 C209_FIVE_OPERAND_PARAMS(ai),C209_FIVE_OPERAND_PARAMS(bi),C209_FIVE_RESULT_PARAMS(co),FiveScoreBlocks &scores,MetaStream &sm,
 FiveProbBlocks &probability,MetaStream &pm,C209UpdateStream &updates,MetaStream &um,
 AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
 QueryGroup q[2];QueryScales qm[2];ScoreScales scale;ValueScales vscale;
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
#pragma HLS ARRAY_PARTITION variable=qm complete dim=1
#pragma HLS BIND_STORAGE variable=q type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=scale type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=vscale type=ram_2p impl=bram
 // Five distinct Q tiles of the same head share each K/V load. Two waves
 // retain ten contexts; M2048/M4096 tails contain four/three valid query lanes.
 // Dummy lanes still execute and retire complete physical packets.
 // The absent second tail wave issues no commands and consumes no metadata.
 for(unsigned hb=0;hb<c209tp4::LOCAL_HEADS;hb++)for(unsigned rb=0;rb<j.m;rb+=10*QR) {
  const unsigned waves=rb+5*QR<j.m?2u:1u;
  for(unsigned slot=0;slot<waves;slot++)
   load_queries(a,j,hb,rb+slot*5*QR,q[slot],qm[slot],requests,results);
  for(unsigned kb=0;kb<j.kc;kb+=128) {
   load_scales(x,j,hb,kb,qm[0],scale,vscale);
   // One static call site retains a single same-head exchange controller.
   // Normal K/V loads are followed by reuse of the same two physical banks.
   for(unsigned slot=0;slot<waves;slot++) {
#pragma HLS LOOP_FLATTEN off
    if(slot)replace_query_scales(qm[1],scale);
    exchange_pair(b,j,hb,rb+slot*5*QR,kb,slot*5,bool(slot),q[slot],scale,vscale,commands,C209_FIVE_STREAM_ARGS(ai),
     C209_FIVE_STREAM_ARGS(bi),C209_FIVE_STREAM_ARGS(co),scores,sm,probability,pm,updates,um);
   }
  }
 }
}
static void update_ddr(bus_t *c,const FixedJob &j,unsigned count,C209UpdateStream &updates,MetaStream &metadata) {
#pragma HLS INLINE off
 FixedAttentionDDRSink sink(c,j);update_five_to(sink,count,updates,metadata);
}
static void pipeline(bus_t *a,bus_t *b,bus_t *c,bus_t *x,const FixedJob &j,CmdStream &commands,
 C209_FIVE_OPERAND_PARAMS(ai),C209_FIVE_OPERAND_PARAMS(bi),C209_FIVE_RESULT_PARAMS(co),
 AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
 unsigned count=(j.m==4096u?130u:65u)*c209tp4::LOCAL_HEADS*blocks(unsigned(j.kc));
 // Fixed head dimension128; no arbitrary runtime attention factor exists.
 const float factor=0.08838834764831845f;
 FiveScoreBlocks scores;FiveProbBlocks probability;MetaStream sm,pm,um;C209UpdateStream updates;
#pragma HLS STREAM variable=sm depth=2
#pragma HLS STREAM variable=pm depth=5
#pragma HLS STREAM variable=um depth=2
#pragma HLS STREAM variable=updates depth=32
#pragma HLS BIND_STORAGE variable=updates type=fifo impl=srl
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(sequence(a,b,x,j,commands,C209_FIVE_STREAM_ARGS(ai),C209_FIVE_STREAM_ARGS(bi),C209_FIVE_STREAM_ARGS(co),scores,sm,probability,pm,updates,um,requests,results));
 C209_FIXED_TASK(softmax_five(count,j.n-j.kc,factor,scores,sm,probability,pm));
 C209_FIXED_TASK(update_ddr(c,j,count,updates,um));
}
} // namespace attention_detail
static unsigned attention_ddr(c2::bus_t *a,c2::bus_t *b,c2::bus_t *c,c2::bus_t *x,const FixedJob &j,
 CmdStream &commands,C209_FIVE_OPERAND_PARAMS(ai),C209_FIVE_OPERAND_PARAMS(bi),C209_FIVE_RESULT_PARAMS(co),unsigned &tiles,
 AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
 tiles=0;
#ifndef __SYNTHESIS__
 if((j.m!=2048&&j.m!=4096)||j.heads!=8||
    !((j.n==4096&&j.kc==4096)||(j.n==512&&(j.kc==11||j.kc==16)))||
    j.fa!=c2::F16||j.fc!=c2::F16)return c2::BAD_SHAPE;
#endif
 attention_detail::pipeline(a,b,c,x,j,commands,C209_FIVE_STREAM_ARGS(ai),C209_FIVE_STREAM_ARGS(bi),C209_FIVE_STREAM_ARGS(co),requests,results);
 // Public status counts logical work; internal consumers use physical_count
 // including the two/one dummy Q blocks per head for M4096/M2048.
 tiles=2u*c209tp4::LOCAL_HEADS*(unsigned(j.m)/32u)*((unsigned(j.kc)+127u)/128u);return c2::OK;
}
} // namespace c209five
