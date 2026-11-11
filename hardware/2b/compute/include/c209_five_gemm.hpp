#pragma once
#include "c209_five_protocol.hpp"
#include "c209_gemm_scaled_integer.hpp"
#include "c209_five_half_max.hpp"
#include "c209_five_auxiliary.hpp"
#include "c209_five_fixed_job.hpp"
#include "c209_fixed_compute_islands.hpp"

// Five output-column shards. DDR row words go directly to the shared RTL
// transpose services; no HLS panel locks, storage or transpose controllers.
namespace c209five {
using namespace c2;

static unsigned gemm_k(const FixedJob &j) {
#pragma HLS INLINE
 // Fixed apple sampler.patches guarantees columns64..79 are zero. The
 // allocation/DDR stride stays80; only structural zero products are omitted.
 return j.op==GEMM&&j.k==80?64u:j.k.to_uint();
}
static unsigned tile_rows(const FixedJob &j,unsigned rb) {
#pragma HLS INLINE
 return j.m-rb>=256u?256u:128u;
}
static unsigned tile_columns(const FixedJob &j) {
#pragma HLS INLINE
 return j.n==64?64u:128u;
}
static void group_commands(const FixedJob &j,CmdStream &commands) {
#pragma HLS INLINE off
 unsigned inner=gemm_k(j),n=tile_columns(j);
 for(ap_uint<13> rb=0;rb<j.m;rb+=TM)for(ap_uint<13> nb=0;nb<j.n;nb+=SHARDS*TN)
  for(ap_uint<13> kb=0;kb<inner;kb+=TK) {
#pragma HLS PIPELINE II=1
   unsigned remaining=inner-kb.to_uint();unsigned kk=remaining<TK?remaining:TK;
   commands.write(make_command(tile_rows(j,rb),n,kk,kb==0,kb+kk==inner,active_mask(j.n,nb),false));
  }
}
static void operand_write(unsigned shard,ap_uint<256> value,C209_FIVE_OPERAND_PARAMS(out)) {
#pragma HLS INLINE
 switch(shard) {
  case 0:out0.write(value);break;case 1:out1.write(value);break;
  case 2:out2.write(value);break;case 3:out3.write(value);break;
  default:out4.write(value);break;
 }
}
// Each local quantizer owns four lanes and its own elastic boundary. This
// removes a single HLS ap_block from all sixteen floating-point lanes.
using AQuarter=hls::stream<ap_uint<97>>;
using AQuarterResult=hls::stream<ap_uint<64>>;
static ap_uint<22> a_word_count(const FixedJob &j) {
#pragma HLS INLINE
 // The admitted shapes give four exact constants. Do not synthesize a
 // general ceil(N/640) divider and two wide multipliers for a fixed task.
 ap_uint<22> rows=j.m;
 if(j.k==80)return rows<<4;                 // PATCH: 4 groups * K64/16
 if(j.n==64)return rows<<7;                // FINAL: 1 group * K2048/16
 if(j.n==1024||j.k==1024)return rows<<8;    // PROJ2*K128 or OUT4*K64
 if(j.n==4096)return (rows<<10)-(rows<<7); // FFN_UP: 7 groups * K2048/16
 return rows<<10;                         // FFN_DOWN: 4 groups * K4096/16
}
static unsigned input_row_words(const FixedJob &j) {
#pragma HLS INLINE
 // PATCH keeps its allocated K80 stride although its last16 values are zero.
 return j.k==80?5u:j.k==1024?64u:j.k==4096?256u:128u;
}
static void load_a_quarters(bus_t *memory,const FixedJob &j,
 AQuarter &q0,AQuarter &q1,AQuarter &q2,AQuarter &q3) {
#pragma HLS INLINE off
 unsigned inner=gemm_k(j),stride_words=input_row_words(j);bool convert=j.op==GEMM_F16_A_PARTIAL;
 WordOffset row_block_offset=0;
 float inverse[TM];
#pragma HLS BIND_STORAGE variable=inverse type=ram_1p impl=bram
 for(ap_uint<13> rb=0;rb<j.m;rb+=TM) {
  unsigned rows=tile_rows(j,rb);
  if(convert)for(unsigned r=0;r<rows;r++) {
#pragma HLS PIPELINE II=1
   inverse[r]=1.f/readf(memory,j.s0,rb+r);
  }
  for(ap_uint<13> nb=0;nb<j.n;nb+=SHARDS*TN)for(ap_uint<13> kb=0;kb<inner;kb+=TK) {
   unsigned remaining=inner-kb.to_uint();unsigned kk=remaining<TK?remaining:TK;
   WordOffset row_word_offset=row_block_offset+(kb>>4);
   for(unsigned panel=0;panel<rows;panel+=32)
    for(unsigned r=0;r<32;r++) {
     for(unsigned word=0;word<kk/16;word++) {
#pragma HLS PIPELINE II=1
     bus_t value=memory[word_address(j.a,WordOffset(row_word_offset+word))];unsigned scale=convert?fu(inverse[panel+r]):0u;
     ap_uint<97> p0=0,p1=0,p2=0,p3=0;
     p0.range(63,0)=value.range(63,0);p1.range(63,0)=value.range(127,64);
     p2.range(63,0)=value.range(191,128);p3.range(63,0)=value.range(255,192);
     p0.range(95,64)=scale;p1.range(95,64)=scale;p2.range(95,64)=scale;p3.range(95,64)=scale;
     p0[96]=convert;p1[96]=convert;p2[96]=convert;p3[96]=convert;
     q0.write(p0);q1.write(p1);q2.write(p2);q3.write(p3);
     }
     row_word_offset+=stride_words;
    }
  }
  row_block_offset+=rows==256u?(WordOffset(stride_words)<<8):(WordOffset(stride_words)<<7);
 }
}
template<unsigned GROUP> static void quantize_a_quarter(ap_uint<22> count,AQuarter &input,AQuarterResult &output) {
#pragma HLS INLINE off
 for(ap_uint<22> word=0;word<count;word++) {
#pragma HLS PIPELINE II=1
  ap_uint<97> packet=input.read();ap_uint<64> value=packet.range(63,0);
  float inverse=uf(unsigned(packet.range(95,64)));
  if(packet[96]) {
   ap_uint<64> converted=0;
   for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
    c209mp::activation f=c209fixed::auxiliary_detail::input_half(ap_uint<16>(value.range(16*lane+15,16*lane)));
    converted.range(16*lane+15,16*lane)=ap_uint<16>(quant(c209mp::widen(f),inverse));
   }
   value=converted;
  }
  output.write(value);
 }
}
static void join_a_quarters(ap_uint<22> count,AQuarterResult &q0,AQuarterResult &q1,
 AQuarterResult &q2,AQuarterResult &q3,OperandStream &output) {
#pragma HLS INLINE off
 for(ap_uint<22> word=0;word<count;word++) {
#pragma HLS PIPELINE II=1
  bus_t value=0;value.range(63,0)=q0.read();value.range(127,64)=q1.read();
  value.range(191,128)=q2.read();value.range(255,192)=q3.read();output.write(value);
 }
}
static void load_a_panels(bus_t *memory,const FixedJob &j,OperandStream &output) {
#pragma HLS INLINE off
 ap_uint<22> count=a_word_count(j);
 AQuarter q0,q1,q2,q3;AQuarterResult r0,r1,r2,r3;
#pragma HLS STREAM variable=q0 depth=8
#pragma HLS STREAM variable=q1 depth=8
#pragma HLS STREAM variable=q2 depth=8
#pragma HLS STREAM variable=q3 depth=8
#pragma HLS STREAM variable=r0 depth=8
#pragma HLS STREAM variable=r1 depth=8
#pragma HLS STREAM variable=r2 depth=8
#pragma HLS STREAM variable=r3 depth=8
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(load_a_quarters(memory,j,q0,q1,q2,q3));
 C209_FIXED_TASK(quantize_a_quarter<0>(count,q0,r0));
 C209_FIXED_TASK(quantize_a_quarter<1>(count,q1,r1));
 C209_FIXED_TASK(quantize_a_quarter<2>(count,q2,r2));
 C209_FIXED_TASK(quantize_a_quarter<3>(count,q3,r3));
 C209_FIXED_TASK(join_a_quarters(count,r0,r1,r2,r3,output));
}
static void load_b_panels(bus_t *memory,const FixedJob &j,C209_FIVE_OPERAND_PARAMS(out)) {
#pragma HLS INLINE off
 // Packed-only contract (descriptor flag0x40, checked at admission):
 // N640 group -> K128 segment -> shard -> N row -> contiguous K words.
 // The PC changes only layout, never INT16 values or scales. PATCH retains
 // the successful F3 structural-zero skip: descriptor K80, physical K64.
 // One complete shard slice is contiguous, permitting real AXI bursts.
 unsigned inner=gemm_k(j),n=tile_columns(j);
 for(ap_uint<13> rb=0;rb<j.m;rb+=TM) {
  WordOffset packed_offset=0;
  for(ap_uint<13> nb=0;nb<j.n;nb+=SHARDS*TN) {
   for(ap_uint<13> kb=0;kb<inner;kb+=TK) {
   unsigned remaining=inner-kb.to_uint();unsigned kk=remaining<TK?remaining:TK;
   for(ap_uint<3> shard=0;shard<SHARDS;shard++) {
    if(nb+shard*TN>=j.n)continue;
    unsigned slice_words=n*(kk>>4);
    C209_PACKED_B_CONTIGUOUS:for(unsigned word=0;word<slice_words;word++) {
#pragma HLS PIPELINE II=1
     operand_write(shard,memory[word_address(j.b,WordOffset(packed_offset+word))],C209_FIVE_STREAM_ARGS(out));
    }
    packed_offset+=slice_words;
   }
   }
  }
 }
}
using GemmPairs=hls::stream<ap_uint<96>>;
#define C209_GEMM_PAIR_PARAMS(p) GemmPairs &p##0,GemmPairs &p##1,GemmPairs &p##2,GemmPairs &p##3,GemmPairs &p##4
// Each static instance is placed beside its own 384-bit result boundary.
// A beat contains eight INT48 values; four accepted subbeats consume it.
// Keep one local beat while stalled. No 768-bit packet crosses the selector.
template<unsigned SHARD> static void gemm_result_pairs_local_group(unsigned rows,unsigned n,
 ResultStream &input,GemmPairs &output) {
#pragma HLS INLINE off
 // This finite function boundary must drain before requesting the next
 // output group. Flattening nb into this pipeline makes its next-group
 // input-empty stall hold the preceding group's final pair. The downstream
 // selector then cannot advance to the other shards, whose unfinished OUT
 // states prevent the MAC group from producing that next input: deadlock.
 ap_uint<384> holding=0;
 C209_GEMM_RESULT_LOCAL_II1:for(unsigned pair=0;pair<rows*(n/2);pair++) {
#pragma HLS PIPELINE II=1
  if((pair&3u)==0)holding=input.read();
  output.write(ap_uint<96>(holding.range(95,0)));
  holding>>=96;
 }
}
template<unsigned SHARD> static void gemm_result_pairs_local(const FixedJob &j,
 ResultStream &input,GemmPairs &output) {
#pragma HLS INLINE off
 unsigned n=tile_columns(j);
 for(ap_uint<13> rb=0;rb<j.m;rb+=TM) {
#pragma HLS LOOP_FLATTEN off
  unsigned rows=tile_rows(j,rb);
  for(ap_uint<13> nb=0;nb<j.n;nb+=SHARDS*TN) {
#pragma HLS LOOP_FLATTEN off
   if(nb+SHARD*TN>=j.n)continue;
   gemm_result_pairs_local_group<SHARD>(rows,n,input,output);
  }
 }
}
#include "c209_five_gemm_finish.hpp"

static unsigned gemm_ddr(bus_t *a,bus_t *b,bus_t *c,bus_t *x,const FixedJob &j,
                        CmdStream &commands,C209_FIVE_OPERAND_PARAMS(left),C209_FIVE_OPERAND_PARAMS(right),
                        C209_FIVE_RESULT_PARAMS(results),unsigned &tiles,c2fast::gelu_command_stream &gc,
                        c2fast::nonlinear_pair_stream &gi,c2fast::pair_output_stream &go,c2fast::gelu_ack_stream &ga) {
#pragma HLS INLINE off
 GemmPairs result0,result1,result2,result3,result4;
#pragma HLS STREAM variable=result0 depth=4
#pragma HLS STREAM variable=result1 depth=4
#pragma HLS STREAM variable=result2 depth=4
#pragma HLS STREAM variable=result3 depth=4
#pragma HLS STREAM variable=result4 depth=4
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(group_commands(j,commands));
 C209_FIXED_TASK(load_a_panels(a,j,left0));
 C209_FIXED_TASK(load_b_panels(b,j,C209_FIVE_STREAM_ARGS(right)));
 C209_FIXED_TASK(gemm_result_pairs_local<0>(j,results0,result0));
 C209_FIXED_TASK(gemm_result_pairs_local<1>(j,results1,result1));
 C209_FIXED_TASK(gemm_result_pairs_local<2>(j,results2,result2));
 C209_FIXED_TASK(gemm_result_pairs_local<3>(j,results3,result3));
 C209_FIXED_TASK(gemm_result_pairs_local<4>(j,results4,result4));
 C209_FIXED_TASK(finish_groups(c,x,j,C209_FIVE_STREAM_ARGS(result),gc,gi,go,ga));
 tiles=((j.m+255)/256)*((j.n+127)/128);
 return OK;
}
} // namespace c209five
#undef C209_GEMM_PAIR_PARAMS

