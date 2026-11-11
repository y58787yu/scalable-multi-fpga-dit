#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include "c209_tp4_config.hpp"
// HLS2023.2 blackbox binding traces stream arrays even through scalar
// wrappers. Keep the physical channel objects and every task port scalar.
#define C209_FIVE_OPERAND_PARAMS(p) \
 ::c209five::OperandStream &p##0,::c209five::OperandStream &p##1, \
 ::c209five::OperandStream &p##2,::c209five::OperandStream &p##3,::c209five::OperandStream &p##4
#define C209_FIVE_RESULT_PARAMS(p) \
 ::c209five::ResultStream &p##0,::c209five::ResultStream &p##1, \
 ::c209five::ResultStream &p##2,::c209five::ResultStream &p##3,::c209five::ResultStream &p##4
#define C209_FIVE_STREAM_ARGS(p) p##0,p##1,p##2,p##3,p##4
// Fixed five-shard boundary. No DDR pointers or general tensor descriptors
// enter the 300MHz matrix island. One group has one geometry and active mask.
// Operands enter the 200MHz broker as row-major 32-row panels; ten shared
// RTL transpose services produce the original column-major MAC packets.
namespace c209five {
constexpr unsigned SHARDS=5,TM=256,TN=128,TK=128,ATTN_ROWS=32;
using Command=ap_uint<12>;
using CmdStream=hls::stream<Command>;
using OperandStream=hls::stream<ap_uint<256>>;
using ResultStream=hls::stream<ap_uint<384>>;
using ResultPacketStream=hls::stream<ap_uint<768>>;
using PerfStream=hls::stream<ap_uint<256>>;
static ap_uint<384> read_result_beat(unsigned shard,C209_FIVE_RESULT_PARAMS(input)) {
#pragma HLS INLINE
 switch(shard) {
  case 0:return input0.read();case 1:return input1.read();
  case 2:return input2.read();case 3:return input3.read();default:return input4.read();
 }
}
static ap_uint<768> read_result(unsigned shard,C209_FIVE_RESULT_PARAMS(input)) {
#pragma HLS INLINE
 ap_uint<768> packet=0;
 packet.range(383,0)=read_result_beat(shard,C209_FIVE_STREAM_ARGS(input));
 packet.range(767,384)=read_result_beat(shard,C209_FIVE_STREAM_ARGS(input));
 return packet;
}
enum FixedOp : unsigned { PREPARE_KV_GLOBAL=13,NORM_ADALN_FIXED=14,
 GEMM_GELU_F16=15,GEMM_F16_A_PARTIAL=16,RMS_ROPE_FUSED=17 };
static Command make_command(unsigned m,unsigned n,unsigned k,bool reset,bool commit,
                            unsigned mask,bool private_a,bool stop=false,bool reuse_b=false) {
#pragma HLS INLINE
 // Closed geometry: M32/128/256,K16/64/128; shared N64/128,
 // private N16/128. N16 operands still carry a complete32-row B panel.
 // This wire carries only finite sizes and prefix masks1/3/7/31.
 // The two added high bits encode the2B N1024/N4096 tail groups.
 // Previously reserved M code3 retains M32 and reuses the matching physical
 // B bank. Only the second query group of fixed attention emits this code;
 // the low10 command bits retain their original meaning.
 Command p=0;p.range(1,0)=reuse_b?3u:m==256?2u:m==128?1u:0u;p[2]=n==128;
 p.range(4,3)=k==128?2u:k==64?1u:0u;
 p[5]=reset;p[6]=commit;p[7]=stop;p[8]=mask==31;p[9]=private_a;
 p.range(11,10)=mask==3?1u:mask==7?2u:0u;return p;
}
static unsigned active_mask(unsigned total_columns,unsigned base) {
#pragma HLS INLINE
 unsigned mask=0;for(unsigned s=0;s<SHARDS;s++) {
#pragma HLS UNROLL
  if(base+s*TN<total_columns)mask|=1u<<s;
 }return mask;
}
} // namespace c209five
