#pragma once
#include "c209_fixed_compute_islands.hpp"

namespace c209five {
namespace attention_detail {
using Sum32Requests=hls::stream<ap_uint<256>>;
using Sum32Results=hls::stream<ap_uint<128>>;

// One physical add site per context. Each finite stage drains its pipeline
// before waiting for the next stage's requests; an empty request FIFO cannot
// hold the previous stage's last responses inside a stalled FLP pipeline.
static void sum32_add4_service(Sum32Requests &requests,Sum32Results &results) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances=fadd limit=4
 for(ap_uint<3> stage=0;stage<5;stage++) {
#pragma HLS LOOP_FLATTEN off
  ap_uint<5> count=16u>>stage;
  SUM32_ADD4_PIPELINE:for(ap_uint<5> index=0;index<count;index++) {
#pragma HLS PIPELINE II=1 style=flp
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
   ap_uint<256> operands=requests.read();ap_uint<128> output=0;
   for(unsigned ctx=0;ctx<4;ctx++) {
#pragma HLS UNROLL
    float left=::uf(unsigned(operands.range(64*ctx+31,64*ctx)));
    float right=::uf(unsigned(operands.range(64*ctx+63,64*ctx+32)));
    float sum=left+right;
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=8
    output.range(32*ctx+31,32*ctx)=::fu(sum);
   }
   results.write(output);
  }
 }
}

// Every tree level has a distinct destination. The request loops only read
// floating-point bits; the result loops only write them. No dynamic in-place
// floating-point writeback or false dependency exemption is required.
static void sum32_tree_broker(const float values[4][32],float sums[4],
 Sum32Requests &requests,Sum32Results &results) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=values complete dim=0
#pragma HLS ARRAY_PARTITION variable=sums complete
 float s16[4][16],s8[4][8],s4[4][4],s2[4][2];
// Four context banks keep four additions/clock. Each stage is read twice
// per context while sending, then written once while receiving; those
// phases never overlap. Keep levels distinct: no in-place feedback mux.
#pragma HLS ARRAY_PARTITION variable=s16 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
#pragma HLS BIND_STORAGE variable=s16 type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=s8 type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=s4 type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=s2 type=ram_2p impl=lutram
 for(ap_uint<3> stage=0;stage<5;stage++) {
#pragma HLS LOOP_FLATTEN off
  ap_uint<5> count=16u>>stage;
  SUM32_SEND_LEVEL:for(ap_uint<5> index=0;index<count;index++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
   ap_uint<256> operands=0;
   for(unsigned ctx=0;ctx<4;ctx++) {
#pragma HLS UNROLL
    float left,right;
    if(stage==0){left=values[ctx][index];right=values[ctx][index+16];}
    else if(stage==1){left=s16[ctx][index];right=s16[ctx][index+8];}
    else if(stage==2){left=s8[ctx][index];right=s8[ctx][index+4];}
    else if(stage==3){left=s4[ctx][index];right=s4[ctx][index+2];}
    else{left=s2[ctx][index];right=s2[ctx][index+1];}
    operands.range(64*ctx+31,64*ctx)=::fu(left);
    operands.range(64*ctx+63,64*ctx+32)=::fu(right);
   }
   requests.write(operands);
  }
  // Sixteen slots hold the entire first stage while its requests are sent.
  // Subsequent stages contain 8/4/2/1 packets, all bounded by that capacity.
  SUM32_RECEIVE_LEVEL:for(ap_uint<5> index=0;index<count;index++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
   ap_uint<128> output=results.read();
   for(unsigned ctx=0;ctx<4;ctx++) {
#pragma HLS UNROLL
    float value=::uf(unsigned(output.range(32*ctx+31,32*ctx)));
    if(stage==0)s16[ctx][index]=value;
    else if(stage==1)s8[ctx][index]=value;
    else if(stage==2)s4[ctx][index]=value;
    else if(stage==3)s2[ctx][index]=value;
    else sums[ctx]=value;
   }
  }
 }
}

static void sum32_four(const float values[4][32],float sums[4]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=values complete dim=0
#pragma HLS ARRAY_PARTITION variable=sums complete
 Sum32Requests requests;Sum32Results results;
#pragma HLS STREAM variable=requests depth=4
#pragma HLS STREAM variable=results depth=16
#pragma HLS BIND_STORAGE variable=requests type=fifo impl=lutram
#pragma HLS BIND_STORAGE variable=results type=fifo impl=lutram
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(sum32_tree_broker(values,sums,requests,results));
 C209_FIXED_TASK(sum32_add4_service(requests,results));
}
} // namespace attention_detail
} // namespace c209five
