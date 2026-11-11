#pragma once
#ifndef COSMOS_C207
#error C207 MAC service requires the explicit COSMOS_C207 candidate build
#endif
#ifdef COSMOS_C209
#include "c209_mac_rtl.hpp"
#else
#include "c207_mac_rtl.hpp"
#endif
// Preserve external ABI; pack the private FIFO without HLS struct layout assumptions.
namespace c204 {
static void c207_pack_mac_commands(cmd_stream &in,hls::stream<ap_uint<99>> &out) {
#pragma HLS INLINE off
    bool stop=false;
    while(!stop) {
#pragma HLS PIPELINE II=1
        MacCommand d=in.read();
        ap_uint<99> p=0;
        p.range(31,0)=d.m; p.range(63,32)=d.n; p.range(95,64)=d.k;
        p[96]=d.reset; p[97]=d.commit; p[98]=d.stop;
        out.write(p); stop=d.stop;
    }
}
#ifdef COSMOS_C209
template<unsigned ENGINE> static void c209_mac_service(cmd_stream &commands,input_stream &ain,
                               input_stream &bin,result_stream &results) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<ap_uint<99>> packed;
#pragma HLS STREAM variable=packed depth=4
    c207_pack_mac_commands(commands,packed);
    if(ENGINE==0)c209_mac0_rtl_service(packed,ain,bin,results);
    else c209_mac1_rtl_service(packed,ain,bin,results);
}
// Native-only legacy helper used by numerical reference fixtures.
#ifndef __SYNTHESIS__
static void single_mac_service(cmd_stream &commands,input_stream &ain,
                               input_stream &bin,result_stream &results) {
    c209_mac_service<0>(commands,ain,bin,results);
}
#endif
#else
static void single_mac_service(cmd_stream &commands,input_stream &ain,
                               input_stream &bin,result_stream &results) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<ap_uint<99>> packed;
#pragma HLS STREAM variable=packed depth=2
    c207_pack_mac_commands(commands,packed);
    c207_mac_rtl_service(packed,ain,bin,results);
}
#endif
}
