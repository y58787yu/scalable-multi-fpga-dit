#pragma once
#include <ap_int.h>
#include <hls_stream.h>
// HLS interface uses 200 MHz; physical black boxes require explicit common
// mac_fast_clk=300 MHz and active-high mac_fast_rst, wired by prepare-IP.
// M<=256, N<=128, K<=256; operands panel32/K/half16, signed48 results.
void c209_mac0_rtl_service(hls::stream<ap_uint<99>>& commands,
                           hls::stream<ap_uint<256>>& ain,
                           hls::stream<ap_uint<256>>& bin,
                           hls::stream<ap_uint<768>>& results);
void c209_mac1_rtl_service(hls::stream<ap_uint<99>>& commands,
                           hls::stream<ap_uint<256>>& ain,
                           hls::stream<ap_uint<256>>& bin,
                           hls::stream<ap_uint<768>>& results);
