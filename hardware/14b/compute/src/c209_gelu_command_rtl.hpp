#pragma once
#include <ap_int.h>
#include <hls_stream.h>

// One transaction per invocation. The synthesis entry MUST bind the matching
// RTL JSON; final export checks the exact RTL bytes and reachable instance.
// request = {source[0], columns[31:0], rows[31:0]}
// command = {stop=0, source[0], count[31:0]}
void c209_gelu_command_rtl(hls::stream<ap_uint<65>>& requests,
                           hls::stream<ap_uint<34>>& commands);
