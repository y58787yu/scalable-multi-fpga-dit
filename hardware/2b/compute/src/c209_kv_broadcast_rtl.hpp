#pragma once
#include <ap_int.h>
#include <hls_stream.h>

// One finite FIFO request per invocation. Production MUST register the RTL
// JSON; the C body is only the host functional model. No scalar DATAFLOW port.
void c209_kv_broadcast_rtl(hls::stream<ap_uint<12>>& requests,
 hls::stream<ap_uint<256>>& payload,
 hls::stream<ap_uint<256>>& output0,hls::stream<ap_uint<256>>& output1,
 hls::stream<ap_uint<256>>& output2,hls::stream<ap_uint<256>>& output3,
 hls::stream<ap_uint<256>>& output4);
