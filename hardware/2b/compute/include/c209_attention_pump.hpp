#pragma once
#include <ap_int.h>
#include <hls_stream.h>
// Transport oracle only; generated hardware must bind both RTL JSON leaves.
static inline void c209_attention_pump_oracle(hls::stream<ap_uint<256>>& input,
 hls::stream<ap_uint<256>>& output,unsigned count) {
 for(unsigned i=0;i<count;i++)output.write(input.read());
}
void c209_attention_q_pump(hls::stream<ap_uint<32>>& lengths,hls::stream<ap_uint<256>>& input,hls::stream<ap_uint<256>>& output);
void c209_attention_k_pump(hls::stream<ap_uint<32>>& lengths,hls::stream<ap_uint<256>>& input,hls::stream<ap_uint<256>>& output);
