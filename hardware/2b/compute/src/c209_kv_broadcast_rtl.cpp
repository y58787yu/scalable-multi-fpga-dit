#include "c209_kv_broadcast_rtl.hpp"

// Functional cardinality/order model, not the production timing/control.
void c209_kv_broadcast_rtl(hls::stream<ap_uint<12>>& requests,
 hls::stream<ap_uint<256>>& payload,
 hls::stream<ap_uint<256>>& output0,hls::stream<ap_uint<256>>& output1,
 hls::stream<ap_uint<256>>& output2,hls::stream<ap_uint<256>>& output3,
 hls::stream<ap_uint<256>>& output4) {
 const unsigned words=requests.read().to_uint();
 for(unsigned index=0;index<words;index++) {
  ap_uint<256> word=payload.read();
  output0.write(word);output1.write(word);output2.write(word);
  output3.write(word);output4.write(word);
 }
}
