#pragma once
#include "c209_fixed_auxiliary.hpp"
#include "c209_fixed_compute_islands.hpp"

namespace c209five {
using namespace c2;
// Sequential auxiliary phases share one four-lane FP32 -> INT16 datapath.
// Bits63:0 are four stored activation halves,95:64 the original FP32 inverse.
// A single producer (the supervisor and its active phase) owns the request
// stream. The final stop is sent only after every response has been consumed.
using AuxQuantRequest=ap_uint<97>;
using AuxQuantRequests=hls::stream<AuxQuantRequest>;
using AuxQuantResults=hls::stream<ap_uint<64>>;
static AuxQuantRequest aux_quant_request(ap_uint<64> halves,float inverse) {
#pragma HLS INLINE
 AuxQuantRequest request=0;request.range(63,0)=halves;
 request.range(95,64)=fu(inverse);return request;
}
static void aux_quant4_service(AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
 bool stop=false;
 while(!stop) {
#pragma HLS PIPELINE II=1 style=flp
  AuxQuantRequest request=requests.read();stop=request[96];
  if(!stop) {
   float inverse=uf(request.range(95,64));ap_uint<64> output=0;
   for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
    c209mp::activation value=c209fixed::auxiliary_detail::input_half(
      ap_uint<16>(request.range(16*lane+15,16*lane)));
    output.range(16*lane+15,16*lane)=ap_uint<16>(::quant(c209mp::widen(value),inverse));
   }
   results.write(output);
  }
 }
}

static void aux_quant_row_requests(const c209mp::activation *row,unsigned n,
 float inverse,AuxQuantRequests &requests) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 for(unsigned d=0;d<(n+3u)/4u;d++) {
#pragma HLS PIPELINE II=1
  ap_uint<64> halves=0;
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   unsigned column=d*4+lane;
   if(column<n)halves.range(16*lane+15,16*lane)=c209mp::bits(row[column]);
  }
  requests.write(aux_quant_request(halves,inverse));
 }
}
static void aux_quant_row_store(bus_t *output,uint64_t address,unsigned n,unsigned stride,
 AuxQuantResults &results) {
#pragma HLS INLINE off
 bus_t packed=0;unsigned groups=(n+3u)/4u;
 for(unsigned group=0;group<(stride+3u)/4u;group++) {
#pragma HLS PIPELINE II=1
  ap_uint<64> value=group<groups?results.read():ap_uint<64>(0);
  // Patch68 has four real values in its last group and twelve zero values
  // in the fixed stride80. Other admitted widths are multiples of16.
  packed>>=64;packed.range(255,192)=value;
  if((group&3u)==3)output[(address>>5)+(group>>2)]=packed;
 }
}
static void aux_quant_row(const c209mp::activation *row,bus_t *output,uint64_t address,
 unsigned n,unsigned stride,float inverse,AuxQuantRequests &requests,AuxQuantResults &results) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(aux_quant_row_requests(row,n,inverse,requests));
 C209_FIXED_TASK(aux_quant_row_store(output,address,n,stride,results));
}
} // namespace c209five
