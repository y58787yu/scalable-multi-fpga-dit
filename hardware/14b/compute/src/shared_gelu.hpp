#pragma once
#include "cosmos.hpp"
#include <hls_math.h>
#include "c209_nonlinear.hpp"
#include "c209_gelu_command_rtl.hpp"

// Exactly one top-level DATAFLOW process owns TWO continuously pipelined
// narrow GELU lanes (FP32 when C209_MIXED_PRECISION=0). One pair is accepted
// per clock after fill; the command count and ACK still count pairs. Internal
// payloads change; external FP32 DDR/SFP ABI stays unchanged. Unpack/math/repack
// processes, so no caller waits for the full nonlinear latency per pair.
namespace c2fast {
using scalar_word=ap_uint<c209mp::BITS>;
using pair_word=ap_uint<2*c209mp::BITS>;
using pair_output_stream=hls::stream<pair_word>;
struct NonlinearPair {pair_word values;ap_uint<2> valid;};
using nonlinear_pair_stream=hls::stream<NonlinearPair>;
struct NonlinearScalar {scalar_word value;bool valid;};
using nonlinear_scalar_stream=hls::stream<NonlinearScalar>;
using scalar_output_stream=hls::stream<scalar_word>;
struct GeluCommand {unsigned count;ap_uint<1> source;bool stop;};
struct GeluAck {unsigned count;ap_uint<1> source;};
using gelu_command_stream=hls::stream<GeluCommand>;
using gelu_ack_stream=hls::stream<GeluAck>;

static inline float shared_gelu_from_bits(unsigned value) {
 return c209mp::widen(c209mp::from_bits(scalar_word(value)));
}
static inline unsigned shared_gelu_to_bits(float value) {
 return c209mp::bits(c209mp::narrow(value)).to_uint();
}
static inline scalar_word scalar_from_float(float value) {
 return c209mp::bits(c209mp::narrow(value));
}
static inline float scalar_to_float(scalar_word value) {
 return c209mp::widen(c209mp::from_bits(value));
}
static inline pair_word pack_nonlinear_pair(float low,float high) {
 pair_word result=0;result.range(c209mp::BITS-1,0)=scalar_from_float(low);
 result.range(2*c209mp::BITS-1,c209mp::BITS)=scalar_from_float(high);return result;
}
static inline float unpack_nonlinear_lane(const pair_word &value,unsigned lane) {
 return scalar_to_float(value.range(c209mp::BITS*(lane+1)-1,c209mp::BITS*lane));
}
static inline float shared_gelu_formula(float value) {
 return c209mp::widen(c209nl::gelu(c209mp::narrow(value)));
}
static void nonlinear_split_pairs(nonlinear_pair_stream &input,
 nonlinear_scalar_stream &output,unsigned count) {
#pragma HLS INLINE off
 NonlinearPair pair;
 C209_NONLINEAR_SPLIT:for(unsigned i=0;i<2*count;i++) {
#pragma HLS PIPELINE II=1
  bool high=i&1u;if(!high)pair=input.read();
  NonlinearScalar value={high?scalar_word(pair.values.range(2*c209mp::BITS-1,c209mp::BITS)):scalar_word(pair.values.range(c209mp::BITS-1,0)),bool(high?pair.valid[1]:pair.valid[0])};
  output.write(value);
 }
}
static void nonlinear_join_pairs(scalar_output_stream &input,
 pair_output_stream &output,unsigned count) {
#pragma HLS INLINE off
 scalar_word low=0;
 C209_NONLINEAR_JOIN:for(unsigned i=0;i<2*count;i++) {
#pragma HLS PIPELINE II=1
  scalar_word value=input.read();
  if(i&1u){pair_word pair=0;pair.range(c209mp::BITS-1,0)=low;pair.range(2*c209mp::BITS-1,c209mp::BITS)=value;output.write(pair);}
  else low=value;
 }
}
static void shared_gelu_scalar_math(nonlinear_scalar_stream &input,
 scalar_output_stream &output,unsigned count) {
#pragma HLS INLINE off
 SHARED_GELU_SINGLE_LANE:for(unsigned i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
  NonlinearScalar value=input.read();c209mp::activation result=c209mp::activation(0);
  if(value.valid)result=c209nl::gelu(c209mp::from_bits(value.value));
  output.write(c209mp::bits(result));
 }
}
static void nonlinear_split_dual(nonlinear_pair_stream &input,
 nonlinear_scalar_stream &low,nonlinear_scalar_stream &high,unsigned count) {
#pragma HLS INLINE off
 C209_NONLINEAR_SPLIT_DUAL:for(unsigned i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
  NonlinearPair pair=input.read();
  NonlinearScalar a={scalar_word(pair.values.range(c209mp::BITS-1,0)),bool(pair.valid[0])};
  NonlinearScalar b={scalar_word(pair.values.range(2*c209mp::BITS-1,c209mp::BITS)),bool(pair.valid[1])};
  low.write(a);high.write(b);
 }
}
template<unsigned LANE> static void shared_gelu_regional_scalar_math(
 nonlinear_scalar_stream &input,scalar_output_stream &output,unsigned count) {
#pragma HLS INLINE off
 SHARED_GELU_DUAL_LANE:for(unsigned i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
  NonlinearScalar value=input.read();c209mp::activation result=c209mp::activation(0);
  if(value.valid)result=c209nl::gelu(c209mp::from_bits(value.value));
  output.write(c209mp::bits(result));
 }
}
static void nonlinear_join_dual(scalar_output_stream &low,scalar_output_stream &high,
 pair_output_stream &output,unsigned count) {
#pragma HLS INLINE off
 C209_NONLINEAR_JOIN_DUAL:for(unsigned i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
  pair_word pair=0;pair.range(c209mp::BITS-1,0)=low.read();
  pair.range(2*c209mp::BITS-1,c209mp::BITS)=high.read();output.write(pair);
 }
}
static void shared_gelu_pair_math(nonlinear_pair_stream &input,
 pair_output_stream &output,unsigned count) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 nonlinear_scalar_stream low,high;scalar_output_stream low_results,high_results;
#pragma HLS STREAM variable=low depth=16
#pragma HLS STREAM variable=high depth=16
#pragma HLS STREAM variable=low_results depth=16
#pragma HLS STREAM variable=high_results depth=16
 nonlinear_split_dual(input,low,high,count);
 shared_gelu_regional_scalar_math<0>(low,low_results,count);
 shared_gelu_regional_scalar_math<1>(high,high_results,count);
 nonlinear_join_dual(low_results,high_results,output,count);
}
static void shared_gelu_service(gelu_command_stream &commands,
 nonlinear_pair_stream &input,pair_output_stream &output,gelu_ack_stream &ack) {
#pragma HLS INLINE off
 bool stop=false;
 while(!stop) {
  GeluCommand command=commands.read();stop=command.stop;
  if(!stop) {
   shared_gelu_pair_math(input,output,command.count);
   GeluAck completion={command.count,command.source};ack.write(completion);
  }
 }
}
static void send_shared_gelu_command(gelu_command_stream &commands,unsigned count,unsigned source) {
#pragma HLS INLINE off
 // This is a standalone DATAFLOW process inside each exchange, not an inline
 // write in the caller that HLS can co-schedule with the final ACK read.
 GeluCommand command={count,ap_uint<1>(source),false};commands.write(command);
}
#ifdef __SYNTHESIS__
static void c209_vector_gelu_request(unsigned rows,unsigned columns,unsigned source,
 hls::stream<ap_uint<65>>&requests) {
#pragma HLS INLINE off
 ap_uint<65> request=0;
 request.range(31,0)=rows;request.range(63,32)=columns;request[64]=source&1u;
 requests.write(request);
}
static void c209_vector_gelu_publish(hls::stream<ap_uint<34>>&packed,
 gelu_command_stream&commands) {
#pragma HLS INLINE off
 ap_uint<34> p=packed.read();
 GeluCommand command={p.range(31,0).to_uint(),ap_uint<1>(p[32]),bool(p[33])};
 commands.write(command);
}
static void send_registered_vector_gelu_command(gelu_command_stream&commands,
 unsigned rows,unsigned columns,unsigned source) {
#pragma HLS INLINE off
 hls::stream<ap_uint<65>> requests("c209_gelu_requests");
 hls::stream<ap_uint<34>> packed("c209_gelu_commands");
#pragma HLS STREAM variable=requests depth=2
#pragma HLS STREAM variable=packed depth=2
#pragma HLS BIND_STORAGE variable=requests type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=packed type=fifo impl=srl
#pragma HLS DATAFLOW
 c209_vector_gelu_request(rows,columns,source,requests);
 c209_gelu_command_rtl(requests,packed);
 c209_vector_gelu_publish(packed,commands);
}
#endif
static void end_shared_gelu(gelu_ack_stream &ack) {
#pragma HLS INLINE off
#ifdef __SYNTHESIS__
 // Called only after the consumer has read every output pair and completed
 // its stores. Reading the service's completion then proves both channel
 // directions are drained before the supervisor switches request source.
 GeluAck completion=ack.read();(void)completion;
#else
 (void)ack;
#endif
}
} // namespace c2fast
