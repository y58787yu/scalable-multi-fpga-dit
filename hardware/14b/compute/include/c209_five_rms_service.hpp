#pragma once
#include "c209_fixed_auxiliary.hpp"
#include "c209_five_protocol.hpp"
#include "c209_five_fixed_job.hpp"
#include "c209_fixed_compute_islands.hpp"
#include "c209_five_aux_mixed.hpp"

// One physical RMS128 front end for both fixed cross-Q and self-Q/K.
// Only its output tail differs. Keep four FP32 lanes and the exact original
// 16-phase/eight-lane reduction DAG; no reciprocal approximation or FMA.
namespace c209five {
using namespace c2;
namespace rms128_detail {
using namespace c2;
using HalfQuads=hls::stream<ap_uint<64>>;
using NormalizedRow=ap_uint<128>[16];
using NormalizedRows=hls::stream_of_blocks<NormalizedRow,2>;
constexpr unsigned MAX_HEAD_ROWS=c209tp4::MAX_LOCAL_ROWS*c209tp4::LOCAL_HEADS;

static void normalize4(ap_uint<14> rows,const float gamma[128],
 c204::short_norm_raw_words &raw,c204::short_norm_scalars &inverse,HalfQuads &output) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
 float inv=0.f;c204::short_norm_raw_word original=0;
 // Flatten across row boundaries: the floating pipelines do not drain and
 // refill for every128-value head. One row occupies32 II1 iterations.
 for(ap_uint<19> half=0;half<(ap_uint<19>(rows)<<5);half++) {
#pragma HLS PIPELINE II=1 style=flp
  unsigned column=half&31u;
  if(column==0)inv=inverse.read();
  if(!(column&1u))original=raw.read();
  ap_uint<64> selected=(column&1u)?ap_uint<64>(original.range(127,64)):
                                      ap_uint<64>(original.range(63,0));
  ap_uint<64> encoded=0;
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   c209mp::word bits=selected.range(16*lane+15,16*lane);
   float centered=c209mp::widen(c209mp::from_bits(bits))-0.f;
   float scaled=centered*inv;
   float normalized=scaled*gamma[column*4+lane];
   float value=normalized+0.f;
   encoded.range(16*lane+15,16*lane)=c209mp::bits(c209mp::narrow(c209mp::round(value)));
  }
  output.write(encoded);
 }
}

static void store_plain(bus_t *output,uint64_t address,ap_uint<14> rows,HalfQuads &input) {
#pragma HLS INLINE off
 ap_uint<64> quarters[3];
#pragma HLS ARRAY_PARTITION variable=quarters complete
 for(ap_uint<19> half=0;half<(ap_uint<19>(rows)<<5);half++) {
#pragma HLS PIPELINE II=1 style=flp
  ap_uint<64> value=input.read();unsigned slot=half&3u;
  if(slot==3) {
   bus_t word;
   word.range(63,0)=quarters[0];word.range(127,64)=quarters[1];
   word.range(191,128)=quarters[2];word.range(255,192)=value;
   output[word_address(address,WordOffset(half>>2))]=word;
  } else quarters[slot]=value;
 }
}

static void pack_rows(ap_uint<14> rows,HalfQuads &input,NormalizedRows &output) {
#pragma HLS INLINE off
 for(ap_uint<14> r=0;r<rows;r++) {
  hls::write_lock<NormalizedRow> row(output);
  ap_uint<64> low=0;
  for(unsigned half=0;half<32;half++) {
#pragma HLS PIPELINE II=1
   ap_uint<64> value=input.read();
   if(half&1u) {
    ap_uint<128> pair;
    pair.range(63,0)=low;pair.range(127,64)=value;row[half>>1]=pair;
   } else low=value;
  }
 }
}

// One coefficient bank per token, prefetched while the previous token's ten
// heads are consumed. Four binary16 RoPE lanes; their pipeline crosses
// head and token boundaries rather than draining after every128 values.
using RopeCoefficients=ap_uint<128>[32];
using RopeCoefficientBlocks=hls::stream_of_blocks<RopeCoefficients,2>;

// Keep the variable token loop outside this fixed-size AXI reader. The old
// flattened token/word loop issued a cosine request on its terminal iteration
// without consuming that response. Each call here owns exactly one bank and
// reads exactly16 cosine/sine pairs before returning and releasing the bank.
static void load_rope_token_coefficients(bus_t *aux,uint64_t cosine,uint64_t sine,
 ap_uint<11> token,RopeCoefficientBlocks &output) {
#pragma HLS INLINE off
 hls::write_lock<RopeCoefficients> coefficients(output);
 C209_ROPE_TOKEN_WORDS:for(unsigned w=0;w<16;w++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=2
  WordOffset offset=(WordOffset(token)<<4)+w;
  bus_t cv=aux[word_address(cosine,offset)];
  bus_t sv=aux[word_address(sine,offset)];
  ap_uint<128> low=0,high=0;
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   low.range(16*lane+15,16*lane)=c209mp::bits(c209mp::narrow(uf(cv.range(32*lane+31,32*lane))));
   low.range(64+16*lane+15,64+16*lane)=c209mp::bits(c209mp::narrow(uf(sv.range(32*lane+31,32*lane))));
   high.range(16*lane+15,16*lane)=c209mp::bits(c209mp::narrow(uf(cv.range(128+32*lane+31,128+32*lane))));
   high.range(64+16*lane+15,64+16*lane)=c209mp::bits(c209mp::narrow(uf(sv.range(128+32*lane+31,128+32*lane))));
  }
  coefficients[w*2]=low;coefficients[w*2+1]=high;
 }
}

static void load_rope_coefficients(bus_t *aux,uint64_t cosine,uint64_t sine,
 ap_uint<11> tokens,RopeCoefficientBlocks &output) {
#pragma HLS INLINE off
 C209_ROPE_TOKENS:for(ap_uint<11> token=0;token<tokens;token++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
  load_rope_token_coefficients(aux,cosine,sine,token,output);
 }
}
// The normalized row, coefficient bank and arithmetic have one consumer.
// Nothing exports a385-bit {values,coefficients} request to a distant actor.
// Locks are held for exactly one head/token. Explicit acquire/release lets
// this remain one continuous II1 loop across all heads and tokens.
static void rope_local4(ap_uint<14> rows,NormalizedRows &input,
 RopeCoefficientBlocks &coefficients,HalfQuads &output) {
#pragma HLS INLINE off
 hls::read_buf<NormalizedRow> row(input);
 hls::read_buf<RopeCoefficients> coefficient(coefficients);
 ap_uint<5> quarter=0;ap_uint<9> token_quarter=0;
 const ap_uint<19> count=ap_uint<19>(rows)<<5;
 C209_FIXED_ROPE_LOCAL4:for(ap_uint<19> index=0;index<count;index++) {
#pragma HLS PIPELINE II=1 style=flp
  if(token_quarter==0)coefficient.acquire();
  if(quarter==0)row.acquire();
  ap_uint<128> own=row[quarter>>1],opposite=row[(quarter>>1)^8u];
  ap_uint<64> values=(quarter&1u)?ap_uint<64>(own.range(127,64)):ap_uint<64>(own.range(63,0));
  ap_uint<64> others=(quarter&1u)?ap_uint<64>(opposite.range(127,64)):ap_uint<64>(opposite.range(63,0));
  ap_uint<128> local_coefficients=coefficient[quarter];
  // All operands are captured before releasing a bank. Its lifetime does
  // not include the later floating-point/output stages; the next head/token
  // can acquire the next bank while these captured values finish locally.
  if(quarter==31)row.release();
  if(token_quarter==319)coefficient.release();
  ap_uint<64> encoded=0;
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   c209mp::activation value=c209mp::from_bits(c209mp::word(values.range(16*lane+15,16*lane)));
   c209mp::activation rotated=c209mp::from_bits(c209mp::word(others.range(16*lane+15,16*lane)));
   if(quarter<16)rotated=aux_mixed::negate(rotated);
   c209mp::activation cosine=c209mp::from_bits(c209mp::word(local_coefficients.range(16*lane+15,16*lane)));
   c209mp::activation sine=c209mp::from_bits(c209mp::word(local_coefficients.range(64+16*lane+15,64+16*lane)));
   c209mp::activation rounded=aux_mixed::add(aux_mixed::multiply(value,cosine),aux_mixed::multiply(rotated,sine));
   encoded.range(16*lane+15,16*lane)=c209mp::bits(rounded);
  }
  output.write(encoded);
  if(token_quarter==319)token_quarter=0;
  else token_quarter++;
  quarter++;
 }
}
static void rope_tail(bus_t *output,bus_t *aux,uint64_t address,uint64_t cosine,
 uint64_t sine,ap_uint<14> rows,HalfQuads &input) {
#pragma HLS INLINE off
 NormalizedRows normalized;RopeCoefficientBlocks coefficients;HalfQuads rotated;
 //64 four-value results buffer two complete heads. This is an elastic
 //boundary; no DDR-ready signal directly drives every local coefficient bit.
#pragma HLS STREAM variable=rotated depth=64
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(pack_rows(rows,input,normalized));
 C209_FIXED_TASK(load_rope_coefficients(aux,cosine,sine,rows/10,coefficients));
 C209_FIXED_TASK(rope_local4(ap_uint<14>(rows),normalized,coefficients,rotated));
 C209_FIXED_TASK(store_plain(output,address,rows,rotated));
}

static void output_tail(bus_t *output,bus_t *aux,uint64_t address,uint64_t cosine,
 uint64_t sine,ap_uint<14> rows,bool rope,HalfQuads &input) {
#pragma HLS INLINE off
 if(rope)rope_tail(output,aux,address,cosine,sine,rows,input);
 else store_plain(output,address,rows,input);
}

static void pipeline(bus_t *input,bus_t *output,bus_t *aux,uint64_t input_address,
 uint64_t output_address,uint64_t cosine,uint64_t sine,ap_uint<14> rows,bool rope,
 const float gamma[128]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
 c204::short_norm_raw_words raw;
 c204::short_norm_words squares,pairs8,pairs4,pairs2,lanes;
 c204::short_norm_scalars inverse;
 HalfQuads normalized;
#pragma HLS STREAM variable=raw depth=64
#pragma HLS STREAM variable=squares depth=4
#pragma HLS STREAM variable=pairs8 depth=4
#pragma HLS STREAM variable=pairs4 depth=4
#pragma HLS STREAM variable=pairs2 depth=4
#pragma HLS STREAM variable=lanes depth=4
#pragma HLS STREAM variable=inverse depth=4
#pragma HLS STREAM variable=normalized depth=64
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(c209fixed::auxiliary_detail::rms_read_square<MAX_HEAD_ROWS>(input,input_address,raw,squares,rows));
 C209_FIXED_TASK((c204::short_norm_phase_pairs<16,2>(rows,squares,pairs8)));
 C209_FIXED_TASK((c204::short_norm_phase_pairs<8,1>(rows,pairs8,pairs4)));
 C209_FIXED_TASK((c204::short_norm_phase_pairs<4,1>(rows,pairs4,pairs2)));
 C209_FIXED_TASK((c204::short_norm_phase_pairs<2,1>(rows,pairs2,lanes)));
 C209_FIXED_TASK(c204::short_norm_inverse(rows,1e-6f,lanes,inverse));
 C209_FIXED_TASK(normalize4(rows,gamma,raw,inverse,normalized));
 C209_FIXED_TASK(output_tail(output,aux,output_address,cosine,sine,rows,rope,normalized));
}
} // namespace rms128_detail

// There is exactly one production call site, shared by case4/case17 in the
// supervisor. Do not put separate wrappers/calls around these two modes.
static unsigned rms128_service(bus_t *input,bus_t *output,bus_t *aux,const FixedJob &j,unsigned &tiles) {
#pragma HLS INLINE off
 bool rope=j.op==RMS_ROPE_FUSED;
 // Both production modes have exactly1280 or1408 tokens and ten heads.
 // Do not carry a general row multiplier through the shared service.
 bool long_slice=rope?(j.m==1408u):(j.m==14080u);
 ap_uint<14> row_count=long_slice?14080u:12800u;
 float gamma[128];
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=8
 for(unsigned w=0;w<16;w++) {
#pragma HLS PIPELINE II=1
  bus_t value=aux[word_address(j.s0,WordOffset(w))];
  for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
   gamma[w*8+lane]=uf(value.range(32*lane+31,32*lane));
  }
 }
 rms128_detail::pipeline(input,output,aux,j.a,j.c,j.s1,j.s2,row_count,rope,gamma);
 tiles=j.m;return OK;
}
} // namespace c209five
