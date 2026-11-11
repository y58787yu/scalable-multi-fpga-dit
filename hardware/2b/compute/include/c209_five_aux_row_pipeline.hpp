#pragma once
#include "c209_aux_quant_service.hpp"
#include "c209_five_aux_mixed.hpp"
#include "c209_five_fixed_job.hpp"

namespace c209five { namespace aux_row_detail {
using Row=c209mp::activation[c209fixed::HIDDEN];
// Two prefetch slots plus the supervisor's one compute row. The generated
// single-slot PIPO used a rendezvous full/empty protocol incompatible with
// its generated full_n-gated publisher and could never publish its first row.
// Use the standard two-slot PIPO; never patch only its reset/full flag.
// A consumer never writes through a read_lock. After the local copy releases
// a prefetch slot, subsequent DDR loading overlaps current-row work.
// Explicit port width is essential: partitioning the downstream scalar-row
// parameter does not partition a stream-of-blocks memory. The scalar version
// inferred one16-bit dual-port bank and serialized8 values into4 cycles.
// The last word belongs to this same committed slot: mean/inverse_std.
// Data and statistics therefore cannot be published or recycled separately.
constexpr unsigned ROW_META=c209fixed::HIDDEN/8;
using PrefetchRow=ap_uint<128>[ROW_META+1];
using PrefetchRows=hls::stream_of_blocks<PrefetchRow,2>;
using Scales=hls::stream<float>;
using RowReady=hls::stream<ap_uint<1>>;
using Coefficients=ap_uint<32>[2048]; // low16 shift, high16 gain

static void load_coefficients(bus_t *aux,uint64_t shift,uint64_t gain,
 Coefficients &coefficients) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=coefficients cyclic factor=4
#pragma HLS BIND_STORAGE variable=coefficients type=ram_2p impl=bram
 // Four packed coefficient pairs per cycle; one pair uses one bank port.
 // DDR ABI is still two FP32 arrays, narrowed once on this admission boundary.
 C209_AUX_COEFFICIENT_WORDS:for(unsigned word=0;word<256;word++) {
#pragma HLS PIPELINE off
  bus_t shifts=aux[word_address(shift,WordOffset(word))];
  bus_t gains=aux[word_address(gain,WordOffset(word))];
  C209_AUX_COEFFICIENT_HALVES:for(unsigned halfword=0;halfword<2;halfword++) {
#pragma HLS PIPELINE II=1
   ap_uint<128> shift_half=halfword?ap_uint<128>(shifts.range(255,128)):ap_uint<128>(shifts.range(127,0));
   ap_uint<128> gain_half=halfword?ap_uint<128>(gains.range(255,128)):ap_uint<128>(gains.range(127,0));
   for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
    unsigned position=halfword*4+lane;
    ap_uint<32> pair=0;
    pair.range(15,0)=c209mp::bits(c209mp::narrow(uf(shift_half.range(lane*32+31,lane*32))));
    pair.range(31,16)=c209mp::bits(c209mp::narrow(uf(gain_half.range(lane*32+31,lane*32))));
    coefficients[word*8+position]=pair;
   }
  }
 }
}

static void load_row_packed(bus_t *input,uint64_t address,ap_uint<13> columns,
 bool fp32,PrefetchRow &row) {
#pragma HLS INLINE off
 bus_t held=0;
 ap_uint<10> words=(ap_uint<14>(columns)+7u)>>3;
 C209_AUX_PREFETCH_PACK8:for(ap_uint<10> word=0;word<words;word++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1
  // FP32: one AXI256 beat becomes eight half values. F16: retain the AXI
  // beat for two iterations and select its low/high128 halves in order.
  if(fp32||!word[0])held=input[word_address(address,WordOffset(fp32?word:ap_uint<10>(word>>1)))];
  ap_uint<128> selected=word[0]?ap_uint<128>(held.range(255,128)):ap_uint<128>(held.range(127,0));
  ap_uint<128> packed=0;ap_uint<13> first=ap_uint<13>(word)<<3;
  for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
   if(first+lane<columns) {
    c209mp::activation value=fp32?c209mp::narrow(uf(held.range(32*lane+31,32*lane))):
      c209fixed::auxiliary_detail::input_half(ap_uint<16>(selected.range(16*lane+15,16*lane)));
    packed.range(16*lane+15,16*lane)=c209mp::bits(value);
   }
  }
  // Exactly one128-bit memory write per iteration. Unused patch68 lanes
  // are zero, and only ceil(columns/8) entries are made visible.
  row[word]=packed;
 }
}

static void adaln_load_statistics(bus_t *input,uint64_t address,PrefetchRow &row);

static void load_rows(bus_t *input,uint64_t address,ap_uint<14> rows,
 ap_uint<13> columns,ap_uint<15> stride,bool fp32,bool adaln,PrefetchRows &prefetched,RowReady &ready) {
#pragma HLS INLINE off
 ByteOffset offset=0;
 for(ap_uint<14> r=0;r<rows;r++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
  {
   // One acquire/commit site for both formats, one physical producer.
   hls::write_lock<PrefetchRow> row(prefetched);
   if(adaln)adaln_load_statistics(input,byte_address(address,offset),row);
   else load_row_packed(input,byte_address(address,offset),columns,fp32,row);
  }
  // This token keeps the threaded C model from reading its write-acquire
  // allocation early. HLS may schedule it before the helper call; hardware
  // readiness must therefore come from the PIPO empty_n/acquire handshake.
  ready.write(1);
  offset+=stride;
 }
}

static float copy_row(PrefetchRow &input,Row &row,ap_uint<13> columns) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
 ap_uint<15> maxima[4]={0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=maxima complete
 ap_uint<10> words=(ap_uint<14>(columns)+7u)>>3;
 C209_AUX_COPY_ROW8:for(ap_uint<10> word=0;word<words;word++) {
#pragma HLS PIPELINE II=1
  ap_uint<128> packed=input[word];ap_uint<13> first=ap_uint<13>(word)<<3;
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   unsigned i=first+lane;
   ap_uint<15> local=0;
   if(i<columns) {
    c209mp::activation value=c209mp::from_bits(c209mp::word(packed.range(16*lane+15,16*lane)));row[i]=value;
    local=aux_mixed::magnitude(value);
   }
   if(i+4<columns) {
    c209mp::activation value=c209mp::from_bits(c209mp::word(packed.range(64+16*lane+15,64+16*lane)));row[i+4]=value;
    ap_uint<15> mag=aux_mixed::magnitude(value);
    if(mag>local)local=mag;
   }
   if(local>maxima[lane])maxima[lane]=local;
  }
 }
 return aux_mixed::maximum4(maxima);
}

// Four lanes keep the original centered FP32 arithmetic and reduction order.
// Pass zero loads FP32 DDR, rounds to the existing half boundary, and computes
// mean while filling the owned PIPO slot. Pass one rereads that SAME slot.
// The producer can process row r+1 while the consumer emits row r; no new
// full-row array or independently blocking metadata FIFO is introduced.
static void adaln_load_statistics(bus_t *input,uint64_t address,PrefetchRow &row) {
#pragma HLS INLINE
 const uint64_t base=word_address(address,WordOffset(0));
 bus_t held=input[base];ap_uint<128> packed=0;
 const float first_value=c209mp::widen(c209mp::narrow(uf(held.range(31,0))));
 float origin=first_value,mean=0.f,variance=0.f;
 float ring[4][c204::OVERLAP_PHASES];
#pragma HLS ARRAY_PARTITION variable=ring complete dim=1
#pragma HLS BIND_STORAGE variable=ring type=ram_2p impl=lutram
 C209_OVERLAP_STAT_PASSES:for(unsigned pass=0;pass<2;pass++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
 C209_OVERLAP_STAT_SCAN4:for(unsigned halfword=0;halfword<512;halfword++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=ring type=inter direction=RAW distance=16 true
// Within this INNER invocation pass is immutable: pass0 writes only,
// pass1 reads only. The outer non-pipelined pass boundary still waits for
// all stores and the reduction before starting the next invocation.
#pragma HLS DEPENDENCE variable=row type=inter direction=RAW false
  if(!(halfword&1u)) {
   if(pass==0){if(halfword!=0)held=input[base+(halfword>>1)];}
   else packed=row[halfword>>1];
  }
  ap_uint<128> raw=(halfword&1u)?ap_uint<128>(held.range(255,128)):ap_uint<128>(held.range(127,0));
  ap_uint<64> selected=(halfword&1u)?ap_uint<64>(packed.range(127,64)):ap_uint<64>(packed.range(63,0));
  ap_uint<64> converted=0;
  unsigned phase=halfword&(c204::OVERLAP_PHASES-1);
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   c209mp::activation value=pass==0?c209mp::narrow(uf(raw.range(lane*32+31,lane*32))):
     c209mp::from_bits(c209mp::word(selected.range(lane*16+15,lane*16)));
   converted.range(lane*16+15,lane*16)=c209mp::bits(value);
   float preceding=halfword<c204::OVERLAP_PHASES?0.f:ring[lane][phase];
   float delta=c209mp::widen(value)-origin;
   ring[lane][phase]=preceding+(pass==1?delta*delta:delta);
  }
  if(pass==0) {
   if(halfword&1u){packed.range(127,64)=converted;row[halfword>>1]=packed;}
   else packed.range(63,0)=converted;
  }
 }
  float result=c204::reduce_overlap_phases(ring)*(1.f/2048.f);
  if(pass==0){mean=first_value+result;origin=mean;}
  else variance=result;
 }
 if(variance<0.f)variance=0.f;
 float inverse_std=1.f/hls::sqrtf(variance+1e-6f);
 ap_uint<128> metadata=0;
 metadata.range(31,0)=fu(mean);metadata.range(63,32)=fu(inverse_std);
 row[ROW_META]=metadata;
}

static void adaln_emit_quant(PrefetchRow &row,const Coefficients &coefficients,float mean,
 float inverse_std,float inverse_scale,AuxQuantRequests &requests) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=coefficients cyclic factor=4
 // Retain centered FP32 variance and each original half arithmetic boundary.
 // The already-published SCL1 scale removes the post-modulation max/re-read.
 ap_uint<128> packed=0;
 C209_ADALN_DIRECT_QUANT4:for(unsigned first=0;first<2048;first+=4) {
#pragma HLS PIPELINE II=1
  if(!(first&4u))packed=row[first>>3];
  ap_uint<64> selected=(first&4u)?ap_uint<64>(packed.range(127,64)):ap_uint<64>(packed.range(63,0));
  ap_uint<64> halves=0;
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   unsigned i=first+lane;ap_uint<32> pair=coefficients[i];
   c209mp::activation value=c209mp::from_bits(c209mp::word(selected.range(lane*16+15,lane*16)));
   float normalized_wide=(c209mp::widen(value)-mean)*inverse_std;
   c209mp::activation normalized=c209mp::narrow(normalized_wide);
   c209mp::activation shift=c209mp::from_bits(c209mp::word(pair.range(15,0)));
   c209mp::activation gain=c209mp::from_bits(c209mp::word(pair.range(31,16)));
   c209mp::activation rounded=aux_mixed::add(aux_mixed::multiply(normalized,gain),shift);
   halves.range(lane*16+15,lane*16)=c209mp::bits(rounded);
  }
  requests.write(aux_quant_request(halves,inverse_scale));
 }
}

static void compute_rows(ap_uint<14> rows,ap_uint<13> columns,bool adaln,
 Row &row,const Coefficients &coefficients,PrefetchRows &prefetched,
 RowReady &ready,Scales &scales,AuxQuantRequests &requests,float fixed_scale,float fixed_inverse) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=coefficients cyclic factor=4
 for(ap_uint<14> r=0;r<rows;r++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
  ready.read();
  // The store must receive its scale before bounded quant responses arrive.
  if(adaln)scales.write(fixed_scale);
  float maximum=0.f;
  {
   // A single acquire site owns data and statistics until the last emitted
   // request is accepted. Backpressure cannot recycle or change this row.
   hls::read_lock<PrefetchRow> input(prefetched);
   if(adaln) {
    ap_uint<128> metadata=input[ROW_META];
    float mean=uf(metadata.range(31,0)),inverse_std=uf(metadata.range(63,32));
    adaln_emit_quant(input,coefficients,mean,inverse_std,fixed_inverse,requests);
   } else maximum=copy_row(input,row,columns);
  }
  if(!adaln) {
   float scale=maximum==0.f?1.f:maximum/32760.f;
   scales.write(scale);
   aux_quant_row_requests(row,columns,1.f/scale,requests);
  }
 }
}

static void store_rows(bus_t *output,bus_t *aux,uint64_t address,uint64_t scale_address,
 ap_uint<14> rows,ap_uint<13> columns,ap_uint<13> stride,ap_uint<14> byte_stride,
 Scales &scales,AuxQuantResults &results) {
#pragma HLS INLINE off
 ByteOffset offset=0;
 for(ap_uint<14> r=0;r<rows;r++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
  float scale=scales.read();writef(aux,scale_address,r,scale);
  aux_quant_row_store(output,byte_address(address,offset),columns,stride,results);
  offset+=byte_stride;
 }
}

static void pipeline(bus_t *input,bus_t *output,bus_t *aux,const FixedJob &j,
 Row &workspace,const Coefficients &coefficients,ap_uint<13> columns,
 ap_uint<13> output_stride,ap_uint<15> input_byte_stride,ap_uint<14> output_byte_stride,
 bool adaln,bool input_fp32,AuxQuantRequests &requests,AuxQuantResults &results,
 float fixed_scale,float fixed_inverse) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=workspace cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=coefficients cyclic factor=4
 PrefetchRows prefetched;Scales scales;RowReady ready;
#pragma HLS STREAM variable=scales depth=2
#pragma HLS STREAM variable=ready depth=2
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(load_rows(input,j.a,j.m,columns,input_byte_stride,input_fp32,adaln,prefetched,ready));
 C209_FIXED_TASK(compute_rows(j.m,columns,adaln,workspace,coefficients,prefetched,ready,scales,requests,fixed_scale,fixed_inverse));
 C209_FIXED_TASK(store_rows(output,aux,j.c,adaln?j.s3:j.s0,j.m,columns,output_stride,output_byte_stride,scales,results));
}
} }
