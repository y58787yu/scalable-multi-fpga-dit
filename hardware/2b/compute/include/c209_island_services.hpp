#pragma once
// Current stream-only island services, shared by the small probe and future
// system adapters. This is still the G3 contract: GELU remains rejected and
// B retains its temporary 16 KiB final-store adapter. No production top is
// instantiated here. Keep the established math/MAC dependency order.
#include <cassert>
#include "c209_compute_support.hpp"
#include "../src/c207_mac_stream.hpp"
#include "../src/shared_gelu.hpp"
#include "../src/fast_gemm.hpp"
#include "../src/fast_vector.hpp"
#ifdef COSMOS_C205
#include "../src/c205_mac.hpp"
#else
#include "../src/c204_mac.hpp"
#endif
#include "../src/c204_attention.hpp"
#include "c209_island_contract.hpp"

namespace c209island {
using ScaleBlock=c2::bus_t[(c207::QR+128)/8];
using ScaleBlocks=hls::stream_of_blocks<ScaleBlock,2>;

static void dispatch(stream64 &input,stream64 &a,stream64 &b,unsigned &error) {
#pragma HLS INLINE off
 error=c2::OK;
 bool stop=false;
 while(!stop) {
  word64 x=input.read(),y=input.read(),z=input.read();Job j=decode(x,y,z);
  unsigned status=validate(j);
  if(status!=c2::OK){if(error==c2::OK)error=status;continue;}
  a.write(x);b.write(x);a.write(y);b.write(y);a.write(z);b.write(z);
  stop=j.op==STOP;
 }
}
static c207::Meta block_meta(const Job &j,unsigned index) {
 unsigned ctx=index%contexts(j),kb=(index/contexts(j))*128;
 c207::Meta m={0,ctx*32,kb,minv(32u,j.rows-ctx*32),minv(128u,j.inner-kb),ctx,kb==0,kb+128>=j.inner};
 return m;
}
static void read_float_words(stream64 &input,float *values,unsigned count) {
#pragma HLS INLINE off
 for(unsigned p=0;p<count;p+=2) {
#pragma HLS PIPELINE II=1
  word64 word=input.read();values[p]=from_bits(word.range(31,0));
  if(p+1<count)values[p+1]=from_bits(word.range(63,32));
 }
}
static void copy_words(stream256 &in,input_stream &out,unsigned count) {
#pragma HLS INLINE off
 for(unsigned p=0;p<count;p++) {
#pragma HLS PIPELINE II=1
  out.write(in.read());
 }
}

// Inputs are the existing panel32/scalarK/half16 MAC ABI; no new transpose or
// quantization arithmetic is hidden in the probe ingress.
static void qk_issue(const Job &j,stream64 &scales,
 ScaleBlocks &saved,c207::MetaStream &metadata,cmd_stream &cmd) {
#pragma HLS INLINE off
 for(unsigned block=0;block<blocks(j);block++) {
  c207::Meta m=block_meta(j,block);
  {
   hls::write_lock<ScaleBlock> slot(saved);
   for(unsigned w=0;w<(c207::QR+128)/8;w++) {
    c2::bus_t packed=0;
    for(unsigned lane=0;lane<4;lane++) {
#pragma HLS PIPELINE II=1
     packed.range(lane*64+63,lane*64)=scales.read();
    }
    slot[w]=packed;
   }
  }
  metadata.write(m);
  MacCommand command={m.mm,128,128,true,true,false};cmd.write(command);
 }
}
static void qk_collect(unsigned count,ScaleBlocks &scales,c207::MetaStream &metadata,
 result_stream &co,c207::ScoreBlocks &scores,c207::MetaStream &sm) {
#pragma HLS INLINE off
 for(unsigned block=0;block<count;block++) {
  c207::Meta m=metadata.read();hls::read_lock<ScaleBlock> slot(scales);
  c207::c209_qk_receive(m,slot,co,scores,sm);
 }
}
static void send_probability(const Job &j,c207::ProbBlocks &input,c207::MetaStream &metadata,
 stream64 &headers,stream64 &data) {
#pragma HLS INLINE off
 for(unsigned block=0;block<blocks(j);block++) {
  c207::Meta m=metadata.read();hls::read_lock<c207::ProbBlock> slot(input);
  word64 word=0;word.range(63,32)=j.tag;word.range(31,0)=block;headers.write(word);
  word.range(31,0)=m.h;word.range(63,32)=m.rb;headers.write(word);
  word.range(31,0)=m.kb;word.range(63,32)=m.mm;headers.write(word);
  word.range(31,0)=m.nn;word.range(63,32)=m.context;headers.write(word);
  word=0;word[0]=m.first;word[1]=m.last;headers.write(word);
  for(unsigned p=0;p<2*c207::QR;p+=2) {
#pragma HLS PIPELINE II=1
   word=0;
   for(unsigned lane=0;lane<2;lane++) {
#pragma HLS UNROLL
    unsigned index=p+lane;
    if(index%c207::QR<m.mm)word.range(lane*32+31,lane*32)=to_bits(c207::get_float(slot[c207::PROB_SCALE+index/8],index%8));
   }
   headers.write(word);
  }
  for(unsigned r=0;r<m.mm;r++)for(unsigned w=0;w<32;w++) {
#pragma HLS PIPELINE II=1
   c2::bus_t packed=slot[r*8+w/4];data.write(packed.range((w%4)*64+63,(w%4)*64));
  }
 }
}
static void attention_a(const Job &j,stream64 &scales,stream256 &q,stream256 &k,
 stream64 &pmeta,stream64 &probability,cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 ScaleBlocks saved;c207::ScoreBlocks scores;c207::ProbBlocks probs;
 c207::MetaStream issued,sm,pm;
#pragma HLS STREAM variable=issued depth=2
#pragma HLS STREAM variable=sm depth=2
#pragma HLS STREAM variable=pm depth=2
 // Different stream writes have no implicit scheduling order in HLS. Keep
 // command production independent of operand copies: joining copy completion
 // before cmd.write deadlocks finite ai/bi FIFOs while the MAC waits for cmd.
 // Each operand stream retains its original per-block panel order and count.
 qk_issue(j,scales,saved,issued,cmd);
 copy_words(q,ai,blocks(j)*256);
 copy_words(k,bi,blocks(j)*1024);
 qk_collect(blocks(j),saved,issued,co,scores,sm);
 c207::softmax_pairs(blocks(j),j.zero_count,j.factor==0?0.08838834764831845f:j.factor,scores,sm,probs,pm);
 send_probability(j,probs,pm,pmeta,probability);
}
static void receive_probability(const Job &j,stream64 &headers,stream64 &data,
 c207::ProbBlocks &output,c207::MetaStream &metadata) {
#pragma HLS INLINE off
 for(unsigned block=0;block<blocks(j);block++) {
  // The only producer is island A. Preserve its complete block/context token.
  word64 identity=headers.read(),x=headers.read(),y=headers.read(),z=headers.read(),flags=headers.read();
  c207::Meta m={x.range(31,0).to_uint(),x.range(63,32).to_uint(),y.range(31,0).to_uint(),
   y.range(63,32).to_uint(),z.range(31,0).to_uint(),z.range(63,32).to_uint(),bool(flags[0]),bool(flags[1])};
#ifndef __SYNTHESIS__
  assert(identity.range(63,32)==j.tag&&identity.range(31,0)==block);
  c207::Meta expected=block_meta(j,block);
  assert(m.h==expected.h&&m.rb==expected.rb&&m.kb==expected.kb&&m.mm==expected.mm&&m.nn==expected.nn&&m.context==expected.context&&m.first==expected.first&&m.last==expected.last);
#endif
  {
   hls::write_lock<c207::ProbBlock> slot(output);
   for(unsigned w=0;w<2*c207::QR/8;w++) {
    c2::bus_t packed=0;
    for(unsigned lane=0;lane<4;lane++) {
#pragma HLS PIPELINE II=1
     packed.range(lane*64+63,lane*64)=headers.read();
    }
    slot[c207::PROB_SCALE+w]=packed;
   }
   for(unsigned r=0;r<m.mm;r++)for(unsigned w=0;w<8;w++) {
    c2::bus_t packed=0;
    for(unsigned lane=0;lane<4;lane++) {
#pragma HLS PIPELINE II=1
     packed.range(lane*64+63,lane*64)=data.read();
    }
    slot[r*8+w]=packed;
   }
  }
  metadata.write(m); // block commit precedes consumer acquisition
 }
}
static void receive_values(const Job &j,stream64 &scales,stream256 &values,c207::C209VBlocks &output) {
#pragma HLS INLINE off
 for(unsigned block=0;block<blocks(j);block++) {
  hls::write_lock<c207::C209VBlock> slot(output);
  for(unsigned w=0;w<128/8;w++) {
   c2::bus_t packed=0;
   for(unsigned lane=0;lane<4;lane++) {
#pragma HLS PIPELINE II=1
    packed.range(lane*64+63,lane*64)=scales.read();
   }
   slot[c207::VW+w]=packed;
  }
  for(unsigned w=0;w<c207::VW;w++) {
#pragma HLS PIPELINE II=1
   slot[w]=values.read();
  }
 }
}
static void attention_b_compute(const Job &j,stream64 &pmeta,stream64 &probability,
 stream64 &vscale,stream256 &values,c2::bus_t output[512],
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 c2::Job math={};math.c=0;math.lc=128;math.fc=c2::F16;
 c207::ProbBlocks probs;c207::C209VBlocks v;c207::MetaStream pm,um;c207::C209UpdateStream updates;
#pragma HLS STREAM variable=pm depth=2
#pragma HLS STREAM variable=um depth=2
#pragma HLS STREAM variable=updates depth=32
#pragma HLS BIND_STORAGE variable=updates type=fifo impl=srl
#pragma HLS DATAFLOW
 receive_probability(j,pmeta,probability,probs,pm);
 receive_values(j,vscale,values,v);
 c207::c209_pv(blocks(j),probs,pm,v,updates,um,cmd,ai,bi,co);
 c207::c209_update(output,math,blocks(j),updates,um);
}
static void attention_b(const Job &j,stream64 &pmeta,stream64 &probability,
 stream64 &vscale,stream256 &values,stream128 &output,stream64 &events,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 // Temporary 16KiB, B-owned adapter for the exact production DDR-store updater.
 // Its end-of-job drain is NOT a proof of final continuous production throughput.
 c2::bus_t result[512];
#pragma HLS BIND_STORAGE variable=result type=ram_2p impl=bram
 attention_b_compute(j,pmeta,probability,vscale,values,result,cmd,ai,bi,co);
 events.write(event(j.tag,c2::F16,8,j.rows*16));
 for(unsigned w=0;w<j.rows*8;w++)for(unsigned half=0;half<2;half++) {
#pragma HLS PIPELINE II=1
  output.write(result[w].range(half*128+127,half*128));
 }
}
// A and B have independent backpressure. Serially emptying all A before B
// prevents the physical loader from filling both memories concurrently.
static void gemm_operands(stream256 &left,stream256 &right,input_stream &ai,
 input_stream &bi,unsigned a_words,unsigned b_words) {
#pragma HLS INLINE off
 unsigned a_sent=0,b_sent=0;
 while(a_sent<a_words||b_sent<b_words) {
#pragma HLS PIPELINE II=1
  if(a_sent<a_words&&!left.empty()&&!ai.full()) {ai.write(left.read());++a_sent;}
  if(b_sent<b_words&&!right.empty()&&!bi.full()) {bi.write(right.read());++b_sent;}
 }
}
static void gemm_issue(const Job &j,stream256 &left,stream256 &right,
 cmd_stream &cmd,input_stream &ai,input_stream &bi) {
#pragma HLS INLINE off
 unsigned np=padded_columns(j);
 for(unsigned kb=0;kb<j.inner;kb+=256) {
  unsigned kp=(minv(256u,j.inner-kb)+15)&~15u;
  MacCommand command={j.rows,np,kp,kb==0,false,false};cmd.write(command);
  gemm_operands(left,right,ai,bi,((j.rows+31)/32)*kp*2,(np/32)*kp*2);
 }
 MacCommand commit={j.rows,np,0,false,true,false};cmd.write(commit);
}
static void encode_pairs(const Job &j,c2fast::linear_pair_stream &input,stream128 &output) {
#pragma HLS INLINE off
 ap_uint<128> packed=0;
 for(unsigned pair=0;pair<j.rows*padded_columns(j)/2;pair++) {
#pragma HLS PIPELINE II=1
  ap_uint<64> value=input.read();
  if(j.flags&c2::LOCAL_PARTIAL) {
   if(pair&1){packed.range(127,64)=value;output.write(packed);}
   else packed.range(63,0)=value;
  } else {
   ap_uint<32> group=0;
   for(unsigned lane=0;lane<2;lane++) {
#pragma HLS UNROLL
    group.range(lane*16+15,lane*16)=
     c209mp::bits(c209mp::narrow(from_bits(value.range(lane*32+31,lane*32))));
   }
   switch(pair&3u) {
    case 0:packed.range(31,0)=group;break;
    case 1:packed.range(63,32)=group;break;
    case 2:packed.range(95,64)=group;break;
    default:packed.range(127,96)=group;output.write(packed);break;
   }
  }
 }
}
static void gemm_math(const Job &j,result_stream &co,const float row_scale[c2::MAC_ROWS],
 const float col_scale[c2::GEMM_N],const float bias[c2::GEMM_N],stream128 &output) {
#pragma HLS INLINE off
 c2::Job math={};math.flags=j.flags;
 FastGemmTileMeta tile={0,0,j.rows,j.columns,padded_columns(j),false};
 c2fast::linear_pair_stream pairs;
#pragma HLS STREAM variable=pairs depth=8
#pragma HLS DATAFLOW
 c2fast::stream_linear_pairs(math,tile,co,row_scale,col_scale,bias,pairs);
 encode_pairs(j,pairs,output);
}
static void gemm_compute(const Job &j,stream256 &left,stream256 &right,
 const float row_scale[c2::MAC_ROWS],const float col_scale[c2::GEMM_N],const float bias[c2::GEMM_N],
 stream128 &output,cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 gemm_issue(j,left,right,cmd,ai,bi);
 gemm_math(j,co,row_scale,col_scale,bias,output);
}
static void gemm_job(const Job &j,stream64 &metadata,stream256 &left,stream256 &right,
 stream128 &output,stream64 &events,cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 float row_scale[c2::MAC_ROWS],col_scale[c2::GEMM_N],bias[c2::GEMM_N];
#pragma HLS ARRAY_PARTITION variable=col_scale cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=2
 read_float_words(metadata,row_scale,j.rows);read_float_words(metadata,col_scale,j.columns);read_float_words(metadata,bias,j.columns);
 unsigned lanes=(j.flags&c2::LOCAL_PARTIAL)?4:8;
 events.write(event(j.tag,(j.flags&c2::LOCAL_PARTIAL)?c2::F32:c2::F16,lanes,j.rows*padded_columns(j)/lanes));
 gemm_compute(j,left,right,row_scale,col_scale,bias,output,cmd,ai,bi,co);
}
static void worker_a(stream64 &jobs,stream64 &metadata,stream256 &left,stream256 &right,
 stream64 &pmeta,stream64 &probability,stream128 &output,stream64 &events,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 bool stop=false;
 while(!stop) {
  Job j=read_job(jobs);stop=j.op==STOP;
  if(j.op==ATTN){attention_a(j,metadata,left,right,pmeta,probability,cmd,ai,bi,co);events.write(event(j.tag,c2::F16,8,0));}
  else if(j.op==GEMM)gemm_job(j,metadata,left,right,output,events,cmd,ai,bi,co);
 }
 MacCommand end={0,0,0,false,false,true};cmd.write(end);
}
static void worker_b(stream64 &jobs,stream64 &metadata,stream256 &left,stream256 &right,
 stream64 &pmeta,stream64 &probability,stream128 &output,stream64 &events,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 bool stop=false;
 while(!stop) {
  Job j=read_job(jobs);stop=j.op==STOP;
  if(j.op==ATTN)attention_b(j,pmeta,probability,metadata,right,output,events,cmd,ai,bi,co);
  else if(j.op==GEMM)gemm_job(j,metadata,left,right,output,events,cmd,ai,bi,co);
 }
 MacCommand end={0,0,0,false,false,true};cmd.write(end);
}
static void island_a(stream64 &jobs,stream64 &metadata,stream256 &left,stream256 &right,
 stream64 &pmeta,stream64 &probability,stream128 &output,stream64 &events) {
#pragma HLS INLINE off
 cmd_stream cmd;input_stream ai,bi;result_stream co;
#pragma HLS STREAM variable=cmd depth=8
#pragma HLS STREAM variable=ai depth=64
#pragma HLS STREAM variable=bi depth=64
#pragma HLS STREAM variable=co depth=32
#pragma HLS BIND_STORAGE variable=co type=fifo impl=srl
#pragma HLS DATAFLOW
 worker_a(jobs,metadata,left,right,pmeta,probability,output,events,cmd,ai,bi,co);
 c204::c209_mac_service<0>(cmd,ai,bi,co);
}
static void island_b(stream64 &jobs,stream64 &metadata,stream256 &left,stream256 &right,
 stream64 &pmeta,stream64 &probability,stream128 &output,stream64 &events) {
#pragma HLS INLINE off
 cmd_stream cmd;input_stream ai,bi;result_stream co;
#pragma HLS STREAM variable=cmd depth=8
#pragma HLS STREAM variable=ai depth=64
#pragma HLS STREAM variable=bi depth=64
#pragma HLS STREAM variable=co depth=32
#pragma HLS BIND_STORAGE variable=co type=fifo impl=srl
#pragma HLS DATAFLOW
 worker_b(jobs,metadata,left,right,pmeta,probability,output,events,cmd,ai,bi,co);
 c204::c209_mac_service<1>(cmd,ai,bi,co);
}
}

