#pragma once
#include "c209_island_services.hpp"
#include "c209_attention_pump.hpp"
#include "../src/c209_shared_mac_rtl.hpp"
#ifndef __SYNTHESIS__
#include <vector>
#include <stdexcept>
#endif

// Fixed-profile logical owners share one external physical MAC service.
// A owns GEMM/QK/softmax; B owns PV/update and has no GEMM datapath.
namespace c209fixed {
using c209island::stream64;using c209island::stream128;using c209island::stream256;
using command_word=ap_uint<64>;
using command_stream=hls::stream<command_word>;
enum ComputeOp { COMPUTE_STOP=0, COMPUTE_GEMM=1, COMPUTE_ATTN=2 };

// Private, narrow owner ABI: op2, N32, partial, K/stored13, zeros13, factor32.
// Bit62 selects the true M128 GEMM tail; M is128/256 for GEMM and64 for attention. The validated DDR adapter owns
// global head/query/output addresses; none of those widen the owner boundary.
static command_word gemm_command(unsigned local_n,unsigned inner,bool partial,unsigned rows=256) {
 command_word p=COMPUTE_GEMM;p[2]=local_n==32;p[3]=partial;p.range(16,4)=inner;p[62]=rows==128u;return p;
}
static command_word attention_command(unsigned stored,unsigned zeros,float factor) {
 command_word p=COMPUTE_ATTN;p.range(16,4)=stored;p.range(29,17)=zeros;
 p.range(61,30)=c209island::to_bits(factor);return p;
}
static c209island::Job decode_command(command_word p) {
 c209island::Job j={};j.op=p.range(1,0);j.inner=p.range(16,4);
 j.rows=j.op==COMPUTE_ATTN?64:(p[62]?128:256);j.columns=p[2]?32:128;
 j.flags=p[3]?c2::LOCAL_PARTIAL:0;j.zero_count=p.range(29,17);
 j.factor=c209island::from_bits(p.range(61,30));return j;
}
static unsigned attention_blocks(const c209island::Job &j) {return ((j.inner+127)>>7)<<1;}
static c207::Meta attention_meta(const c209island::Job &j,unsigned index) {
 unsigned ctx=index&1u,kb=(index>>1)<<7;
 c207::Meta m={0,ctx*32,kb,32,minv(128u,j.inner-kb),ctx,kb==0,kb+128>=j.inner};return m;
}

#ifndef __SYNTHESIS__
// Native numerical/data-order tests need actual concurrent C processes.
// They are not a substitute for the finite FIFO/AXI generated RTL tests.
struct NativeGroup {
 std::vector<std::thread> tasks;
 template<class F> void run(F body) {
  hls::stream_globals::incr_task_counter();
  tasks.emplace_back([body]{c207::NativeTaskGuard guard;body();});
 }
 ~NativeGroup(){for(auto &t:tasks)t.join();}
};
#define C209_FIXED_GROUP c209fixed::NativeGroup c209_native_group
#define C209_FIXED_TASK(call) c209_native_group.run([&]{call;})
#else
#define C209_FIXED_GROUP
#define C209_FIXED_TASK(call) call
#endif

static void read_scale_block(stream64 &input,c209island::ScaleBlocks &saved) {
#pragma HLS INLINE off
 hls::write_lock<c209island::ScaleBlock> slot(saved);
 for(unsigned w=0;w<20;w++) {
  c2::bus_t packed=0;
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS PIPELINE II=1
   packed.range(lane*64+63,lane*64)=input.read();
  }
  slot[w]=packed;
 }
}
static void fixed_qk_issue(const c209island::Job &j,unsigned count,stream64 &scales,
 c209island::ScaleBlocks &saved,c207::MetaStream &issued,cmd_stream &cmd) {
#pragma HLS INLINE off
 // Deliberate block transaction boundary: pending cmd must not depend on a
 // future block's scale TVALID. Operand movers are separate DATAFLOW tasks.
 for(unsigned block=0;block<count;block++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
  read_scale_block(scales,saved);
  c207::Meta m=attention_meta(j,block);issued.write(m);
  MacCommand d={32,128,128,true,true,false};cmd.write(d);
 }
}
struct AttentionControl {
 ap_uint<7> count;ap_uint<15> q_words;ap_uint<17> k_words;float factor;
};
static AttentionControl prepare_attention_control(const c209island::Job &j) {
#pragma HLS INLINE off
#pragma HLS LATENCY min=1
 // A non-DATAFLOW call boundary prevents automatic entry processes from
 // chaining count -> Q length -> K length in the same200MHz timing path.
#ifndef __SYNTHESIS__
 assert(j.inner>0&&j.inner<=4096);
#endif
 ap_uint<14> rounded=ap_uint<13>(j.inner);rounded+=127;
 AttentionControl control;control.count=ap_uint<7>(rounded>>7)<<1;
 control.q_words=ap_uint<15>(control.count)<<8;
 control.k_words=ap_uint<17>(control.count)<<10;
 control.factor=j.factor==0?0.08838834764831845f:j.factor;
 return control;
}
static void publish_attention_pump_lengths(unsigned q_words,unsigned k_words,
 hls::stream<ap_uint<32>>& q_lengths,hls::stream<ap_uint<32>>& k_lengths) {
#pragma HLS INLINE off
 q_lengths.write(ap_uint<32>(q_words));k_lengths.write(ap_uint<32>(k_words));
}
static void fixed_attention_a_dataflow(const c209island::Job &j,unsigned count,
 unsigned q_words,unsigned k_words,unsigned zero_count,float factor,stream64 &scales,
 stream256 &q,stream256 &k,stream64 &pmeta,stream64 &probability,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 c209island::ScaleBlocks saved;c207::ScoreBlocks scores;c207::ProbBlocks probs;
 c207::MetaStream issued,sm,pm;
 hls::stream<ap_uint<32>> q_lengths,k_lengths;
#pragma HLS STREAM variable=q_lengths depth=2
#pragma HLS STREAM variable=k_lengths depth=2
#pragma HLS STREAM variable=issued depth=2
#pragma HLS STREAM variable=sm depth=2
#pragma HLS STREAM variable=pm depth=2
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(fixed_qk_issue(j,count,scales,saved,issued,cmd));
 // Two independent, normally bound RTL leaves. A pending final beat drains
 // without the next context's input valid; no FIFO depth or math is changed.
 C209_FIXED_TASK(publish_attention_pump_lengths(q_words,k_words,q_lengths,k_lengths));
 C209_FIXED_TASK(c209_attention_q_pump(q_lengths,q,ai));
 C209_FIXED_TASK(c209_attention_k_pump(k_lengths,k,bi));
 C209_FIXED_TASK(c209island::qk_collect(count,saved,issued,co,scores,sm));
 C209_FIXED_TASK(c207::softmax_pairs(count,zero_count,factor,scores,sm,probs,pm));
 C209_FIXED_TASK(c209island::send_probability(j,probs,pm,pmeta,probability));
}
static void fixed_attention_a(const c209island::Job &j,stream64 &scales,
 stream256 &q,stream256 &k,stream64 &pmeta,stream64 &probability,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 AttentionControl control=prepare_attention_control(j);
 unsigned count=control.count.to_uint(),q_words=control.q_words.to_uint(),k_words=control.k_words.to_uint();
 unsigned zero_count=j.zero_count;float factor=control.factor;
 fixed_attention_a_dataflow(j,count,q_words,k_words,zero_count,factor,scales,q,k,pmeta,probability,cmd,ai,bi,co);
}
static void fixed_values(const c209island::Job &j,stream64 &scales,
 stream256 &values,c207::C209VBlocks &output,hls::stream<bool> &ready) {
#pragma HLS INLINE off
 for(unsigned block=0;block<attention_blocks(j);block++) {
  {
   hls::write_lock<c207::C209VBlock> slot(output);
   for(unsigned w=0;w<16;w++) {
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
  ready.write(true); // explicit commit, also protects native PIPO publication
 }
}
static void join_probability_values(unsigned count,c207::MetaStream &input,
 hls::stream<bool> &ready,c207::MetaStream &output) {
#pragma HLS INLINE off
 for(unsigned i=0;i<count;i++){c207::Meta m=input.read();ready.read();output.write(m);}
}
static void fixed_attention_b(const c209island::Job &j,stream64 &pmeta,stream64 &probability,
 stream64 &scales,stream256 &values,stream128 &output,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 c207::ProbBlocks probs;c207::C209VBlocks v;c207::MetaStream pm,joined,um;
 c207::C209UpdateStream updates;hls::stream<bool> ready;
#pragma HLS STREAM variable=pm depth=2
#pragma HLS STREAM variable=joined depth=2
#pragma HLS STREAM variable=um depth=2
#pragma HLS STREAM variable=ready depth=2
#pragma HLS STREAM variable=updates depth=32
#pragma HLS BIND_STORAGE variable=updates type=fifo impl=srl
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(c209island::receive_probability(j,pmeta,probability,probs,pm));
 C209_FIXED_TASK(fixed_values(j,scales,values,v,ready));
 C209_FIXED_TASK(join_probability_values(attention_blocks(j),pm,ready,joined));
 C209_FIXED_TASK(c207::c209_pv(attention_blocks(j),probs,joined,v,updates,um,cmd,ai,bi,co));
 C209_FIXED_TASK(c207::c209_update_stream(attention_blocks(j),updates,um,output));
}

// Exactly two scalar lanes per owner, one shared mathematical body. Bias is
// compile-time absent; partial selects only the existing final rounding point.
static void fixed_linear_pairs(unsigned rows,unsigned n,bool partial,result_stream &co,
 const float row_scale[256],const float column_scale[128],c2fast::linear_pair_stream &out) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=2
 unsigned r=0,col=0,pair=0;ap_uint<768> accum=0;
 for(unsigned word=0;word<rows*(n/2);word++) {
#pragma HLS PIPELINE II=1
  if(pair==0)accum=co.read();ap_uint<96> selected=0;
  switch(pair) {
   case 0:selected=accum.range(95,0);break;case 1:selected=accum.range(191,96);break;
   case 2:selected=accum.range(287,192);break;case 3:selected=accum.range(383,288);break;
   case 4:selected=accum.range(479,384);break;case 5:selected=accum.range(575,480);break;
   case 6:selected=accum.range(671,576);break;default:selected=accum.range(767,672);break;
  }
  ap_uint<64> pack=0;
  for(unsigned lane=0;lane<2;lane++) {
#pragma HLS UNROLL
   c2::i48 value=selected.range(lane*48+47,lane*48);
   float f=c2fast::linear_epilogue_scalar(value,row_scale[r],column_scale[col+lane],0.f,partial?c2::LOCAL_PARTIAL:0);
   pack.range(lane*32+31,lane*32)=fu(f);
  }
  out.write(pack);pair=(pair+1)&7u;col+=2;if(col==n){col=0;r++;}
 }
}
static void fixed_gemm_compute(const c209island::Job &j,stream256 &left,stream256 &right,
 const float row_scale[256],const float column_scale[128],stream128 &output,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 c2fast::linear_pair_stream pairs;
#pragma HLS STREAM variable=pairs depth=8
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(c209island::gemm_issue(j,left,right,cmd,ai,bi));
 C209_FIXED_TASK(fixed_linear_pairs(j.rows,j.columns,j.flags==c2::LOCAL_PARTIAL,co,row_scale,column_scale,pairs));
 C209_FIXED_TASK(c209island::encode_pairs(j,pairs,output));
}
static void fixed_gemm_job(const c209island::Job &j,stream64 &metadata,stream256 &left,
 stream256 &right,stream128 &output,cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 float row_scale[256],column_scale[128];
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=2
 c209island::read_float_words(metadata,row_scale,j.rows);
 c209island::read_float_words(metadata,column_scale,j.columns);
 fixed_gemm_compute(j,left,right,row_scale,column_scale,output,cmd,ai,bi,co);
}
// Two bounded metadata slots let the next GEMM publish its first physical
// command while the previous result epilogue drains. No second C matrix or
// arithmetic lane is added. ATTN/STOP markers are acknowledged only after
// every older GEMM result has reached the output stream.
using GemmScaleBlock=float[384];
using GemmScaleBlocks=hls::stream_of_blocks<GemmScaleBlock,2>;
static void fixed_gemm_epilogue(const c209island::Job &j,result_stream &co,
 const float row_scale[256],const float column_scale[128],stream128 &output) {
#pragma HLS INLINE off
 c2fast::linear_pair_stream pairs;
#pragma HLS STREAM variable=pairs depth=8
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(fixed_linear_pairs(j.rows,j.columns,j.flags==c2::LOCAL_PARTIAL,co,row_scale,column_scale,pairs));
 C209_FIXED_TASK(c209island::encode_pairs(j,pairs,output));
}
static void fixed_copy_gemm_scales(const c209island::Job &j,GemmScaleBlocks &scales,
 float row_scale[256],float column_scale[128]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=2
 // The lock is owned by this finite copy call, not the whole result worker.
 // Its release must precede this function's completion. The parent cannot
 // enter the epilogue until the private row/column arrays are fully produced.
 hls::read_lock<GemmScaleBlock> slot(scales);
 for(unsigned r=0;r<j.rows;r++) {
#pragma HLS PIPELINE II=1
  row_scale[r]=slot[r];
 }
 for(unsigned c=0;c<j.columns;c++) {
#pragma HLS PIPELINE II=1
  column_scale[c]=slot[256+c];
 }
}
static void fixed_a_results(command_stream &pending,GemmScaleBlocks &scales,
 hls::stream<bool> &drained,result_stream &co,result_stream &attention_co,stream128 &output) {
#pragma HLS INLINE off
 bool stop=false;
 while(!stop) {
  c209island::Job j=decode_command(pending.read());stop=j.op==COMPUTE_STOP;
  if(j.op==COMPUTE_GEMM) {
   float row_scale[256],column_scale[128];
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=2
   fixed_copy_gemm_scales(j,scales,row_scale,column_scale);
   // The private epilogue arrays now own these values. Release the PIPO
   // slot BEFORE waiting for co/output, otherwise a two-buffer PIPO prevents
   // publication of the next GEMM until this entire epilogue has completed.
   fixed_gemm_epilogue(j,co,row_scale,column_scale,output);
  } else {
   drained.write(true);
   // Exactly one process reads the physical result FIFO. A bounded relay
   // hands QK packets to the original attention path after the GEMM barrier.
   if(j.op==COMPUTE_ATTN)for(unsigned packet=0;packet<attention_blocks(j)*256u;packet++) {
#pragma HLS PIPELINE II=1
    attention_co.write(co.read());
   }
  }
 }
}
static void fixed_publish_barrier(command_word word,command_stream &pending) {
#pragma HLS INLINE off
#pragma HLS LATENCY min=1
 pending.write(word);
}
static void fixed_wait_barrier(hls::stream<bool> &drained) {
#pragma HLS INLINE off
#pragma HLS LATENCY min=1
 drained.read();
}
static void fixed_a_issue(command_stream &jobs,stream64 &metadata,stream256 &left,
 stream256 &right,stream64 &pmeta,stream64 &probability,command_stream &pending,
 GemmScaleBlocks &scales,hls::stream<bool> &drained,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &attention_co) {
#pragma HLS INLINE off
 bool stop=false;
 while(!stop) {
  command_word word=jobs.read();c209island::Job j=decode_command(word);stop=j.op==COMPUTE_STOP;
  if(j.op==COMPUTE_GEMM) {
   {
    hls::write_lock<GemmScaleBlock> slot(scales);
    for(unsigned r=0;r<j.rows;r+=2) {
#pragma HLS PIPELINE II=1
     ap_uint<64> pair=metadata.read();slot[r]=c209island::from_bits(pair.range(31,0));slot[r+1]=c209island::from_bits(pair.range(63,32));
    }
    for(unsigned c=0;c<j.columns;c+=2) {
#pragma HLS PIPELINE II=1
     ap_uint<64> pair=metadata.read();slot[256+c]=c209island::from_bits(pair.range(31,0));slot[257+c]=c209island::from_bits(pair.range(63,32));
    }
   }
   pending.write(word);
   c209island::gemm_issue(j,left,right,cmd,ai,bi);
  } else {
   // Do not merge these stream operations into one HLS FSM state: publishing
   // a marker conditional on the acknowledgement it creates is an empty-ring
   // deadlock. Separate non-inline calls require publication to complete
   // before the acknowledgement wait is launched (checked in generated RTL).
   fixed_publish_barrier(word,pending);
   fixed_wait_barrier(drained);
   if(j.op==COMPUTE_ATTN)fixed_attention_a(j,metadata,left,right,pmeta,probability,cmd,ai,bi,attention_co);
  }
 }
 MacCommand end={0,0,0,false,false,true};cmd.write(end);
}
static void fixed_worker_a_overlapped(command_stream &jobs,stream64 &metadata,
 stream256 &left,stream256 &right,stream64 &pmeta,stream64 &probability,stream128 &output,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 command_stream pending;GemmScaleBlocks scales;hls::stream<bool> drained;result_stream attention_co;
#pragma HLS STREAM variable=pending depth=2
#pragma HLS STREAM variable=drained depth=1
#pragma HLS STREAM variable=attention_co depth=2
#pragma HLS BIND_STORAGE variable=attention_co type=fifo impl=srl
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(fixed_a_issue(jobs,metadata,left,right,pmeta,probability,pending,scales,drained,cmd,ai,bi,attention_co));
 C209_FIXED_TASK(fixed_a_results(pending,scales,drained,co,attention_co,output));
}
template<unsigned ENGINE> static void fixed_worker(command_stream &jobs,stream64 &metadata,
 stream256 &left,stream256 &right,stream64 &pmeta,stream64 &probability,stream128 &output,
 cmd_stream &cmd,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 bool stop=false;
 while(!stop) {
  c209island::Job j=decode_command(jobs.read());stop=j.op==COMPUTE_STOP;
  if(j.op==COMPUTE_GEMM) {
   // Compile-time specialization removes the unused second GEMM epilogue.
   // The fixed DDR dispatcher sends GEMM only to A.
   if(ENGINE==0)fixed_gemm_job(j,metadata,left,right,output,cmd,ai,bi,co);
#ifndef __SYNTHESIS__
   else throw std::logic_error("single256 GEMM was sent to PV-only client B");
#endif
  }
  else if(j.op==COMPUTE_ATTN) {
   if(ENGINE==0)fixed_attention_a(j,metadata,left,right,pmeta,probability,cmd,ai,bi,co);
   else fixed_attention_b(j,pmeta,probability,metadata,right,output,cmd,ai,bi,co);
  }
 }
 MacCommand end={0,0,0,false,false,true};cmd.write(end);
}
template<unsigned ENGINE> static void fixed_island(command_stream &jobs,stream64 &metadata,
 stream256 &left,stream256 &right,stream64 &pmeta,stream64 &probability,stream128 &output,
 hls::stream<ap_uint<99>> &packed,input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 cmd_stream cmd;
#pragma HLS STREAM variable=cmd depth=8
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 if(ENGINE==0) {
  C209_FIXED_TASK(fixed_worker_a_overlapped(jobs,metadata,left,right,pmeta,probability,output,cmd,ai,bi,co));
 } else {
  C209_FIXED_TASK(fixed_worker<ENGINE>(jobs,metadata,left,right,pmeta,probability,output,cmd,ai,bi,co));
 }
 // Packing is independent of operand/result tasks, as in the proven0800
 // flow. It must not wait for data before publishing a core command.
 C209_FIXED_TASK(c204::c207_pack_mac_commands(cmd,packed));
}
static void island_a(command_stream &jobs,stream64 &metadata,stream256 &left,stream256 &right,
 stream64 &pmeta,stream64 &probability,stream128 &output,hls::stream<ap_uint<99>> &packed,
 input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 fixed_island<0>(jobs,metadata,left,right,pmeta,probability,output,packed,ai,bi,co);
}
static void island_b(command_stream &jobs,stream64 &metadata,stream256 &left,stream256 &right,
 stream64 &pmeta,stream64 &probability,stream128 &output,hls::stream<ap_uint<99>> &packed,
 input_stream &ai,input_stream &bi,result_stream &co) {
#pragma HLS INLINE off
 fixed_island<1>(jobs,metadata,left,right,pmeta,probability,output,packed,ai,bi,co);
}
static void single_metrics(hls::stream<ap_uint<256>> &snapshot,
 unsigned &elapsed,unsigned &owner_a,unsigned &owner_b,unsigned &source_empty,
 unsigned &result_stall,unsigned &a_commits,unsigned &b_commits,unsigned &no_request) {
#pragma HLS INLINE off
 ap_uint<256> word=snapshot.read();
 elapsed=word.range(31,0);owner_a=word.range(63,32);owner_b=word.range(95,64);
 source_empty=word.range(127,96);result_stall=word.range(159,128);
 a_commits=word.range(191,160);b_commits=word.range(223,192);no_request=word.range(255,224);
}
static void stop_islands(command_stream &a,command_stream &b){a.write(command_word(0));b.write(command_word(0));}
}
