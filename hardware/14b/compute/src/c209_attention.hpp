#pragma once
// Included inside namespace c207. Two independent MAC owners; no QK/PV
// request/response cycle through a shared broker. Numerical block remains128.
constexpr unsigned C209_V_WORDS=VW+128/8;
static_assert(QR==32,"C209 two-context formatter requires32 query rows");
using C209VBlock=c2::bus_t[C209_V_WORDS];
using C209VBlocks=hls::stream_of_blocks<C209VBlock,2>;
using C209UpdateStream=hls::stream<ap_uint<768>>;
#include "c209_attention_io.hpp"

#ifndef __SYNTHESIS__
template<unsigned ENGINE> static void c209_native_mac(cmd_stream&cmd,input_stream&ai,
 input_stream&bi,result_stream&co) {
 MacCommand stop={0,0,0,false,false,true};cmd.write(stop);
 c204::c209_mac_service<ENGINE>(cmd,ai,bi,co);
 if(!cmd.empty()||!ai.empty()||!bi.empty())throw std::runtime_error("C209 MAC operand cardinality");
}
#endif
// Five-shard transport is eight INT48 values per beat. Legacy callers keep
// their768-bit packet; reconstruction is local to its actual consumer.
template<int BITS> static ap_uint<768> c209_read_result_packet(hls::stream<ap_uint<BITS>>&input) {
#pragma HLS INLINE
 if(BITS==768)return ap_uint<768>(input.read());
 ap_uint<768> packet=0;
 packet.range(383,0)=input.read();packet.range(767,384)=input.read();return packet;
}
template<int BITS> static void c209_qk_receive(const Meta&m,const c2::bus_t scale[(QR+128)/8],
 hls::stream<ap_uint<BITS>>&co,ScoreBlocks&scores,MetaStream&metadata,unsigned physical_columns=128) {
#pragma HLS INLINE off
 {
  hls::write_lock<ScoreBlock> result(scores);
  for(unsigned r=0;r<m.mm;r++)for(unsigned w=0;w<8;w++) {
#pragma HLS PIPELINE II=2
   ap_uint<768> p=w<(physical_columns>>4)?c209_read_result_packet(co):ap_uint<768>(0);
   for(unsigned lane=0;lane<16;lane++) {
#pragma HLS UNROLL
    if(w*16+lane>=m.nn)p.range(lane*48+47,lane*48)=0;
   }
   result[r*8+w]=p;
  }
  for(unsigned w=0;w<(QR+128+23)/24;w++)result[DOT_WORDS+w]=0;
  for(unsigned p=0;p<QR+128;p++) {
#pragma HLS PIPELINE II=2
   put_float(result[DOT_WORDS+p/24],p%24,get_float(scale[p/8],p%8));
  }
 }
 metadata.write(m); // PIPO commit precedes metadata: consumer may acquire now.
}
static void c209_qk_exchange(const Meta&m,const KVBlock&data,
 const c2::bus_t scale[(QR+128)/8],cmd_stream&cmd,input_stream&ai,input_stream&bi,
 result_stream&co,ScoreBlocks&scores,MetaStream&metadata) {
#pragma HLS INLINE off
#ifdef __SYNTHESIS__
#pragma HLS DATAFLOW
 c209_qk_inputs(m.mm,data,cmd,ai,bi);
 c209_qk_receive(m,scale,co,scores,metadata);
#else
 cmd_stream nc;input_stream na,nb;result_stream nr;
 c209_qk_inputs(m.mm,data,nc,na,nb);
 c209_native_mac<0>(nc,na,nb,nr);c209_qk_receive(m,scale,nr,scores,metadata);
 if(!nr.empty())throw std::runtime_error("C209 QK result cardinality");
#endif
}
static void c209_qk(unsigned count,KVBlocks&operands,MetaStream&metadata,
 ScoreBlocks&scores,MetaStream&sm,C209VBlocks&values,
 cmd_stream&cmd,input_stream&ai,input_stream&bi,result_stream&co) {
#pragma HLS INLINE off
 c2::bus_t scales[(QR+128)/8];
#pragma HLS BIND_STORAGE variable=scales type=ram_2p impl=lutram
 for(unsigned i=0;i<count;i++) {
  Meta m=metadata.read();
  {
   // A single reader borrows the Q/K payload. V must be committed before
   // publishing QK results: PV consumes that independently buffered V block.
   hls::read_lock<KVBlock> in(operands);
   {
    hls::write_lock<C209VBlock> v(values);
    for(unsigned w=0;w<VW;w++) {
#pragma HLS PIPELINE II=1
     v[w]=in[QW+KW+w];
    }
    for(unsigned w=0;w<(QR+128)/8;w++) {
#pragma HLS PIPELINE II=1
     scales[w]=in[KV_SCALE+w];
    }
    for(unsigned w=0;w<16;w++) {
#pragma HLS PIPELINE II=1
     v[VW+w]=in[KV_SCALE+(QR+128)/8+w];
    }
   }
   c209_qk_exchange(m,in,scales,cmd,ai,bi,co,scores,sm);
  }
 }
}
template<int BITS> static void c209_pv_receive(const Meta&m,const c2::bus_t scales[(2*QR+128)/8],
 hls::stream<ap_uint<BITS>>&co,C209UpdateStream&updates,MetaStream&metadata) {
#pragma HLS INLINE off
 // FIFO is intentionally header-first. Publishing metadata BEFORE payload
 // lets the updater drain a finite FIFO; publishing it last would deadlock.
 metadata.write(m);
 for(unsigned word=0;word<(2*QR+128+23)/24;word++) {
#pragma HLS PIPELINE II=1
  ap_uint<768> p=0;
  for(unsigned lane=0;lane<3;lane++) {
#pragma HLS UNROLL
   unsigned n=word*3+lane;
   if(n<(2*QR+128)/8)p.range(lane*256+255,lane*256)=scales[n];
  }
  updates.write(p);
 }
 for(unsigned r=0;r<m.mm;r++)for(unsigned w=0;w<8;w++) {
#pragma HLS PIPELINE II=2
  updates.write(c209_read_result_packet(co));
 }
}
static void c209_pv_exchange(const Meta&m,unsigned inner,const ProbBlock&p,const C209VBlock&v,
 const c2::bus_t scales[(2*QR+128)/8],cmd_stream&cmd,input_stream&ai,input_stream&bi,
 result_stream&co,C209UpdateStream&updates,MetaStream&metadata) {
#pragma HLS INLINE off
#ifdef __SYNTHESIS__
#pragma HLS DATAFLOW
 c209_pv_inputs(m.mm,inner,p,v,cmd,ai,bi);
 c209_pv_receive(m,scales,co,updates,metadata);
#else
 cmd_stream nc;input_stream na,nb;result_stream nr;
 c209_pv_inputs(m.mm,inner,p,v,nc,na,nb);
 c209_native_mac<1>(nc,na,nb,nr);c209_pv_receive(m,scales,nr,updates,metadata);
 if(!nr.empty())throw std::runtime_error("C209 PV result cardinality");
#endif
}
static void c209_pv(unsigned count,ProbBlocks&probability,MetaStream&metadata,C209VBlocks&values,
 C209UpdateStream&updates,MetaStream&um,cmd_stream&cmd,input_stream&ai,input_stream&bi,result_stream&co) {
#pragma HLS INLINE off
 c2::bus_t scales[(2*QR+128)/8];
#pragma HLS BIND_STORAGE variable=scales type=ram_2p impl=lutram
 for(unsigned i=0;i<count;i++) {
  Meta m=metadata.read();
  {
   // Two independent PIPOs, one owner each; neither is recycled while its
   // finite panel producer may still read it under MAC input backpressure.
   hls::read_lock<ProbBlock> probability_in(probability);
   hls::read_lock<C209VBlock> values_in(values);
   for(unsigned w=0;w<2*QR/8;w++) {
#pragma HLS PIPELINE II=1
    scales[w]=probability_in[PROB_SCALE+w];
   }
   for(unsigned w=0;w<16;w++) {
#pragma HLS PIPELINE II=1
    scales[2*QR/8+w]=values_in[VW+w];
   }
   c209_pv_exchange(m,(m.nn+15)&~15u,probability_in,values_in,scales,cmd,ai,bi,co,updates,um);
  }
 }
}
// The numerical updater has one body. Its sink selects the legacy DDR
// destination or the production island's narrow output without a final RAM.
struct C209UpdateDDRSink {
 c2::bus_t *c;const c2::Job &j;ap_uint<128> low_half;
 C209UpdateDDRSink(c2::bus_t *memory,const c2::Job &job):c(memory),j(job),low_half(0){}
 void write(const Meta&m,unsigned r,unsigned word,unsigned half,
            ap_uint<128> half_pack,c2::bus_t float_pack) {
#pragma HLS INLINE
  if(j.fc==c2::F16) {
   if(!half)low_half=half_pack;
   else {c2::bus_t out=0;out.range(127,0)=low_half;out.range(255,128)=half_pack;
    c[(j.c>>5)+((uint64_t)(m.rb+r)*j.lc+m.h*128+word*16)/16]=out;}
  }else c[(j.c>>5)+((uint64_t)(m.rb+r)*j.lc+m.h*128+word*16+half*8)/8]=float_pack;
 }
};
struct C209UpdateStreamSink {
 hls::stream<ap_uint<128>> &output;
 explicit C209UpdateStreamSink(hls::stream<ap_uint<128>> &out):output(out){}
 void write(const Meta&,unsigned,unsigned,unsigned,ap_uint<128> half_pack,c2::bus_t) {
#pragma HLS INLINE
  output.write(half_pack);
 }
};
template<class Sink> static void c209_update_to(Sink &sink,unsigned count,
 C209UpdateStream&results,MetaStream&metadata) {
#pragma HLS INLINE off
 using namespace c2;
 // Online softmax/PV state remains FP32 across every key block. Only the
 // final ordinary activation is rounded before its typed external store.
 float u[2][QR][128],alpha[QR],den[QR],scale[128];
#pragma HLS ARRAY_PARTITION variable=u cyclic factor=8 dim=3
#pragma HLS BIND_STORAGE variable=u type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=scale cyclic factor=8
 for(unsigned block=0;block<count;block++) {
  Meta m=metadata.read();
  for(unsigned w=0;w<(2*QR+128+23)/24;w++) {
   ap_uint<768> packed=results.read();
   for(unsigned lane=0;lane<24;lane++) {
#pragma HLS PIPELINE II=1
    unsigned index=w*24+lane;float f=uf((unsigned)packed.range(31,0));packed>>=32;
    if(index<QR)alpha[index]=f;else if(index<2*QR)den[index-QR]=f;
    else if(index<2*QR+128)scale[index-2*QR]=f;
   }
  }
  for(unsigned r=0;r<m.mm;r++) {
   float al=alpha[r],inv=m.last?(den[r]==0?0.f:1.f/den[r]):0.f;
   for(unsigned word=0;word<8;word++) {
    ap_uint<768> dotword=results.read();
    for(unsigned half=0;half<2;half++) {
#pragma HLS PIPELINE II=1
     // Fresh fixed slices keep eight independent lane results out of a
     // 256-bit read/modify/write chain. Only one complete 128-bit half is
     // retained between iterations; F32 stores remain eight lanes per beat.
     ap_uint<128> half_pack=0;bus_t float_pack=0;
     ap_uint<384> dots=dotword.range(383,0);dotword>>=384;
     for(unsigned lane=0;lane<8;lane++) {
#pragma HLS UNROLL
      unsigned d=word*16+half*8+lane;i48 dot=i48(dots.range(lane*48+47,lane*48));
      float previous=m.first?0.f:u[m.context][r][d];
      float next=al*previous+(float)dot*(1.f/32760.f)*scale[d];u[m.context][r][d]=next;
      float rounded=m.last?c209mp::round(next*inv):0.f;
      half_pack.range(lane*16+15,lane*16)=c209mp::bits(c209mp::narrow(rounded));
      float_pack.range(lane*32+31,lane*32)=fu(rounded);
     }
     if(m.last)sink.write(m,r,word,half,half_pack,float_pack);
    }
   }
  }
 }
}
static void c209_update(c2::bus_t*c,const c2::Job&j,unsigned count,
 C209UpdateStream&results,MetaStream&metadata) {
#pragma HLS INLINE off
 C209UpdateDDRSink sink(c,j);c209_update_to(sink,count,results,metadata);
}
static void c209_update_stream(unsigned count,C209UpdateStream&results,
 MetaStream&metadata,hls::stream<ap_uint<128>> &output) {
#pragma HLS INLINE off
 C209UpdateStreamSink sink(output);c209_update_to(sink,count,results,metadata);
}
static void c209_attention_pipeline(c2::bus_t*a,c2::bus_t*b,c2::bus_t*c,c2::bus_t*x,
 const c2::Job&j,unsigned stored,unsigned count,float factor,
 cmd_stream&cmd0,input_stream&ai0,input_stream&bi0,result_stream&co0,
 cmd_stream&cmd1,input_stream&ai1,input_stream&bi1,result_stream&co1) {
#pragma HLS INLINE off
 KVBlocks kv;ScoreBlocks scores;ProbBlocks probability;C209VBlocks values;
 C209UpdateStream updates;MetaStream km,sm,pm,um;
#pragma HLS STREAM variable=updates depth=32
#pragma HLS BIND_STORAGE variable=updates type=fifo impl=srl
#pragma HLS STREAM variable=km depth=4
#pragma HLS STREAM variable=sm depth=4
#pragma HLS STREAM variable=pm depth=4
#pragma HLS STREAM variable=um depth=4
#ifdef __SYNTHESIS__
#pragma HLS DATAFLOW
 load_pairs(a,b,x,j,stored,kv,km);
 c209_qk(count,kv,km,scores,sm,values,cmd0,ai0,bi0,co0);
 softmax_pairs(count,j.n-stored,factor,scores,sm,probability,pm);
 c209_pv(count,probability,pm,values,updates,um,cmd1,ai1,bi1,co1);
 c209_update(c,j,count,updates,um);
#else
 for(unsigned i=0;i<4;i++)hls::stream_globals::incr_task_counter();
 std::thread loader([&]{NativeTaskGuard guard;load_pairs(a,b,x,j,stored,kv,km);});
 std::thread qk([&]{NativeTaskGuard guard;c209_qk(count,kv,km,scores,sm,values,cmd0,ai0,bi0,co0);});
 std::thread softmax([&]{NativeTaskGuard guard;softmax_pairs(count,j.n-stored,factor,scores,sm,probability,pm);});
 std::thread pv([&]{NativeTaskGuard guard;c209_pv(count,probability,pm,values,updates,um,cmd1,ai1,bi1,co1);});
 c209_update(c,j,count,updates,um);
 loader.join();qk.join();softmax.join();pv.join();
#endif
}
static unsigned attention_prequant_stage(c2::bus_t*a,c2::bus_t*b,c2::bus_t*c,c2::bus_t*x,
 const c2::Job&j,cmd_stream&cmd0,input_stream&ai0,input_stream&bi0,result_stream&co0,
 cmd_stream&cmd1,input_stream&ai1,input_stream&bi1,result_stream&co1,unsigned&tiles) {
#pragma HLS INLINE off
 using namespace c2;
 if(!j.m||j.m>MAX_ROWS||!j.n||j.n>MAX_ROWS||!j.heads||j.heads>16||
    j.la<j.heads*128||j.lb<j.heads*128||j.lc<j.heads*128||j.la%tensor::lanes(j.fa)||j.lb%16||j.lc%tensor::lanes(j.fc))return BAD_SHAPE;
 if(!(j.flags&PREQUANT_KV)||(j.flags&~(PREQUANT_KV|ZERO_KV_TAIL)))return BAD_FLAGS;
 unsigned stored=(j.flags&ZERO_KV_TAIL)?j.kc:j.n;
 if(stored>j.n)return BAD_SHAPE;
 tiles=0;
 if(!stored) {
  for(unsigned r=0;r<j.m;r++)for(unsigned d=0;d<j.heads*128;d+=tensor::lanes(j.fc)) {
#pragma HLS PIPELINE II=1
   c[(j.c>>5)+((uint64_t)r*j.lc+d)/tensor::lanes(j.fc)]=0;
  }
  return OK;
 }
 unsigned count=j.heads*((j.m+QR-1)/QR)*blocks(stored);
 float factor=j.factor==0?0.08838834764831845f:j.factor;
 c209_attention_pipeline(a,b,c,x,j,stored,count,factor,cmd0,ai0,bi0,co0,cmd1,ai1,bi1,co1);
 tiles=count*2;
 return OK;
}
