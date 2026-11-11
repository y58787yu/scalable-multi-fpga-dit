#pragma once
#include <hls_streamofblocks.h>
#ifndef __SYNTHESIS__
#include <thread>
#include <stdexcept>
#endif

// C207-only ABI. Original 128-key V scales and all FP32 associations survive.
// K: [token,head,128] i16; V: [head,block128,column128,key128] i16.
// K scale: [token,aligned(heads,8)] f32; V scale: [head,block128,column128] f32.
#ifndef C207_ATTN_QUERY_ROWS
#define C207_ATTN_QUERY_ROWS 32
#endif
#ifndef C207_ATTENTION_PIPELINE
#define C207_ATTENTION_PIPELINE 1
#endif
#ifndef C207_ATTENTION_DIRECT_BANKS
#define C207_ATTENTION_DIRECT_BANKS 1
#endif
#if C207_ATTENTION_DIRECT_BANKS && !defined(EXPLICIT_ACQUIRE_RELEASE)
#error Direct attention bank ownership requires EXPLICIT_ACQUIRE_RELEASE before the first hls_streamofblocks include
#endif
namespace c207 {
#ifndef __SYNTHESIS__
// HLS csim enables the vendor deadlock detector. Register real software
// workers just as hls::task does; otherwise any ordinary blocked FIFO reader
// is misclassified as a deadlock while another worker is actively computing.
struct NativeTaskGuard {
 NativeTaskGuard()=default;
 ~NativeTaskGuard(){hls::stream_globals::decr_task_counter();}
};
#endif
constexpr unsigned KV_BLOCK=128, QR=C207_ATTN_QUERY_ROWS, CONTEXTS=2;
static_assert(QR<=c2::BM && QR%4==0,"C207 query rows must fit MAC scratch and static softmax contexts");
static_assert(c2::BN>=128 && c2::BK>=128,"C207 preserves the original128-key attention block");
constexpr unsigned QW=QR*8, KW=128*8, VW=128*8;
constexpr unsigned KV_SCALE=QW+KW+VW, KV_WORDS=KV_SCALE+(QR+256)/8;
constexpr unsigned DOT_WORDS=QR*8, SCORE_WORDS=DOT_WORDS+(QR+128+23)/24;
constexpr unsigned PROB_SCALE=QW, PROB_WORDS=QW+(2*QR)/8;
constexpr unsigned UPDATE_WORDS=DOT_WORDS+(2*QR+128+23)/24;
using KVBlock=c2::bus_t[KV_WORDS];
using ScoreBlock=ap_uint<768>[SCORE_WORDS];
using ProbBlock=c2::bus_t[PROB_WORDS];
using UpdateBlock=ap_uint<768>[UPDATE_WORDS];
using KVBlocks=hls::stream_of_blocks<KVBlock,2>;
using ScoreBlocks=hls::stream_of_blocks<ScoreBlock,2>;
using ProbBlocks=hls::stream_of_blocks<ProbBlock,2>;
using UpdateBlocks=hls::stream_of_blocks<UpdateBlock,2>;
struct Meta { unsigned h,rb,kb,mm,nn,context; bool first,last; };
using MetaStream=hls::stream<Meta>;
// Independent physical slots allow QK(k+1) while V(k) remains borrowed. Each
// bank stores packed16-column rows; a single reader selects one scalar lane
// per rowbank to emit the exact existing panel32/scalarK/half16 MAC ABI.
constexpr unsigned PAIR_Q_ROWS=2*QR, PAIR_Q_BANK_WORDS=(PAIR_Q_ROWS/16)*8;
constexpr unsigned PAIR_K_BANK_WORDS=(128/16)*8, PAIR_K_OFFSET=PAIR_Q_BANK_WORDS;
constexpr unsigned PAIR_V_OFFSET=PAIR_K_OFFSET+PAIR_K_BANK_WORDS;
constexpr unsigned PAIR_BANK_WORDS=PAIR_V_OFFSET+PAIR_K_BANK_WORDS;
constexpr unsigned PAIR_SCALE_WORDS=(PAIR_Q_ROWS+256)/8;
using PairBank=c2::bus_t[PAIR_BANK_WORDS];
using PairBankChannel=hls::stream_of_blocks<PairBank,1>;
using PairBanks=PairBankChannel[2][16];
using PairScale=c2::bus_t[PAIR_SCALE_WORDS];
using PairScaleChannel=hls::stream_of_blocks<PairScale,1>;
using PairScales=PairScaleChannel[2];
static unsigned blocks(unsigned n){return (n+127)/128;}
static unsigned head_stride(unsigned heads){return (heads+7)&~7u;}
// Fixed32-bit lane selection avoids HLS inferring a full-word barrel shifter.
// Supported domains are lane<8 for256b and lane<24 for768b; no mathematical
// operation is changed. Defensive invalid-index defaults are tested separately.
static float get_float(const c2::bus_t&word,unsigned lane) {
#pragma HLS INLINE
 switch(lane) {
  case 0:return uf((unsigned)word.range(31,0));
  case 1:return uf((unsigned)word.range(63,32));
  case 2:return uf((unsigned)word.range(95,64));
  case 3:return uf((unsigned)word.range(127,96));
  case 4:return uf((unsigned)word.range(159,128));
  case 5:return uf((unsigned)word.range(191,160));
  case 6:return uf((unsigned)word.range(223,192));
  case 7:return uf((unsigned)word.range(255,224));
  default:return 0.f; // Unsupported index; all production callers use modulo8/24.
 }
}
static float get_float(const ap_uint<768>&word,unsigned lane) {
#pragma HLS INLINE
 switch(lane) {
  case 0:return uf((unsigned)word.range(31,0));
  case 1:return uf((unsigned)word.range(63,32));
  case 2:return uf((unsigned)word.range(95,64));
  case 3:return uf((unsigned)word.range(127,96));
  case 4:return uf((unsigned)word.range(159,128));
  case 5:return uf((unsigned)word.range(191,160));
  case 6:return uf((unsigned)word.range(223,192));
  case 7:return uf((unsigned)word.range(255,224));
  case 8:return uf((unsigned)word.range(287,256));
  case 9:return uf((unsigned)word.range(319,288));
  case 10:return uf((unsigned)word.range(351,320));
  case 11:return uf((unsigned)word.range(383,352));
  case 12:return uf((unsigned)word.range(415,384));
  case 13:return uf((unsigned)word.range(447,416));
  case 14:return uf((unsigned)word.range(479,448));
  case 15:return uf((unsigned)word.range(511,480));
  case 16:return uf((unsigned)word.range(543,512));
  case 17:return uf((unsigned)word.range(575,544));
  case 18:return uf((unsigned)word.range(607,576));
  case 19:return uf((unsigned)word.range(639,608));
  case 20:return uf((unsigned)word.range(671,640));
  case 21:return uf((unsigned)word.range(703,672));
  case 22:return uf((unsigned)word.range(735,704));
  case 23:return uf((unsigned)word.range(767,736));
  default:return 0.f; // Unsupported index; all production callers use modulo8/24.
 }
}
static void put_float(c2::bus_t&word,unsigned lane,float value) {
#pragma HLS INLINE
 unsigned bits=fu(value);
 switch(lane) {
  case 0:word.range(31,0)=bits;break;
  case 1:word.range(63,32)=bits;break;
  case 2:word.range(95,64)=bits;break;
  case 3:word.range(127,96)=bits;break;
  case 4:word.range(159,128)=bits;break;
  case 5:word.range(191,160)=bits;break;
  case 6:word.range(223,192)=bits;break;
  case 7:word.range(255,224)=bits;break;
  default:break; // Unsupported index leaves the word unchanged.
 }
}
static void put_float(ap_uint<768>&word,unsigned lane,float value) {
#pragma HLS INLINE
 unsigned bits=fu(value);
 switch(lane) {
  case 0:word.range(31,0)=bits;break;
  case 1:word.range(63,32)=bits;break;
  case 2:word.range(95,64)=bits;break;
  case 3:word.range(127,96)=bits;break;
  case 4:word.range(159,128)=bits;break;
  case 5:word.range(191,160)=bits;break;
  case 6:word.range(223,192)=bits;break;
  case 7:word.range(255,224)=bits;break;
  case 8:word.range(287,256)=bits;break;
  case 9:word.range(319,288)=bits;break;
  case 10:word.range(351,320)=bits;break;
  case 11:word.range(383,352)=bits;break;
  case 12:word.range(415,384)=bits;break;
  case 13:word.range(447,416)=bits;break;
  case 14:word.range(479,448)=bits;break;
  case 15:word.range(511,480)=bits;break;
  case 16:word.range(543,512)=bits;break;
  case 17:word.range(575,544)=bits;break;
  case 18:word.range(607,576)=bits;break;
  case 19:word.range(639,608)=bits;break;
  case 20:word.range(671,640)=bits;break;
  case 21:word.range(703,672)=bits;break;
  case 22:word.range(735,704)=bits;break;
  case 23:word.range(767,736)=bits;break;
  default:break; // Unsupported index leaves the word unchanged.
 }
}
static unsigned prepare_kv_stage(c2::bus_t*a,c2::bus_t*b,c2::bus_t*c,c2::bus_t*x,
 const c2::Job&j,unsigned&tiles) {
#pragma HLS INLINE off
 using namespace c2;
 if(!j.n||j.n>MAX_ROWS||j.m>j.n||!j.heads||j.heads>16||
    j.la<j.heads*128||j.lb<j.heads*128||j.lc<j.heads*128||
    j.la%tensor::lanes(j.fa)||j.lb%tensor::lanes(j.fb)||j.lc%16)return BAD_SHAPE;
 if(j.flags&~ZERO_KV_TAIL)return BAD_FLAGS;
 if((j.flags&ZERO_KV_TAIL)?j.kc!=j.m:j.m!=j.n)return BAD_SHAPE;
 tiles=0;if(!j.m)return OK;
 // Only raw DDR activations cross the mixed-precision boundary. Scales,
 // extrema and the subsequent INT16 quantizer remain FP32 calculations.
 c209mp::activation row[128],vf[128][128];ap_uint<31> vmax[128];
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vf cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=vf cyclic factor=8 dim=2
#pragma HLS BIND_STORAGE variable=vf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=vmax cyclic factor=8
 unsigned nb=blocks(j.m),hs=head_stride(j.heads);
 for(unsigned h=0;h<j.heads;h++)for(unsigned kb=0;kb<j.m;kb+=128) {
  unsigned nn=minv(128,j.m-kb);
  for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
   for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
    vmax[d+l]=0;
   }
  }
  for(unsigned n=0;n<nn;n++) {
   ap_uint<31> maxima[8];
#pragma HLS ARRAY_PARTITION variable=maxima complete
   for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
    maxima[l]=0;
   }
   bus_t pk=0,pv=0;const unsigned al=tensor::lanes(j.fa),bl=tensor::lanes(j.fb);
   for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
    if(d%al==0)pk=a[(j.a>>5)+((uint64_t)(kb+n)*j.la+h*128+d)/al];
    if(d%bl==0)pv=b[(j.b>>5)+((uint64_t)(kb+n)*j.lb+h*128+d)/bl];
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     c209mp::activation raw_k=c209mp::narrow(tensor::lane(pk,d%al+l,j.fa));
     c209mp::activation raw_v=c209mp::narrow(tensor::lane(pv,d%bl+l,j.fb));
     row[d+l]=raw_k;vf[n][d+l]=raw_v;
     float k=c209mp::widen(raw_k),v=c209mp::widen(raw_v);
     ap_uint<31> ak=c204::attention_absbits(k),av=c204::attention_absbits(v);
     maxima[l]=ak>maxima[l]?ak:maxima[l];vmax[d+l]=av>vmax[d+l]?av:vmax[d+l];
    }
   }
   float mx=c204::attention_absmax8(maxima),km=mx==0?1.f:mx/32760.f;
   writef(x,j.s0,(kb+n)*hs+h,km);float ki=1.f/km;
   bus_t packed_k=0;
   C209_PREPARE_K_QUANT4:for(unsigned d=0;d<128;d+=4) {
#pragma HLS PIPELINE II=1
    ap_uint<64> group=0;
    for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
     group.range(16*l+15,16*l)=ap_uint<16>(quant(c209mp::widen(row[d+l]),ki));
    }
    packed_k>>=64;packed_k.range(255,192)=group;
    if((d&15u)==12)c[(j.c>>5)+((uint64_t)(kb+n)*j.lc+h*128+d)/16]=packed_k;
   }
  }
  float vscale[128],vinverse[128];
#pragma HLS ARRAY_PARTITION variable=vscale cyclic factor=8
#pragma HLS BIND_STORAGE variable=vscale type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=vinverse type=ram_2p impl=lutram
  // Keep the original two divisions and column order. They are performed
  // ONCE per column, not inside an auto-flattened scale-RMW/V-write loop.
  C207_PREPARE_V_SCALE:for(unsigned d=0;d<128;d++) {
#pragma HLS PIPELINE off
   float mx=uf((unsigned)vmax[d]),vm=mx==0?1.f:mx/32760.f,vi=1.f/vm;
   vscale[d]=vm;vinverse[d]=vi;
  }
  if((j.s1&31u)==0) {
   C207_WRITE_V_SCALES:for(unsigned w=0;w<128/8;w++) {
#pragma HLS PIPELINE II=1
    bus_t packed=0;
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     put_float(packed,l,vscale[w*8+l]);
    }
    x[(j.s1>>5)+(uint64_t)(h*nb+kb/128)*16+w]=packed;
   }
  } else {
   // Model workspaces are32-byte aligned. Preserve the older scalar access
   // semantics for an otherwise valid direct call with unaligned scale base.
   for(unsigned d=0;d<128;d++)writef(x,j.s1,(h*nb+kb/128)*128+d,vscale[d]);
  }
  bus_t packed_v=0;
  C209_PREPARE_V_QUANT4:for(unsigned group_id=0;group_id<VW*4;group_id++) {
#pragma HLS PIPELINE II=1
    unsigned d=group_id/32,n=(group_id&31u)*4;float vi=vinverse[d];
    ap_uint<64> group=0;
    for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
     i16 value=n+l<nn?quant(c209mp::widen(vf[n+l][d]),vi):i16(0);
     group.range(16*l+15,16*l)=ap_uint<16>(value);
    }
    packed_v>>=64;packed_v.range(255,192)=group;
    if((group_id&3u)==3)x[(j.s2>>5)+(uint64_t)(h*nb+kb/128)*VW+group_id/4]=packed_v;
  }
  tiles++;
 }
 return OK;
}

// Loader reads each prequantized K/V block ONCE for a pair of query contexts.
// Q is quantized once per row, not once per key block. Two block locks bound
// look-ahead; neither context may observe a recycled operand block.
static void load_pairs(c2::bus_t*a,c2::bus_t*b,c2::bus_t*x,const c2::Job&j,
 unsigned stored,KVBlocks&out,MetaStream&metadata) {
#pragma HLS INLINE off
 using namespace c2;
 bus_t qcache[2][QW],kcache[KW],vcache[VW];float qm[2][QR],km[128],vm[128];
 c209mp::activation row[128];
#pragma HLS BIND_STORAGE variable=qcache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=kcache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=vcache type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=km cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vm cyclic factor=8
 unsigned nb=blocks(stored),hs=head_stride(j.heads);
 for(unsigned h=0;h<j.heads;h++)for(unsigned base=0;base<j.m;base+=2*QR) {
  unsigned contexts=minv(2,(j.m-base+QR-1)/QR);
  for(unsigned ctx=0;ctx<contexts;ctx++) {
   unsigned mm=minv(QR,j.m-base-ctx*QR);
   for(unsigned r=mm;r<QR;r++) {
    qm[ctx][r]=1.f;
    for(unsigned d=0;d<8;d++)qcache[ctx][r*8+d]=0;
   }
   for(unsigned r=0;r<mm;r++) {
    ap_uint<31> maxima[8];
#pragma HLS ARRAY_PARTITION variable=maxima complete
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     maxima[l]=0;
    }
    bus_t p=0;const unsigned al=tensor::lanes(j.fa);
    for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
     if(d%al==0)p=a[(j.a>>5)+((uint64_t)(base+ctx*QR+r)*j.la+h*128+d)/al];
     for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
      c209mp::activation raw=c209mp::narrow(tensor::lane(p,d%al+l,j.fa));row[d+l]=raw;
      float f=c209mp::widen(raw);ap_uint<31> v=c204::attention_absbits(f);
      maxima[l]=v>maxima[l]?v:maxima[l];
     }
    }
    float mx=c204::attention_absmax8(maxima);qm[ctx][r]=mx==0?1.f:mx/32760.f;
    float qi=1.f/qm[ctx][r];
    bus_t packed_q=0;
    C209_PREPARE_Q_QUANT4:for(unsigned d=0;d<128;d+=4) {
#pragma HLS PIPELINE II=1
     ap_uint<64> group=0;
     for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
      group.range(16*l+15,16*l)=ap_uint<16>(quant(c209mp::widen(row[d+l]),qi));
     }
     packed_q>>=64;packed_q.range(255,192)=group;
     if((d&15u)==12)qcache[ctx][r*8+d/16]=packed_q;
    }
   }
  }
  for(unsigned kb=0;kb<stored;kb+=128) {
   unsigned nn=minv(128,stored-kb);
   // Load all scales separately from the integer data. No scalar AUX read
   // interrupts each short8-word DDR burst. K's original token/head stride
   // and every IEEE754 payload bit are unchanged.
   for(unsigned n=0;n<128;n++) {
#pragma HLS PIPELINE II=1
    uint64_t address=j.s0+4ull*((uint64_t)(kb+n)*hs+h);
    km[n]=n<nn?get_float(x[address>>5],unsigned((address>>2)&7u)):1.f;
   }
   if((j.s1&31u)==0) {
    C207_READ_V_SCALES:for(unsigned w=0;w<128/8;w++) {
#pragma HLS PIPELINE II=1
     bus_t packed=x[(j.s1>>5)+(uint64_t)(h*nb+kb/128)*16+w];
     for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
      vm[w*8+l]=get_float(packed,l);
     }
    }
   } else {
    for(unsigned d=0;d<128;d++)vm[d]=readf(x,j.s1,(h*nb+kb/128)*128+d);
   }
   C207_READ_PACKED_K:for(unsigned w=0;w<KW;w++) {
#pragma HLS PIPELINE II=1
    unsigned n=w/8;
    kcache[w]=n<nn?b[(j.b>>5)+((uint64_t)(kb+n)*j.lb+h*128)/16+(w&7u)]:bus_t(0);
   }
   C207_READ_PACKED_V:for(unsigned w=0;w<VW;w++) {
#pragma HLS PIPELINE II=1
    vcache[w]=b[(j.s2>>5)+(uint64_t)(h*nb+kb/128)*VW+w];
   }
   for(unsigned ctx=0;ctx<contexts;ctx++) {
    Meta meta={h,base+ctx*QR,kb,minv(QR,j.m-base-ctx*QR),nn,ctx,kb==0,kb+nn==stored};
    {
     hls::write_lock<KVBlock> block(out);
     for(unsigned i=0;i<QW;i++) {
#pragma HLS PIPELINE II=1
      block[i]=qcache[ctx][i];
     }
     for(unsigned i=0;i<KW;i++) {
#pragma HLS PIPELINE II=1
      block[QW+i]=kcache[i];block[QW+KW+i]=vcache[i];
     }
     for(unsigned i=0;i<QR+256;i+=8) {
#pragma HLS PIPELINE II=1
      bus_t p=0;
      for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
       unsigned z=i+l;float f=z<QR?qm[ctx][z]:z<QR+128?km[z-QR]:vm[z-QR-128];put_float(p,l,f);
      }
      block[KV_SCALE+i/8]=p;
     }
    }
    metadata.write(meta);
   }
  }
 }
}

// Five fixed FP32 levels retain the original16/8/4/2/1 reduction edges.
// Separate arrays remove the in-place RAW/WAR ambiguity that previously
// scheduled the compact dynamic-index loop at II21. The 31 static adders
// are intentionally not collapsed back into one request/response service.
static float c209_sum32_compact(float values[32]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=values complete
 float level16[16],level8[8],level4[4],level2[2];
#pragma HLS ARRAY_PARTITION variable=level16 complete
#pragma HLS ARRAY_PARTITION variable=level8 complete
#pragma HLS ARRAY_PARTITION variable=level4 complete
#pragma HLS ARRAY_PARTITION variable=level2 complete
 for(unsigned i=0;i<16;i++) {
#pragma HLS UNROLL
  level16[i]=values[i]+values[i+16];
 }
 for(unsigned i=0;i<8;i++) {
#pragma HLS UNROLL
  level8[i]=level16[i]+level16[i+8];
 }
 for(unsigned i=0;i<4;i++) {
#pragma HLS UNROLL
  level4[i]=level8[i]+level8[i+4];
 }
 for(unsigned i=0;i<2;i++) {
#pragma HLS UNROLL
  level2[i]=level4[i]+level4[i+2];
 }
 return level2[0]+level2[1];
}

using ScoreQuadStream=hls::stream<ap_uint<192>>;
static void c209_read_score_quads(const ScoreBlock&input,unsigned row,ScoreQuadStream&quads) {
#pragma HLS INLINE off
 ap_uint<768> packed=0;
 C209_SCORE_READ_QUADS:for(unsigned group=0;group<32;group++) {
#pragma HLS PIPELINE II=1
  unsigned quarter=group&3u;
  if(quarter==0)packed=input[row*8+group/4];
  ap_uint<192> quad=0;
  switch(quarter) {
   case 0:quad=packed.range(191,0);break;
   case 1:quad=packed.range(383,192);break;
   case 2:quad=packed.range(575,384);break;
   default:quad=packed.range(767,576);break;
  }
  quads.write(quad);
 }
}
static void c209_score_quads(ScoreQuadStream&quads,unsigned valid,float qm,float factor,
 const float kscale[128],float maxima[32],float probability[128]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=kscale cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=maxima complete
#pragma HLS ARRAY_PARTITION variable=probability cyclic factor=4
 C209_SCORE_COMPUTE4:for(unsigned n=0;n<128;n+=4) {
#pragma HLS PIPELINE II=1
  ap_uint<192> quad=quads.read();
  for(unsigned lane=0;lane<4;lane++) {
#pragma HLS UNROLL
   unsigned z=n+lane;c2::i48 dot=quad.range(48*lane+47,48*lane);
   float s=(float)dot*qm*kscale[z]*factor;probability[z]=s;
   unsigned slot=(n&28)+lane;
   if(z<valid&&s>maxima[slot])maxima[slot]=s;
  }
 }
}
static void c209_score_row(const ScoreBlock&input,unsigned row,unsigned valid,float qm,float factor,
 const float kscale[128],float maxima[32],float probability[128]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 // The existing ScoreBlock lock remains owned by softmax_pairs throughout
 // this exchange. Only this bounded four-value stream is new: no second
 // 32x128 INT48 cache and no direct co->softmax backpressure path is added.
 ScoreQuadStream quads;
#pragma HLS STREAM variable=quads depth=4
#pragma HLS BIND_STORAGE variable=quads type=fifo impl=srl
 c209_read_score_quads(input,row,quads);
 c209_score_quads(quads,valid,qm,factor,kscale,maxima,probability);
}
// Contexts are issued sequentially. Keep the score/max scratch inside one
// non-inlined row owner instead of retaining four completely partitioned
// arrays in the caller. The score lanes and the original max comparison tree
// are unchanged, including the strict comparisons for NaNs and signed zeros.
static float c209_score_row_max(const ScoreBlock&input,unsigned row,unsigned valid,
 float qm,float factor,const float kscale[128],float previous,float probability[128]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=kscale cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=probability cyclic factor=4
 float maxima[32];
#pragma HLS ARRAY_PARTITION variable=maxima complete
 for(unsigned l=0;l<32;l++) {
#pragma HLS UNROLL
  maxima[l]=previous;
 }
 c209_score_row(input,row,valid,qm,factor,kscale,maxima,probability);
 return c204::attention_max32(maxima);
}
static void softmax_pairs(unsigned count,unsigned zero_count,float factor,
 ScoreBlocks&scores,MetaStream&sin,ProbBlocks&probability,MetaStream&pout) {
#pragma HLS INLINE off
 using namespace c2;
 float om[2][QR],ol[2][QR],prob[QR][128],qscale[QR],kscale[128];
#pragma HLS ARRAY_PARTITION variable=om complete dim=1
#pragma HLS ARRAY_PARTITION variable=ol complete dim=1
#pragma HLS ARRAY_PARTITION variable=prob cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE variable=prob type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=qscale complete
#pragma HLS ARRAY_PARTITION variable=kscale cyclic factor=4
#pragma HLS BIND_STORAGE variable=kscale type=ram_2p impl=bram
 for(unsigned block_id=0;block_id<count;block_id++) {
  Meta m=sin.read();
  {
  hls::read_lock<ScoreBlock> input(scores);hls::write_lock<ProbBlock> output(probability);
  // Scales stay FP32. The committed Score PIPO is read only once per dot
  // word by a separate fixed-slice reader; no full-tile dots copy remains.
  for(unsigned w=0;w<(QR+128+23)/24;w++) {
#pragma HLS PIPELINE II=1
   ap_uint<768> packed=input[DOT_WORDS+w];
   for(unsigned l=0;l<24;l++) {
#pragma HLS UNROLL
    unsigned index=w*24+l;float value=uf((unsigned)packed.range(32*l+31,32*l));
    if(index<QR)qscale[index]=value;else if(index<QR+128)kscale[index-QR]=value;
   }
  }
  for(unsigned rg=0;rg<m.mm;rg+=4) {
   float sum_ring[32][4],nm[4],alpha[4];
#pragma HLS ARRAY_PARTITION variable=sum_ring complete dim=2
#pragma HLS BIND_STORAGE variable=sum_ring type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=nm complete
#pragma HLS ARRAY_PARTITION variable=alpha complete
   unsigned active=minv(4,m.mm-rg);
   for(unsigned ctx=0;ctx<active;ctx++) {
    unsigned r=rg+ctx;
    if(m.first){om[m.context][r]=zero_count?0.f:-3.402823466e38f;ol[m.context][r]=(float)zero_count;}
    nm[ctx]=c209_score_row_max(input,r,m.nn,qscale[r],factor,kscale,
                             om[m.context][r],prob[r]);
    alpha[ctx]=ol[m.context][r]==0?0.f:hls::expf(om[m.context][r]-nm[ctx]);
   }
   for(unsigned i=0;i<32;i++) {
#pragma HLS PIPELINE II=1
    for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
     sum_ring[i][l]=0.f;
    }
   }
   // Ring position4*slot_group+context represents exactly one original
   // sums[context][4*slot_group+lane] cell.
   // A four-bank LUTRAM circular buffer exposes the REAL32-iteration feedback
   // without shifting128 FP32 registers.128 is a multiple of32, so the final
   // indexed order is bit-identical to the original rotating register ring.
   for(unsigned z=0;z<32*4;z++) {
#pragma HLS PIPELINE II=1 style=flp
#pragma HLS DEPENDENCE variable=sum_ring type=inter direction=RAW distance=32 true
    unsigned ctx=z&3,n=4*(z>>2),r=rg+ctx,head=z&31u;
    for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
     float previous=sum_ring[head][l],next=previous;
     if(ctx<active) {
      float p=n+l<m.nn?hls::expf(prob[r][n+l]-nm[ctx]):0.f;
      prob[r][n+l]=p;next=previous+p;
     }
     sum_ring[head][l]=next;
    }
   }
   for(unsigned ctx=0;ctx<active;ctx++) {
    float sums[32];
#pragma HLS ARRAY_PARTITION variable=sums complete
    for(unsigned g=0;g<8;g++) {
#pragma HLS PIPELINE II=1
     for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
      sums[g*4+l]=sum_ring[g*4+ctx][l];
     }
    }
    unsigned r=rg+ctx;float sum=c209_sum32_compact(sums);
    om[m.context][r]=nm[ctx];ol[m.context][r]=alpha[ctx]*ol[m.context][r]+sum;
    // These two updates deliberately retain the old multiplication/addition
    // order. No FMA, alternative reduction tree, or reassociation is allowed.
    unsigned positions[2]={r,QR+r};float values[2]={alpha[ctx],ol[m.context][r]};
    for(unsigned v=0;v<2;v++){unsigned p=positions[v];put_float(output[PROB_SCALE+p/8],p%8,values[v]);}
   }
  }
  for(unsigned r=0;r<m.mm;r++) {
   bus_t packed_probability=0;
   for(unsigned n=0;n<128;n+=4) {
#pragma HLS PIPELINE II=1
   ap_uint<64> group=0;
   for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
    group.range(16*l+15,16*l)=ap_uint<16>(quant(prob[r][n+l],32760.f));
   }
   packed_probability>>=64;packed_probability.range(255,192)=group;
   if((n&15u)==12)output[r*8+n/16]=packed_probability;
   }
  }
  }
  // Commit the complete PIPO block before advertising its metadata.
  pout.write(m);
 }
}

static void update_pairs(c2::bus_t*c,const c2::Job&j,unsigned count,
 UpdateBlocks&results,MetaStream&metadata) {
#pragma HLS INLINE off
 using namespace c2;
 float u[2][QR][128],alpha[QR],denominator[QR],column_scale[128];
 i48 dots[QR][128];
#pragma HLS ARRAY_PARTITION variable=u cyclic factor=8 dim=3
#pragma HLS BIND_STORAGE variable=u type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=dots cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=dots type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=alpha complete
#pragma HLS ARRAY_PARTITION variable=denominator complete
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=8
#pragma HLS BIND_STORAGE variable=column_scale type=ram_2p impl=bram
 for(unsigned block_id=0;block_id<count;block_id++) {
  Meta m=metadata.read();hls::read_lock<UpdateBlock> input(results);
  // The updater has the same wide-PIPO risk as the score loop. Decode at the
  // block boundary, not eight times per hot iteration; only active rows exist.
  for(unsigned r=0;r<m.mm;r++)for(unsigned w=0;w<8;w++) {
#pragma HLS PIPELINE II=1
   ap_uint<768> packed=input[r*8+w];
   for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
    dots[r][w*16+l]=i48(packed.range(48*l+47,48*l));
   }
  }
  C207_UPDATE_METADATA_WORD:for(unsigned w=0;w<(2*QR+128+23)/24;w++) {
#pragma HLS LOOP_FLATTEN off
   ap_uint<768> packed=input[DOT_WORDS+w];
   // One constant low-word extract/write per cycle avoids the24-way dynamic
   // register/banked-RAM write expansion in HLS cdfg_preprocess. This is only
   // bit transport; FP32 operations and the8-lane update below are unchanged.
   C207_UPDATE_METADATA_VALUE:for(unsigned l=0;l<24;l++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
    unsigned index=w*24+l;float value=uf((unsigned)packed.range(31,0));
    packed>>=32;
    if(index<QR)alpha[index]=value;else if(index<2*QR)denominator[index-QR]=value;
    else if(index<2*QR+128)column_scale[index-2*QR]=value;
   }
  }
  for(unsigned r=0;r<m.mm;r++) {
   float al=alpha[r],den=denominator[r];
   float inv=m.last?(den==0?0.f:1.f/den):0.f;
   bus_t output=0;const unsigned cl=c2::tensor::lanes(j.fc);
   for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
    if(d%cl==0)output=0;
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     unsigned z=d+l;i48 dot=dots[r][z];float vm=column_scale[z];
     float previous=m.first?0.f:u[m.context][r][z];
     float next=al*previous+(float)dot*(1.f/32760.f)*vm;u[m.context][r][z]=next;
     if(m.last)output=c2::tensor::set_lane(output,d%cl+l,next*inv,j.fc);
    }
    if(m.last&&d%cl+8==cl)c[(j.c>>5)+((uint64_t)(m.rb+r)*j.lc+m.h*128+d)/cl]=output;
   }
  }
 }
}

#ifndef COSMOS_C209
// ONLY this task touches the physical MAC command/data/result streams. It
// publishes two QK jobs before awaiting either softmax response: no initial
// response/ACK is required, and neither the loader nor softmax can issue MAC.
#if !defined(__SYNTHESIS__) && defined(C207_UNBATCHED_BROKER_REFERENCE)
static void mac_broker_unbatched_reference(const c2::Job&j,unsigned stored,KVBlocks&operands,MetaStream&kin,
 ScoreBlocks&scores,MetaStream&sout,ProbBlocks&probability,MetaStream&pin,
 UpdateBlocks&updates,MetaStream&uout,c2::i16 at[c2::BM][c2::BK],
 c2::i16 bt[c2::BN][c2::BK],c2::i48 ct[c2::BM][c2::BN],
 cmd_stream&cmd,input_stream&ai,input_stream&bi,result_stream&co,unsigned&tiles) {
#pragma HLS INLINE off
 using namespace c2;
 bus_t retained_v[2][VW];float retained_vm[2][128];Meta issued[2];
#pragma HLS BIND_STORAGE variable=retained_v type=ram_2p impl=bram
 tiles=0;
 for(unsigned h=0;h<j.heads;h++)for(unsigned base=0;base<j.m;base+=2*QR) {
  unsigned contexts=minv(2,(j.m-base+QR-1)/QR);
  for(unsigned kb=0;kb<stored;kb+=128) {
   for(unsigned ctx=0;ctx<contexts;ctx++) {
    Meta m=kin.read();issued[ctx]=m;
    {
     hls::read_lock<KVBlock> block(operands);
     for(unsigned r=0;r<m.mm;r++)for(unsigned d=0;d<128;d+=16) {
#pragma HLS PIPELINE II=1
      bus_t p=block[r*8+d/16];
      for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
       at[r][d+l]=i16(p.range(16*l+15,16*l));
      }
     }
     unsigned padded=((m.nn+PO-1)/PO)*PO;
     for(unsigned n=0;n<padded;n++)for(unsigned d=0;d<128;d+=16) {
#pragma HLS PIPELINE II=1
      bus_t p=block[QW+n*8+d/16];
      for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
       bt[n][d+l]=i16(p.range(16*l+15,16*l));
      }
     }
     for(unsigned i=0;i<VW;i++) {
#pragma HLS PIPELINE II=1
      retained_v[ctx][i]=block[QW+KW+i];
     }
     for(unsigned d=0;d<128;d++)retained_vm[ctx][d]=get_float(block[KV_SCALE+(QR+128+d)/8],(QR+128+d)%8);
     mac_tile(at,bt,ct,m.mm,padded,128,true,true,cmd,ai,bi,co);
     {
      hls::write_lock<ScoreBlock> output(scores);
      for(unsigned r=0;r<m.mm;r++)for(unsigned n=0;n<128;n+=16) {
#pragma HLS PIPELINE II=1
       ap_uint<768> p=0;
       for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
        p.range(48*l+47,48*l)=n+l<m.nn?ap_uint<48>(ct[r][n+l]):ap_uint<48>(0);
       }
       output[r*8+n/16]=p;
      }
      for(unsigned i=0;i<(QR+128+23)/24;i++)output[DOT_WORDS+i]=0;
      for(unsigned i=0;i<QR+128;i++)put_float(output[DOT_WORDS+i/24],i%24,get_float(block[KV_SCALE+i/8],i%8));
     }
    }
    sout.write(m);tiles++;
   }
   for(unsigned ctx=0;ctx<contexts;ctx++) {
    Meta m=pin.read();
    {
     hls::read_lock<ProbBlock> block(probability);
     unsigned inner=(m.nn+15)&~15u;
     for(unsigned r=0;r<m.mm;r++)for(unsigned n=0;n<inner;n+=16) {
#pragma HLS PIPELINE II=1
      bus_t p=block[r*8+n/16];
      for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
       at[r][n+l]=i16(p.range(16*l+15,16*l));
      }
     }
     for(unsigned d=0;d<128;d++)for(unsigned n=0;n<inner;n+=16) {
#pragma HLS PIPELINE II=1
      bus_t p=retained_v[ctx][d*8+n/16];
      for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
       bt[d][n+l]=i16(p.range(16*l+15,16*l));
      }
     }
     mac_tile(at,bt,ct,m.mm,128,inner,true,true,cmd,ai,bi,co);
     {
      hls::write_lock<UpdateBlock> output(updates);
      for(unsigned r=0;r<m.mm;r++)for(unsigned d=0;d<128;d+=16) {
#pragma HLS PIPELINE II=1
       ap_uint<768> p=0;
       for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
        p.range(48*l+47,48*l)=ap_uint<48>(ct[r][d+l]);
       }
       output[r*8+d/16]=p;
      }
      for(unsigned i=0;i<(2*QR+128+23)/24;i++)output[DOT_WORDS+i]=0;
      for(unsigned i=0;i<2*QR;i++)put_float(output[DOT_WORDS+i/24],i%24,get_float(block[PROB_SCALE+i/8],i%8));
      for(unsigned d=0;d<128;d++)put_float(output[DOT_WORDS+(2*QR+d)/24],(2*QR+d)%24,retained_vm[ctx][d]);
     }
    }
    uout.write(m);tiles++;
   }
  }
 }
}

#endif
#include "c207_attention_batched_broker.hpp"

static void attention_pipeline(c2::bus_t*a,c2::bus_t*b,c2::bus_t*c,c2::bus_t*x,
 const c2::Job&j,unsigned stored,c2::i16 at[c2::BM][c2::BK],c2::i16 bt[c2::BN][c2::BK],
 c2::i48 ct[c2::BM][c2::BN],cmd_stream&cmd,input_stream&ai,input_stream&bi,result_stream&co,unsigned&tiles) {
#pragma HLS INLINE off
 using namespace c2;
 unsigned count=j.heads*((j.m+QR-1)/QR)*blocks(stored);
 float factor=j.factor==0?0.08838834764831845f:j.factor;
#if C207_ATTENTION_DIRECT_BANKS
 PairBanks kv;PairScales scales;
#pragma HLS ARRAY_PARTITION variable=kv complete dim=1
#pragma HLS ARRAY_PARTITION variable=kv complete dim=2
#pragma HLS ARRAY_PARTITION variable=scales complete dim=1
#pragma HLS BIND_STORAGE variable=kv type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=scales type=ram_2p impl=bram
#else
 KVBlocks kv;
#endif
 ScoreBlocks score;ProbBlocks probability;UpdateBlocks update;
 MetaStream km,sm,pm,um;
#pragma HLS STREAM variable=km depth=4
#pragma HLS STREAM variable=sm depth=4
#pragma HLS STREAM variable=pm depth=4
#pragma HLS STREAM variable=um depth=4
#if C207_ATTENTION_PIPELINE
#ifdef __SYNTHESIS__
#pragma HLS DATAFLOW
#if C207_ATTENTION_DIRECT_BANKS
 load_pairs_direct(a,b,x,j,stored,kv,scales,km);
 mac_broker_direct(j,stored,kv,scales,km,score,sm,probability,pm,update,um,cmd,ai,bi,co,tiles);
#else
 load_pairs(a,b,x,j,stored,kv,km);
 mac_broker(j,stored,kv,km,score,sm,probability,pm,update,um,at,bt,ct,cmd,ai,bi,co,tiles);
#endif
 softmax_pairs(count,j.n-stored,factor,score,sm,probability,pm);
 update_pairs(c,j,count,update,um);
#else
 // Native uses the SAME task functions, concurrent only to model their token
 // cycle. Vendor native block queues are unbounded; finite RTL is a separate
 // required cosimulation gate, never inferred from this CPU equivalence test.
 // Register before creating workers so startup has no unregistered interval;
 // each real worker unregisters on exit (inactive workers cannot mask a stall).
 hls::stream_globals::incr_task_counter();hls::stream_globals::incr_task_counter();hls::stream_globals::incr_task_counter();
#if C207_ATTENTION_DIRECT_BANKS
 std::thread loader([&]{NativeTaskGuard guard;load_pairs_direct(a,b,x,j,stored,kv,scales,km);});
#else
 std::thread loader([&]{NativeTaskGuard guard;load_pairs(a,b,x,j,stored,kv,km);});
#endif
 std::thread softmax([&]{NativeTaskGuard guard;softmax_pairs(count,j.n-stored,factor,score,sm,probability,pm);});
 std::thread updater([&]{NativeTaskGuard guard;update_pairs(c,j,count,update,um);});
#if C207_ATTENTION_DIRECT_BANKS
 mac_broker_direct(j,stored,kv,scales,km,score,sm,probability,pm,update,um,cmd,ai,bi,co,tiles);
#else
 mac_broker(j,stored,kv,km,score,sm,probability,pm,update,um,at,bt,ct,cmd,ai,bi,co,tiles);
#endif
 loader.join();softmax.join();updater.join();
#endif
#else
#error C207_ATTENTION_PIPELINE must be enabled for this candidate; no silent raw fallback
#endif
}
static unsigned attention_prequant_stage(c2::bus_t*a,c2::bus_t*b,c2::bus_t*c,c2::bus_t*x,
 const c2::Job&j,c2::i16 at[c2::BM][c2::BK],c2::i16 bt[c2::BN][c2::BK],
 c2::i48 ct[c2::BM][c2::BN],cmd_stream&cmd,input_stream&ai,input_stream&bi,result_stream&co,unsigned&tiles) {
#pragma HLS INLINE off
 using namespace c2;
 if(!j.m||j.m>MAX_ROWS||!j.n||j.n>MAX_ROWS||!j.heads||j.heads>16||
    j.la<j.heads*128||j.lb<j.heads*128||j.lc<j.heads*128||j.la%tensor::lanes(j.fa)||j.lb%16||j.lc%tensor::lanes(j.fc))return BAD_SHAPE;
 if(!(j.flags&128u)||(j.flags&~(128u|ZERO_KV_TAIL)))return BAD_FLAGS;
 unsigned stored=(j.flags&ZERO_KV_TAIL)?j.kc:j.n;
 if(stored>j.n)return BAD_SHAPE;
 tiles=0;if(!stored) {
  for(unsigned r=0;r<j.m;r++)for(unsigned d=0;d<j.heads*128;d+=tensor::lanes(j.fc)) {
#pragma HLS PIPELINE II=1
   c[(j.c>>5)+((uint64_t)r*j.lc+d)/tensor::lanes(j.fc)]=0;
  }
  return OK;
 }
 attention_pipeline(a,b,c,x,j,stored,at,bt,ct,cmd,ai,bi,co,tiles);
 return OK;
}
#else
#include "c209_attention.hpp"
#endif
} // namespace c207
