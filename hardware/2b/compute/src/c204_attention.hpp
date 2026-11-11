#pragma once

// Included after mac_tile: the physical MAC service and quantization semantics
// are shared with GEMM. ZERO_KV_TAIL is an explicit host-verified contract: only
// kc compact K/V rows are stored, and the other n-kc rows are exactly zero.
namespace c204 {
static float attention_max32(float v[32]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=v complete
 for(unsigned i=0;i<16;i++) {
#pragma HLS UNROLL
  v[i]=v[i]>v[i+16]?v[i]:v[i+16];
 }
 for(unsigned i=0;i<8;i++) {
#pragma HLS UNROLL
  v[i]=v[i]>v[i+8]?v[i]:v[i+8];
 }
 for(unsigned i=0;i<4;i++) {
#pragma HLS UNROLL
  v[i]=v[i]>v[i+4]?v[i]:v[i+4];
 }
 float a=v[0]>v[2]?v[0]:v[2],b=v[1]>v[3]?v[1]:v[3];
 return a>b?a:b;
}
// Positive IEEE754 magnitudes have exactly unsigned-integer ordering.
// Ignore NaNs as the previous `fabs(x)>max` comparison did; retain infinity.
static ap_uint<31> attention_absbits(float x) {
#pragma HLS INLINE
 ap_uint<31> bits=fu(x);
 return bits>0x7f800000u?ap_uint<31>(0):bits;
}
static float attention_absmax8(ap_uint<31> values[8]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=values complete
 ap_uint<31> pairs[4],halves[2];
#pragma HLS ARRAY_PARTITION variable=pairs complete
#pragma HLS ARRAY_PARTITION variable=halves complete
 for(unsigned i=0;i<4;i++) {
#pragma HLS UNROLL
  pairs[i]=values[2*i]>values[2*i+1]?values[2*i]:values[2*i+1];
 }
 for(unsigned i=0;i<2;i++) {
#pragma HLS UNROLL
  halves[i]=pairs[2*i]>pairs[2*i+1]?pairs[2*i]:pairs[2*i+1];
 }
 return uf((unsigned)(halves[0]>halves[1]?halves[0]:halves[1]));
}
static float attention_sum32(float v[32]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=v complete
 for(unsigned i=0;i<16;i++) {
#pragma HLS UNROLL
  v[i]+=v[i+16];
 }
 for(unsigned i=0;i<8;i++) {
#pragma HLS UNROLL
  v[i]+=v[i+8];
 }
 for(unsigned i=0;i<4;i++) {
#pragma HLS UNROLL
  v[i]+=v[i+4];
 }
 return (v[0]+v[2])+(v[1]+v[3]);
}
#if !defined(__SYNTHESIS__) && defined(C207_ATTENTION_REFERENCE)
static unsigned fast_attention_stage_reference(bus_t*a,bus_t*b,bus_t*c,bus_t*x,const Job&j,
 i16 at[BM][BK],i16 bt[BN][BK],i48 ct[BM][BN],cmd_stream&cmd,
 input_stream&ai,input_stream&bi,result_stream&co,unsigned&tiles) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=at cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=bt cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=ct cyclic factor=16 dim=2
 if(!j.m||j.m>MAX_ROWS||!j.n||j.n>MAX_ROWS||!j.heads||j.heads>16||
    j.la<j.heads*HEAD_DIM||j.lb<j.heads*HEAD_DIM||j.lc<j.heads*HEAD_DIM||
    j.la%8||j.lb%8||j.lc%8)return BAD_SHAPE;
 const bool compact=(j.flags&ZERO_KV_TAIL)!=0;
 if(compact&&(j.flags&SKIP_K))return BAD_FLAGS;
 if(compact&&j.kc>j.n)return BAD_SHAPE;
 const unsigned stored=compact?j.kc:j.n,zero_count=j.n-stored;
 // An all-zero context needs no input or weight access, including Q reads.
 if(stored==0) {
  for(unsigned r=0;r<j.m;r++)for(unsigned d=0;d<j.heads*HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
   c[(j.c>>5)+((uint64_t)r*j.lc+d)/8]=0;
  }
  return OK;
 }
 // One-head cache only: rebuild at rb=0 for every head and every call.
 // Quantized V is cached with its original 128-key-block column scales, so
 // reuse does not change V quantization or online attention semantics.
 const bool cacheable=stored<=512&&j.m>BM;
 ap_uint<128> key_cache[512][HEAD_DIM/8];
 ap_uint<128> value_cache[4][HEAD_DIM][BN/8];
 float key_scale_cache[512];bus_t value_scale_cache[4][HEAD_DIM/8];
#pragma HLS BIND_STORAGE variable=key_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=value_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=key_scale_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=value_scale_cache type=ram_2p impl=bram
 i16 qq[BM][HEAD_DIM];
 float row[HEAD_DIM],vf[BN][HEAD_DIM],prob[BM][BN],u[BM][HEAD_DIM];
 float qm[BM],km[BN],vm[HEAD_DIM],vi[HEAD_DIM],om[BM],ol[BM],alpha[BM];
 ap_uint<31> vmax[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=vmax cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=qq cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=qq type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vf cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=vf cyclic factor=8 dim=2
#ifdef COSMOS_C205
#pragma HLS BIND_STORAGE variable=vf type=ram_1p impl=bram
#endif
#pragma HLS ARRAY_PARTITION variable=prob cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=u cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=km cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vm cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vi cyclic factor=8
 const float factor=j.factor==0?0.08838834764831845f:j.factor;
 for(unsigned h=0;h<j.heads;h++)for(unsigned rb=0;rb<j.m;rb+=BM) {
  const unsigned mm=minv(BM,j.m-rb);
  for(unsigned r=0;r<mm;r++) {
   ap_uint<31> maxima[8];
#pragma HLS ARRAY_PARTITION variable=maxima complete
   for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
    maxima[l]=0;
   }
   // Preseed zero logits with their exact multiplicity. This is the stable
   // online state (m=0,l=zero_count,u=0), not adding zero_count after scaling.
   om[r]=zero_count?0.f:-3.402823466e38f;ol[r]=(float)zero_count;
   for(unsigned d=0;d<HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
    bus_t p=a[(j.a>>5)+((uint64_t)(rb+r)*j.la+h*HEAD_DIM+d)/8];
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     float f=uf(p.range(32*l+31,32*l));ap_uint<31> v=attention_absbits(f);
     row[d+l]=f;u[r][d+l]=0;
     maxima[l]=v>maxima[l]?v:maxima[l];
    }
   }
   float mx=attention_absmax8(maxima);qm[r]=mx==0?1:mx/32760.f;
   float qi=1.f/qm[r];
   // Quantize each Q row once, then replay integers for subsequent K blocks.
   for(unsigned d=0;d<HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     qq[r][d+l]=quant(row[d+l],qi);
    }
   }
  }
  for(unsigned kb=0;kb<stored;kb+=BN) {
   const unsigned nn=minv(BN,stored-kb);
   const unsigned qk_cols=((nn+PO-1)/PO)*PO;
   const unsigned av_inner=(nn+15)&~15u;
   if(j.flags&SKIP_K) {bus_t mask=x[(j.s3>>5)+(kb/BN)/256];if(mask[(kb/BN)&255]==0)continue;}
   const bool reuse=cacheable&&rb!=0;
   const unsigned block=kb/BN;
   if(reuse) {
    for(unsigned n=0;n<qk_cols;n++) {
     km[n]=key_scale_cache[kb+n];
     for(unsigned d=0;d<HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
      ap_uint<128> packed=key_cache[kb+n][d/8];
      for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
       bt[n][d+l]=i16(packed.range(16*l+15,16*l));
      }
     }
    }
   } else {
   for(unsigned d=0;d<HEAD_DIM;d+=8) {
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
    for(unsigned d=0;d<HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
     bus_t pk=b[(j.b>>5)+((uint64_t)(kb+n)*j.lb+h*HEAD_DIM+d)/8];
     bus_t pv=b[(j.s2>>5)+((uint64_t)(kb+n)*j.lb+h*HEAD_DIM+d)/8];
     for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
      float k=uf(pk.range(32*l+31,32*l)),v=uf(pv.range(32*l+31,32*l));
      ap_uint<31> ak=attention_absbits(k),av=attention_absbits(v);
      row[d+l]=k;vf[n][d+l]=v;
      maxima[l]=ak>maxima[l]?ak:maxima[l];
      // Eight V columns advance together; each column recurs only next row.
      vmax[d+l]=av>vmax[d+l]?av:vmax[d+l];
     }
    }
    float mx=attention_absmax8(maxima);km[n]=mx==0?1:mx/32760.f;
    if(cacheable)key_scale_cache[kb+n]=km[n];
    float ki=1.f/km[n];
    for(unsigned d=0;d<HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
     ap_uint<128> packed=0;
     for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
      i16 value=quant(row[d+l],ki);bt[n][d+l]=value;
      packed.range(16*l+15,16*l)=ap_uint<16>(value);
     }
     if(cacheable)key_cache[kb+n][d/8]=packed;
    }
   }
   // Integer padding is cheap; do not normalize and quantize absent rows.
   for(unsigned n=nn;n<qk_cols;n++) {
    km[n]=1.f;if(cacheable)key_scale_cache[kb+n]=1.f;
    for(unsigned d=0;d<HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
     for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
      bt[n][d+l]=0;vf[n][d+l]=0.f;
     }
     if(cacheable)key_cache[kb+n][d/8]=0;
    }
   }
   }
   for(unsigned r=0;r<mm;r++)for(unsigned d=0;d<HEAD_DIM;d+=16) {
#pragma HLS PIPELINE II=1
    for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
     at[r][d+l]=qq[r][d+l];
    }
   }
   mac_tile(at,bt,ct,mm,qk_cols,HEAD_DIM,true,true,cmd,ai,bi,co);
   for(unsigned r=0;r<mm;r++) {
    float maxima[32],sums[32];
#pragma HLS ARRAY_PARTITION variable=maxima complete
#pragma HLS ARRAY_PARTITION variable=sums complete
    for(unsigned l=0;l<32;l++) {
#pragma HLS UNROLL
     maxima[l]=om[r];sums[l]=0;
    }
    for(unsigned n=0;n<qk_cols;n+=8) {
#pragma HLS PIPELINE II=1
     for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
      float s=(float)ct[r][n+l]*qm[r]*km[n+l]*factor;
      unsigned slot=(n&24)+l;prob[r][n+l]=s;
      if(n+l<nn&&s>maxima[slot])maxima[slot]=s;
     }
    }
    float nm=attention_max32(maxima);
    float al=ol[r]==0?0:hls::expf(om[r]-nm);
    // Four exponent lanes and eight-way interleaved sums remove the scalar
    // FP32 feedback bottleneck without instantiating 128 exponent units.
    for(unsigned n=0;n<av_inner;n+=4) {
#pragma HLS PIPELINE II=1
     for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
      float p=n+l<nn?hls::expf(prob[r][n+l]-nm):0;
      unsigned slot=(n&28)+l;prob[r][n+l]=p;sums[slot]+=p;
     }
    }
    float sum=attention_sum32(sums);
    alpha[r]=al;om[r]=nm;ol[r]=al*ol[r]+sum;
   }
   for(unsigned d=0;d<HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
    bus_t cached=0;if(reuse)cached=value_scale_cache[block][d/8];
    bus_t saved=0;
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     float mx=uf((unsigned)vmax[d+l]);
     vm[d+l]=reuse?uf(cached.range(32*l+31,32*l)):(mx==0?1:mx/32760.f);
     if(!reuse)vi[d+l]=1.f/vm[d+l];
     saved.range(32*l+31,32*l)=fu(vm[d+l]);
    }
    if(cacheable&&!reuse)value_scale_cache[block][d/8]=saved;
   }
   for(unsigned r=0;r<mm;r++)for(unsigned n=0;n<av_inner;n+=8) {
#pragma HLS PIPELINE II=1
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     at[r][n+l]=quant(prob[r][n+l],32760.f);
    }
   }
   for(unsigned d=0;d<HEAD_DIM;d++)for(unsigned n=0;n<av_inner;n+=8) {
#pragma HLS PIPELINE II=1
    ap_uint<128> packed=0;if(reuse)packed=value_cache[block][d][n/8];
    ap_uint<128> saved=0;
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     i16 value=reuse?i16(packed.range(16*l+15,16*l)):quant(vf[n+l][d],vi[d]);
     bt[d][n+l]=value;saved.range(16*l+15,16*l)=ap_uint<16>(value);
    }
    if(cacheable&&!reuse)value_cache[block][d][n/8]=saved;
   }
   mac_tile(at,bt,ct,mm,HEAD_DIM,av_inner,true,true,cmd,ai,bi,co);
   for(unsigned r=0;r<mm;r++)for(unsigned d=0;d<HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     u[r][d+l]=alpha[r]*u[r][d+l]+(float)ct[r][d+l]*(1.f/32760.f)*vm[d+l];
    }
   }
   tiles+=2;
  }
  for(unsigned r=0;r<mm;r++) {
   float inv=ol[r]==0?0.f:1.f/ol[r];
   for(unsigned d=0;d<HEAD_DIM;d+=8) {
#pragma HLS PIPELINE II=1
    bus_t p=0;
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     p.range(32*l+31,32*l)=fu(u[r][d+l]*inv);
    }
    c[(j.c>>5)+((uint64_t)(rb+r)*j.lc+h*HEAD_DIM+d)/8]=p;
   }
  }
 }
 return OK;
}
#endif
} // namespace c204

#include "c207_attention.hpp"
#ifndef COSMOS_C209
namespace c204 {
static unsigned fast_attention_stage(bus_t*a,bus_t*b,bus_t*c,bus_t*x,const Job&j,
 i16 at[BM][BK],i16 bt[BN][BK],i48 ct[BM][BN],cmd_stream&cmd,
 input_stream&ai,input_stream&bi,result_stream&co,unsigned&tiles) {
#pragma HLS INLINE off
 return c207::attention_prequant_stage(a,b,c,x,j,at,bt,ct,cmd,ai,bi,co,tiles);
}
}
#endif
