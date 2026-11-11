#pragma once
// Common production compute declarations and helpers. Moved verbatim from
// cosmos.cpp so stream-only owners need not include a production translation
// unit. The helpers and arithmetic associations are unchanged.
#include "cosmos.hpp"
#include "c209_tensor_io.hpp"
#include "c209_job_ranges.hpp"
#include <hls_math.h>
using namespace c2;
static float uf(unsigned u){union{unsigned u;float f;}v;v.u=u;return v.f;}
static unsigned fu(float f){union{unsigned u;float f;}v;v.f=f;return v.u;}
static float af(float x){return x<0?-x:x;}
static float readf(bus_t*p,uint64_t a,unsigned long long i,unsigned format=F32){return tensor::read(p,a,i,format);}
static void writef(bus_t*p,uint64_t a,unsigned long long i,float f,unsigned format=F32){tensor::write(p,a,i,f,format);}
static i16 quant(float x,float inverse_scale){float y=x*inverse_scale;int z=(int)(y>=0?y+.5f:y-.5f);if(z>32760)z=32760;if(z< -32760)z=-32760;return i16(z);}
static float gelu(float x){return .5f*x*(1.f+hls::tanhf(.7978845608f*(x+.044715f*x*x*x)));}
static unsigned minv(unsigned a,unsigned b){return a<b?a:b;}
template<class T> static void frow(bus_t*p,uint64_t a,unsigned count,T row[FFN],unsigned format=F32){
#pragma HLS INLINE off
 const unsigned width=tensor::lanes(format);
 bus_t pack=0;
 // Preserve eight arithmetic lanes/cycle. A packed F16 bus beat is held for
 // two cycles, reducing DDR beats without duplicating conversion hardware.
 for(unsigned j=0;j<count;j+=8){
#pragma HLS PIPELINE II=1
  if(j%width==0)pack=p[(a>>5)+j/width];
  for(unsigned l=0;l<8;l++){
#pragma HLS UNROLL
   if(j+l<count)row[j+l]=c209mp::convert<T>(tensor::lane(pack,j%width+l,format));
  }
 }
}
struct MacCommand{unsigned m,n,k;bool reset,commit,stop;};
using cmd_stream=hls::stream<MacCommand>;using input_stream=hls::stream<bus_t>;
using result_stream=hls::stream<ap_uint<768>>;
