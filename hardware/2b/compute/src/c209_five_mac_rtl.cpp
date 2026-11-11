#include "c209_five_mac_rtl.hpp"
#include <cassert>
// Numerical oracle for the row-major imported physical RTL boundary.
// Transposition is now internal to RTL; never a synthesized C fallback.
void c209_five_mac_rtl(hls::stream<ap_uint<12>>& commands,
 hls::stream<ap_uint<256>>& a0,
 hls::stream<ap_uint<256>>& b0,
 hls::stream<ap_uint<384>>& r0,
 hls::stream<ap_uint<256>>& a1,
 hls::stream<ap_uint<256>>& b1,
 hls::stream<ap_uint<384>>& r1,
 hls::stream<ap_uint<256>>& a2,
 hls::stream<ap_uint<256>>& b2,
 hls::stream<ap_uint<384>>& r2,
 hls::stream<ap_uint<256>>& a3,
 hls::stream<ap_uint<256>>& b3,
 hls::stream<ap_uint<384>>& r3,
 hls::stream<ap_uint<256>>& a4,
 hls::stream<ap_uint<256>>& b4,
 hls::stream<ap_uint<384>>& r4,
 hls::stream<ap_uint<256>>& perf) {
 static ap_int<16> shared_a[256][128],private_a[5][32][128],b[2][5][128][128];
 static ap_int<48> c[5][256][128];
 static unsigned active_bank=1;
 bool b_valid[2]={false,false};unsigned bn[2]={},bk[2]={};
 bool live[5]={false,false,false,false,false};unsigned cm[5]={},cn[5]={};
 hls::stream<ap_uint<256>>* ai[5]={&a0,&a1,&a2,&a3,&a4};
 hls::stream<ap_uint<256>>* bi[5]={&b0,&b1,&b2,&b3,&b4};
 hls::stream<ap_uint<384>>* co[5]={&r0,&r1,&r2,&r3,&r4};
 for(;;) {
  ap_uint<12> fixed=commands.read();ap_uint<106> p=0;
  unsigned mc=fixed.range(1,0),kc=fixed.range(4,3);
  p.range(31,0)=mc==2?256u:mc==1?128u:32u;p.range(63,32)=fixed[2]?128u:fixed[9]?16u:64u;
  p.range(95,64)=kc==2?128u:kc==1?64u:16u;
  p[96]=fixed[5];p[97]=fixed[6];p[98]=fixed[7];p.range(103,99)=fixed[8]?31u:fixed.range(11,10)==1?3u:fixed.range(11,10)==2?7u:1u;p[104]=fixed[9];
  p[105]=fixed[9]&&mc==3;
  if(p[98]) {for(unsigned s=0;s<5;s++)assert(!live[s]);break;}
  unsigned m=p.range(31,0),n=p.range(63,32),k=p.range(95,64),mask=p.range(103,99);
  bool individual=p[104],reset=p[96],commit=p[97],reuse_b=p[105];
  assert(kc!=3&&fixed.range(11,10)!=3&&(!fixed[8]||fixed.range(11,10)==0)&&
         (individual?(fixed[8]&&(mc==0||mc==3)):(mc==1||mc==2)));
  assert(m>0&&m<=256&&n>0&&n<=128&&k>0&&k<=128&&k%16==0&&mask&&(!individual||m<=32));
  unsigned bank=active_bank^1u;
  if(reuse_b)assert(individual&&mask==31&&b_valid[bank]&&bn[bank]==n&&bk[bank]==k);
  else b_valid[bank]=false;
  for(unsigned s=0;s<5;s++)if(mask&(1u<<s)) {
   assert(reset||(live[s]&&cm[s]==m&&cn[s]==n));live[s]=true;cm[s]=m;cn[s]=n;
  }
  for(unsigned s=0;s<5;s++)if(individual?bool(mask&(1u<<s)):s==0)
   for(unsigned rb=0;rb<m;rb+=32)for(unsigned row=0;row<32;row++)for(unsigned word=0;word<k/16;word++) {
    ap_uint<256> w=ai[s]->read();
    for(unsigned lane=0;lane<16;lane++) {
     ap_int<16> value=ap_int<16>(w.range(16*lane+15,16*lane));
     if(rb+row<m) {
      if(individual)private_a[s][rb+row][word*16+lane]=value;
      else shared_a[rb+row][word*16+lane]=value;
     }
    }
   }
  if(!reuse_b)for(unsigned s=0;s<5;s++)if(mask&(1u<<s))
   for(unsigned nb=0;nb<n;nb+=32)for(unsigned row=0;row<32;row++)for(unsigned word=0;word<k/16;word++) {
    ap_uint<256> w=bi[s]->read();
    for(unsigned lane=0;lane<16;lane++)if(nb+row<n)b[bank][s][nb+row][word*16+lane]=ap_int<16>(w.range(16*lane+15,16*lane));
   }
  if(!reuse_b){b_valid[bank]=individual&&mask==31;bn[bank]=n;bk[bank]=k;}
  active_bank=bank;
  for(unsigned s=0;s<5;s++)if(mask&(1u<<s)) {
   for(unsigned r=0;r<m;r++)for(unsigned col=0;col<n;col++) {
    ap_int<48> sum=reset?ap_int<48>(0):c[s][r][col];
    for(unsigned t=0;t<k;t++) {
     ap_int<16> av=individual?private_a[s][r][t]:shared_a[r][t];
     ap_int<32> product=av*b[bank][s][col][t];sum+=ap_int<48>(product);
    }
    c[s][r][col]=sum;
   }
   if(commit) {
    for(unsigned r=0;r<m;r++)for(unsigned col=0;col<n;col+=16) {
     ap_uint<768> w=0;
     for(unsigned lane=0;lane<16;lane++)if(col+lane<n)
      w.range(48*lane+47,48*lane)=ap_uint<48>(c[s][r][col+lane]);
     co[s]->write(ap_uint<384>(w.range(383,0)));
     co[s]->write(ap_uint<384>(w.range(767,384)));
    }
    live[s]=false;
   }
  }
 }
 perf.write(ap_uint<256>(0));
}
