#pragma once
// Included inside namespace c207. One consumer owns all bank reads. This
// avoids a hidden third RAM port (two readers plus the next-slot loader).
#define C207_BANK_CONST const PairBank&b0,const PairBank&b1,const PairBank&b2,const PairBank&b3,const PairBank&b4,const PairBank&b5,const PairBank&b6,const PairBank&b7,const PairBank&b8,const PairBank&b9,const PairBank&b10,const PairBank&b11,const PairBank&b12,const PairBank&b13,const PairBank&b14,const PairBank&b15
#define C207_BANK_MUT PairBank&b0,PairBank&b1,PairBank&b2,PairBank&b3,PairBank&b4,PairBank&b5,PairBank&b6,PairBank&b7,PairBank&b8,PairBank&b9,PairBank&b10,PairBank&b11,PairBank&b12,PairBank&b13,PairBank&b14,PairBank&b15
#define C207_BANK_ARGS b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15
#define C207_SLOT_ARGS(p) p##0,p##1,p##2,p##3,p##4,p##5,p##6,p##7,p##8,p##9,p##10,p##11,p##12,p##13,p##14,p##15
#define C207_DECLARE_SLOT(p,kind,source,slot) hls::kind<PairBank> p##0(source[slot][0]);hls::kind<PairBank> p##1(source[slot][1]);hls::kind<PairBank> p##2(source[slot][2]);hls::kind<PairBank> p##3(source[slot][3]);hls::kind<PairBank> p##4(source[slot][4]);hls::kind<PairBank> p##5(source[slot][5]);hls::kind<PairBank> p##6(source[slot][6]);hls::kind<PairBank> p##7(source[slot][7]);hls::kind<PairBank> p##8(source[slot][8]);hls::kind<PairBank> p##9(source[slot][9]);hls::kind<PairBank> p##10(source[slot][10]);hls::kind<PairBank> p##11(source[slot][11]);hls::kind<PairBank> p##12(source[slot][12]);hls::kind<PairBank> p##13(source[slot][13]);hls::kind<PairBank> p##14(source[slot][14]);hls::kind<PairBank> p##15(source[slot][15]);
#define C207_SLOT_OP(p,op) (p##0).op();(p##1).op();(p##2).op();(p##3).op();(p##4).op();(p##5).op();(p##6).op();(p##7).op();(p##8).op();(p##9).op();(p##10).op();(p##11).op();(p##12).op();(p##13).op();(p##14).op();(p##15).op();

static ap_uint<16> direct_lane16(const c2::bus_t&p,ap_uint<4> lane) {
#pragma HLS INLINE
 // Match the already synthesized bank16 formatter: the compiler knows the
 // entire selector domain before CFG lowering and can share one lane switch.
 switch(unsigned(lane)) {
 case 0:return ap_uint<16>(p.range(15,0));case 1:return ap_uint<16>(p.range(31,16));case 2:return ap_uint<16>(p.range(47,32));case 3:return ap_uint<16>(p.range(63,48));
 case 4:return ap_uint<16>(p.range(79,64));case 5:return ap_uint<16>(p.range(95,80));case 6:return ap_uint<16>(p.range(111,96));case 7:return ap_uint<16>(p.range(127,112));
 case 8:return ap_uint<16>(p.range(143,128));case 9:return ap_uint<16>(p.range(159,144));case 10:return ap_uint<16>(p.range(175,160));case 11:return ap_uint<16>(p.range(191,176));
 case 12:return ap_uint<16>(p.range(207,192));case 13:return ap_uint<16>(p.range(223,208));case 14:return ap_uint<16>(p.range(239,224));case 15:return ap_uint<16>(p.range(255,240));
 }
 return ap_uint<16>(0);
}
static void write_pair_word(unsigned row,unsigned word,c2::bus_t value,C207_BANK_MUT) {
#pragma HLS INLINE
 switch(row&15u) {
 case 0:b0[word]=value;break;case 1:b1[word]=value;break;case 2:b2[word]=value;break;case 3:b3[word]=value;break;
 case 4:b4[word]=value;break;case 5:b5[word]=value;break;case 6:b6[word]=value;break;case 7:b7[word]=value;break;
 case 8:b8[word]=value;break;case 9:b9[word]=value;break;case 10:b10[word]=value;break;case 11:b11[word]=value;break;
 case 12:b12[word]=value;break;case 13:b13[word]=value;break;case 14:b14[word]=value;break;case 15:b15[word]=value;break;
 }
}
static void send_direct_command(unsigned m,unsigned n,unsigned k,cmd_stream&commands) {
#pragma HLS INLINE off
 MacCommand d={m,n,k,true,true,false};commands.write(d);
}
static void emit_direct_banks(unsigned rows,unsigned k,unsigned offset,C207_BANK_CONST,input_stream&output) {
#pragma HLS INLINE off
 for(unsigned panel=0;panel<rows;panel+=32)for(unsigned t=0;t<k;t++)for(unsigned half=0;half<2;half++) {
#pragma HLS PIPELINE II=1
  unsigned word=offset+(panel/16+half)*8+t/16;
  c2::bus_t p=0;
  // fill_pair_slot initializes all Q64/K128/V128 rows, including Q/K tail
  // zeros. Read those valid banks unconditionally so all16 extracts share
  // one lane selector instead of sixteen conditionally entered switches.
#define C207_DIRECT_LANE(l) p.range(16*l+15,16*l)=direct_lane16(b##l[word],ap_uint<4>(t));
  C207_DIRECT_LANE(0) C207_DIRECT_LANE(1) C207_DIRECT_LANE(2) C207_DIRECT_LANE(3)
  C207_DIRECT_LANE(4) C207_DIRECT_LANE(5) C207_DIRECT_LANE(6) C207_DIRECT_LANE(7)
  C207_DIRECT_LANE(8) C207_DIRECT_LANE(9) C207_DIRECT_LANE(10) C207_DIRECT_LANE(11)
  C207_DIRECT_LANE(12) C207_DIRECT_LANE(13) C207_DIRECT_LANE(14) C207_DIRECT_LANE(15)
#undef C207_DIRECT_LANE
  output.write(p);
 }
}
static void feed_qk_banks(unsigned rows,unsigned columns,C207_BANK_CONST,input_stream&ain,input_stream&bin) {
#pragma HLS INLINE off
 // The RTL service accepts either input independently. Sequential delivery
 // saves a second bank read port/formatter; full Q64 costs512 extra load beats
 // versus simultaneous A/B streams. No mathematical operation is changed.
 emit_direct_banks(rows,128,0,C207_BANK_ARGS,ain);
 emit_direct_banks(columns,128,PAIR_K_OFFSET,C207_BANK_ARGS,bin);
}
using PackedProbability=c2::bus_t[16][PAIR_Q_BANK_WORDS];
static void feed_pv_banks(unsigned rows,unsigned k,const PackedProbability&p,C207_BANK_CONST,
 input_stream&ain,input_stream&bin) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
 for(unsigned panel=0;panel<rows;panel+=32)for(unsigned t=0;t<k;t++)for(unsigned half=0;half<2;half++) {
#pragma HLS PIPELINE II=1
  c2::bus_t value=0;unsigned word=(panel/16+half)*8+t/16;
  for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
   // consume_probability_pair explicitly initializes the partial-panel tail.
   value.range(16*l+15,16*l)=direct_lane16(p[l][word],ap_uint<4>(t));
  }
  ain.write(value);
 }
 emit_direct_banks(128,k,PAIR_V_OFFSET,C207_BANK_ARGS,bin);
}
static Meta context_meta(const Meta&pair,unsigned context) {
#pragma HLS INLINE
 Meta m=pair;m.rb+=context*QR;m.mm=minv(QR,pair.mm-context*QR);m.context=context;return m;
}
static void receive_qk_direct(const Meta&pair,unsigned columns,const PairScale&scale,
 result_stream&results,ScoreBlocks&scores,MetaStream&metadata) {
#pragma HLS INLINE off
 unsigned contexts=(pair.mm+QR-1)/QR;
 for(unsigned ctx=0;ctx<contexts;ctx++) {
  Meta m=context_meta(pair,ctx);
  {
   hls::write_lock<ScoreBlock> out(scores);
   for(unsigned r=0;r<m.mm;r++)for(unsigned w=0;w<8;w++) {
#pragma HLS PIPELINE II=1
    ap_uint<768> p=w<columns/16?results.read():ap_uint<768>(0);
    for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
     if(w*16+l>=pair.nn)p.range(48*l+47,48*l)=0;
    }
    out[r*8+w]=p;
   }
   for(unsigned w=0;w<(QR+128+23)/24;w++)out[DOT_WORDS+w]=0;
   for(unsigned i=0;i<QR+128;i++) {
    unsigned source=i<QR?ctx*QR+i:PAIR_Q_ROWS+i-QR;
    put_float(out[DOT_WORDS+i/24],i%24,get_float(scale[source/8],source%8));
   }
  }
  metadata.write(m);
 }
}
static void receive_pv_direct(const Meta&pair,const PairScale&scale,const c2::bus_t pscale[2][2*QR/8],
 result_stream&results,UpdateBlocks&updates,MetaStream&metadata) {
#pragma HLS INLINE off
 unsigned contexts=(pair.mm+QR-1)/QR;
 for(unsigned ctx=0;ctx<contexts;ctx++) {
  Meta m=context_meta(pair,ctx);
  {
   hls::write_lock<UpdateBlock> out(updates);
   for(unsigned r=0;r<m.mm;r++)for(unsigned w=0;w<8;w++) {
#pragma HLS PIPELINE II=1
    out[r*8+w]=results.read();
   }
   for(unsigned w=0;w<(2*QR+128+23)/24;w++)out[DOT_WORDS+w]=0;
   for(unsigned i=0;i<2*QR;i++)put_float(out[DOT_WORDS+i/24],i%24,get_float(pscale[ctx][i/8],i%8));
   for(unsigned d=0;d<128;d++) {
    unsigned source=PAIR_Q_ROWS+128+d;
    put_float(out[DOT_WORDS+(2*QR+d)/24],(2*QR+d)%24,get_float(scale[source/8],source%8));
   }
  }
  metadata.write(m);
 }
}
#ifndef __SYNTHESIS__
static void run_native_direct_model(cmd_stream&commands,input_stream&ain,input_stream&bin,result_stream&co) {
 MacCommand stop={0,0,0,false,false,true};commands.write(stop);
 ::c204::single_mac_service(commands,ain,bin,co);
 if(!commands.empty()||!ain.empty()||!bin.empty())throw std::runtime_error("Direct MAC input cardinality mismatch");
}
#endif
static void qk_direct_exchange(const Meta&pair,unsigned columns,const PairScale&scale,C207_BANK_CONST,
 cmd_stream&cmd,input_stream&ain,input_stream&bin,result_stream&co,ScoreBlocks&scores,MetaStream&metadata) {
#pragma HLS INLINE off
#ifdef __SYNTHESIS__
#pragma HLS DATAFLOW
 send_direct_command(pair.mm,columns,128,cmd);
 feed_qk_banks(pair.mm,columns,C207_BANK_ARGS,ain,bin);
 receive_qk_direct(pair,columns,scale,co,scores,metadata);
#else
 cmd_stream nc;input_stream na,nb;result_stream nr;
 send_direct_command(pair.mm,columns,128,nc);feed_qk_banks(pair.mm,columns,C207_BANK_ARGS,na,nb);
 run_native_direct_model(nc,na,nb,nr);receive_qk_direct(pair,columns,scale,nr,scores,metadata);
 if(!nr.empty())throw std::runtime_error("QK result cardinality mismatch");
#endif
}
static void pv_direct_exchange(const Meta&pair,unsigned inner,const PairScale&scale,const PackedProbability&p,
 const c2::bus_t pscale[2][2*QR/8],C207_BANK_CONST,cmd_stream&cmd,input_stream&ain,input_stream&bin,
 result_stream&co,UpdateBlocks&updates,MetaStream&metadata) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
#ifdef __SYNTHESIS__
#pragma HLS DATAFLOW
 send_direct_command(pair.mm,128,inner,cmd);
 feed_pv_banks(pair.mm,inner,p,C207_BANK_ARGS,ain,bin);
 receive_pv_direct(pair,scale,pscale,co,updates,metadata);
#else
 cmd_stream nc;input_stream na,nb;result_stream nr;
 send_direct_command(pair.mm,128,inner,nc);feed_pv_banks(pair.mm,inner,p,C207_BANK_ARGS,na,nb);
 run_native_direct_model(nc,na,nb,nr);receive_pv_direct(pair,scale,pscale,nr,updates,metadata);
 if(!nr.empty())throw std::runtime_error("PV result cardinality mismatch");
#endif
}

static void fill_pair_slot(c2::bus_t*b,const c2::Job&j,unsigned nb,const Meta&m,
 const c2::bus_t qcache[PAIR_Q_ROWS*8],const float qm[PAIR_Q_ROWS],const float km[128],const float vm[128],
 PairScale&scales,C207_BANK_MUT) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=qm cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=km cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vm cyclic factor=8
 for(unsigned w=0;w<PAIR_Q_ROWS*8;w++) {
#pragma HLS PIPELINE II=1
  unsigned r=w/8;write_pair_word(r,(r/16)*8+(w&7u),qcache[w],C207_BANK_ARGS);
 }
 for(unsigned w=0;w<KW;w++) {
#pragma HLS PIPELINE II=1
  unsigned r=w/8;c2::bus_t p=r<m.nn?b[(j.b>>5)+((uint64_t)(m.kb+r)*j.lb+m.h*128)/16+(w&7u)]:c2::bus_t(0);
  write_pair_word(r,PAIR_K_OFFSET+(r/16)*8+(w&7u),p,C207_BANK_ARGS);
 }
 for(unsigned w=0;w<VW;w++) {
#pragma HLS PIPELINE II=1
  unsigned r=w/8;c2::bus_t p=b[(j.s2>>5)+(uint64_t)(m.h*nb+m.kb/128)*VW+w];
  write_pair_word(r,PAIR_V_OFFSET+(r/16)*8+(w&7u),p,C207_BANK_ARGS);
 }
 for(unsigned w=0;w<PAIR_SCALE_WORDS;w++) {
#pragma HLS PIPELINE II=1
  c2::bus_t p=0;
  for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
   unsigned index=w*8+l;float v=index<PAIR_Q_ROWS?qm[index]:index<PAIR_Q_ROWS+128?km[index-PAIR_Q_ROWS]:vm[index-PAIR_Q_ROWS-128];
   put_float(p,l,v);
  }
  scales[w]=p;
 }
}
static void load_pairs_direct(c2::bus_t*a,c2::bus_t*b,c2::bus_t*x,const c2::Job&j,unsigned stored,
 PairBanks&banks,PairScales&scales,MetaStream&metadata) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=banks complete dim=2
#pragma HLS ARRAY_PARTITION variable=scales complete dim=1
 using namespace c2;
 bus_t qcache[PAIR_Q_ROWS*8];float qm[PAIR_Q_ROWS],km[128],vm[128],row[128];
#pragma HLS BIND_STORAGE variable=qcache type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=qm cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=km cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vm cyclic factor=8
 C207_DECLARE_SLOT(w0,write_buf,banks,0) C207_DECLARE_SLOT(w1,write_buf,banks,1)
 hls::write_buf<PairScale> s0(scales[0]),s1(scales[1]);
 unsigned nb=blocks(stored),hs=head_stride(j.heads);
 for(unsigned h=0;h<j.heads;h++)for(unsigned base=0;base<j.m;base+=PAIR_Q_ROWS) {
  unsigned mm=minv(PAIR_Q_ROWS,j.m-base);
  for(unsigned r=0;r<PAIR_Q_ROWS;r++) {
   if(r<mm) {
    ap_uint<31> maxima[8];
#pragma HLS ARRAY_PARTITION variable=maxima complete
    for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
     maxima[l]=0;
    }
    for(unsigned d=0;d<128;d+=8) {
#pragma HLS PIPELINE II=1
     bus_t p=a[(j.a>>5)+((uint64_t)(base+r)*j.la+h*128+d)/8];
     for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
      float f=get_float(p,l);row[d+l]=f;ap_uint<31> v=c204::attention_absbits(f);
      maxima[l]=v>maxima[l]?v:maxima[l];
     }
    }
    float mx=c204::attention_absmax8(maxima);qm[r]=mx==0?1.f:mx/32760.f;float inverse=1.f/qm[r];
    for(unsigned d=0;d<128;d+=16) {
#pragma HLS PIPELINE II=1
     bus_t p=0;
     for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
      p.range(16*l+15,16*l)=ap_uint<16>(quant(row[d+l],inverse));
     }
     qcache[r*8+d/16]=p;
    }
   } else {
    qm[r]=1.f;for(unsigned w=0;w<8;w++)qcache[r*8+w]=0;
   }
  }
  for(unsigned kb=0;kb<stored;kb+=128) {
   unsigned nn=minv(128,stored-kb);
   for(unsigned n=0;n<128;n++) {
#pragma HLS PIPELINE II=1
    uint64_t address=j.s0+4ull*((uint64_t)(kb+n)*hs+h);
    km[n]=n<nn?get_float(x[address>>5],unsigned((address>>2)&7u)):1.f;
   }
   if((j.s1&31u)==0) {
    for(unsigned w=0;w<16;w++) {
#pragma HLS PIPELINE II=1
     bus_t p=x[(j.s1>>5)+(uint64_t)(h*nb+kb/128)*16+w];
     for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
      vm[w*8+l]=get_float(p,l);
     }
    }
   } else for(unsigned d=0;d<128;d++)vm[d]=readf(x,j.s1,(h*nb+kb/128)*128+d);
   Meta m={h,base,kb,mm,nn,0,kb==0,kb+nn==stored};
   if((kb/128)&1u) {
    C207_SLOT_OP(w1,acquire) s1.acquire();
    fill_pair_slot(b,j,nb,m,qcache,qm,km,vm,s1,C207_SLOT_ARGS(w1));
    s1.release();C207_SLOT_OP(w1,release)
   } else {
    C207_SLOT_OP(w0,acquire) s0.acquire();
    fill_pair_slot(b,j,nb,m,qcache,qm,km,vm,s0,C207_SLOT_ARGS(w0));
    s0.release();C207_SLOT_OP(w0,release)
   }
   metadata.write(m);
  }
 }
}
static void consume_probability_pair(const Meta&pair,ProbBlocks&probability,MetaStream&metadata,
 PackedProbability&p,c2::bus_t pscale[2][2*QR/8]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
 for(unsigned ctx=0;ctx<(pair.mm+QR-1)/QR;ctx++) {
  Meta m=metadata.read();hls::read_lock<ProbBlock> input(probability);
  unsigned inner=(m.nn+15)&~15u;
  for(unsigned r=0;r<m.mm;r++)for(unsigned w=0;w<inner/16;w++) {
#pragma HLS PIPELINE II=1
   unsigned row=ctx*QR+r;p[row&15u][(row/16)*8+w]=input[r*8+w];
  }
  for(unsigned w=0;w<2*QR/8;w++)pscale[ctx][w]=input[PROB_SCALE+w];
 }
 // The service loads complete32-row panels. Keep inactive rows initialized
 // before the branch-free PV feeder reads them; this changes no active value
 // or FP operation and adds at most31*8 packed writes only on a row tail.
 unsigned inner=(pair.nn+15)&~15u,padded_rows=(pair.mm+31)&~31u;
 C207_ZERO_PROB_PANEL_TAIL:for(unsigned row=pair.mm;row<padded_rows;row++)for(unsigned w=0;w<inner/16;w++) {
#pragma HLS PIPELINE II=1
  p[row&15u][(row/16)*8+w]=0;
 }
}
static void mac_broker_direct(const c2::Job&j,unsigned stored,PairBanks&banks,PairScales&scales,MetaStream&kin,
 ScoreBlocks&scores,MetaStream&sout,ProbBlocks&probability,MetaStream&pin,
 UpdateBlocks&updates,MetaStream&uout,cmd_stream&cmd,input_stream&ain,input_stream&bin,result_stream&co,unsigned&tiles) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=banks complete dim=2
#pragma HLS ARRAY_PARTITION variable=scales complete dim=1
 PackedProbability p; c2::bus_t pscale[2][2*QR/8];Meta issued[2];
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
#pragma HLS BIND_STORAGE variable=p type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=pscale type=ram_2p impl=bram
 C207_DECLARE_SLOT(r0,read_buf,banks,0) C207_DECLARE_SLOT(r1,read_buf,banks,1)
 hls::read_buf<PairScale> s0(scales[0]),s1(scales[1]);
 tiles=0;
 for(unsigned h=0;h<j.heads;h++)for(unsigned base=0;base<j.m;base+=PAIR_Q_ROWS) {
  unsigned keyblocks=blocks(stored);
  for(unsigned tick=0;tick<keyblocks+1;tick++) {
   if(tick<keyblocks) {
    Meta m=kin.read();unsigned padded=((m.nn+c2::PO-1)/c2::PO)*c2::PO;
    if(tick&1u) {
     C207_SLOT_OP(r1,acquire) s1.acquire();issued[1]=m;
     qk_direct_exchange(m,padded,s1,C207_SLOT_ARGS(r1),cmd,ain,bin,co,scores,sout);
    } else {
     C207_SLOT_OP(r0,acquire) s0.acquire();issued[0]=m;
     qk_direct_exchange(m,padded,s0,C207_SLOT_ARGS(r0),cmd,ain,bin,co,scores,sout);
    }
    tiles++;
   }
   if(tick>0) {
    unsigned slot=(tick-1)&1u;Meta m=issued[slot];
    consume_probability_pair(m,probability,pin,p,pscale);
    unsigned inner=(m.nn+15)&~15u;
    if(slot) {
     pv_direct_exchange(m,inner,s1,p,pscale,C207_SLOT_ARGS(r1),cmd,ain,bin,co,updates,uout);
     s1.release();C207_SLOT_OP(r1,release)
    } else {
     pv_direct_exchange(m,inner,s0,p,pscale,C207_SLOT_ARGS(r0),cmd,ain,bin,co,updates,uout);
     s0.release();C207_SLOT_OP(r0,release)
    }
    tiles++;
   }
  }
 }
}
