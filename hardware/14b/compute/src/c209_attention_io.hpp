#pragma once
// Borrow the existing payload and transpose through narrow skew banks.
// Each QK/PV path owns 16 banks x 32 INT16 addresses x two bounded slots.
// The A-then-B panel32/K/half16 ABI and independent QK/PV pipelines are unchanged.
using C209PanelBlock=c2::i16[32];
using C209Panels=hls::stream_of_blocks<C209PanelBlock,2>[16];

// Keep the attention-only fixture independent of the GEMM formatter header.
// Four fixed two-way stages; no dynamic wide slice or bank-index crossbar.
// Low word bits are lane 0. Left r maps input lane c to bank (r+c)&15.
static c2::bus_t c209_rotate_left_lanes(const c2::bus_t &word,ap_uint<4> lanes) {
#pragma HLS INLINE
    c2::bus_t a=lanes[0]?c2::bus_t((word<<16)|(word>>240)):word;
    c2::bus_t b=lanes[1]?c2::bus_t((a<<32)|(a>>224)):a;
    c2::bus_t c=lanes[2]?c2::bus_t((b<<64)|(b>>192)):b;
    return lanes[3]?c2::bus_t((c<<128)|(c>>128)):c;
}
static c2::bus_t c209_rotate_right_lanes(const c2::bus_t &word,ap_uint<4> lanes) {
#pragma HLS INLINE
    c2::bus_t a=lanes[0]?c2::bus_t((word>>16)|(word<<240)):word;
    c2::bus_t b=lanes[1]?c2::bus_t((a>>32)|(a<<224)):a;
    c2::bus_t c=lanes[2]?c2::bus_t((b>>64)|(b<<192)):b;
    return lanes[3]?c2::bus_t((c>>128)|(c<<128)):c;
}

static void c209_qk_load_panels(unsigned rows,unsigned inner,
 const KVBlock&data,C209Panels&panels) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
 for(unsigned operand=0;operand<2;operand++) {
  unsigned matrix_rows=operand?128:rows;
  for(unsigned rp=0;rp<matrix_rows;rp+=32)for(unsigned kw=0;kw<inner;kw+=16) {
   hls::write_lock<C209PanelBlock> b0(panels[0]);
   hls::write_lock<C209PanelBlock> b1(panels[1]);
   hls::write_lock<C209PanelBlock> b2(panels[2]);
   hls::write_lock<C209PanelBlock> b3(panels[3]);
   hls::write_lock<C209PanelBlock> b4(panels[4]);
   hls::write_lock<C209PanelBlock> b5(panels[5]);
   hls::write_lock<C209PanelBlock> b6(panels[6]);
   hls::write_lock<C209PanelBlock> b7(panels[7]);
   hls::write_lock<C209PanelBlock> b8(panels[8]);
   hls::write_lock<C209PanelBlock> b9(panels[9]);
   hls::write_lock<C209PanelBlock> b10(panels[10]);
   hls::write_lock<C209PanelBlock> b11(panels[11]);
   hls::write_lock<C209PanelBlock> b12(panels[12]);
   hls::write_lock<C209PanelBlock> b13(panels[13]);
   hls::write_lock<C209PanelBlock> b14(panels[14]);
   hls::write_lock<C209PanelBlock> b15(panels[15]);
   for(unsigned r=0;r<32;r++) {
#pragma HLS PIPELINE II=1
    c2::bus_t word=0;
    if(rp+r<matrix_rows) {
     unsigned address=(rp+r)*8+kw/16;
     word=data[(operand?QW:0)+address];
    }
    for(unsigned lane=0;lane<16;lane++) {
#pragma HLS UNROLL
     if(kw+lane>=inner)word.range(lane*16+15,lane*16)=0;
    }
    // Explicit static banks: rotate once, then write sixteen literal lanes.
    c2::bus_t skew=c209_rotate_left_lanes(word,ap_uint<4>(r));
    b0[r]=c2::i16(skew.range(15,0));
    b1[r]=c2::i16(skew.range(31,16));
    b2[r]=c2::i16(skew.range(47,32));
    b3[r]=c2::i16(skew.range(63,48));
    b4[r]=c2::i16(skew.range(79,64));
    b5[r]=c2::i16(skew.range(95,80));
    b6[r]=c2::i16(skew.range(111,96));
    b7[r]=c2::i16(skew.range(127,112));
    b8[r]=c2::i16(skew.range(143,128));
    b9[r]=c2::i16(skew.range(159,144));
    b10[r]=c2::i16(skew.range(175,160));
    b11[r]=c2::i16(skew.range(191,176));
    b12[r]=c2::i16(skew.range(207,192));
    b13[r]=c2::i16(skew.range(223,208));
    b14[r]=c2::i16(skew.range(239,224));
    b15[r]=c2::i16(skew.range(255,240));
   }
  }
 }
}
static void c209_pv_load_panels(unsigned rows,unsigned inner,
 const ProbBlock&probability,const C209VBlock&values,C209Panels&panels) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
 for(unsigned operand=0;operand<2;operand++) {
  unsigned matrix_rows=operand?128:rows;
  for(unsigned rp=0;rp<matrix_rows;rp+=32)for(unsigned kw=0;kw<inner;kw+=16) {
   hls::write_lock<C209PanelBlock> b0(panels[0]);
   hls::write_lock<C209PanelBlock> b1(panels[1]);
   hls::write_lock<C209PanelBlock> b2(panels[2]);
   hls::write_lock<C209PanelBlock> b3(panels[3]);
   hls::write_lock<C209PanelBlock> b4(panels[4]);
   hls::write_lock<C209PanelBlock> b5(panels[5]);
   hls::write_lock<C209PanelBlock> b6(panels[6]);
   hls::write_lock<C209PanelBlock> b7(panels[7]);
   hls::write_lock<C209PanelBlock> b8(panels[8]);
   hls::write_lock<C209PanelBlock> b9(panels[9]);
   hls::write_lock<C209PanelBlock> b10(panels[10]);
   hls::write_lock<C209PanelBlock> b11(panels[11]);
   hls::write_lock<C209PanelBlock> b12(panels[12]);
   hls::write_lock<C209PanelBlock> b13(panels[13]);
   hls::write_lock<C209PanelBlock> b14(panels[14]);
   hls::write_lock<C209PanelBlock> b15(panels[15]);
   for(unsigned r=0;r<32;r++) {
#pragma HLS PIPELINE II=1
    c2::bus_t word=0;
    if(rp+r<matrix_rows) {
     unsigned address=(rp+r)*8+kw/16;
     word=operand?values[address]:probability[address];
    }
    for(unsigned lane=0;lane<16;lane++) {
#pragma HLS UNROLL
     if(kw+lane>=inner)word.range(lane*16+15,lane*16)=0;
    }
    // Explicit static banks: rotate once, then write sixteen literal lanes.
    c2::bus_t skew=c209_rotate_left_lanes(word,ap_uint<4>(r));
    b0[r]=c2::i16(skew.range(15,0));
    b1[r]=c2::i16(skew.range(31,16));
    b2[r]=c2::i16(skew.range(47,32));
    b3[r]=c2::i16(skew.range(63,48));
    b4[r]=c2::i16(skew.range(79,64));
    b5[r]=c2::i16(skew.range(95,80));
    b6[r]=c2::i16(skew.range(111,96));
    b7[r]=c2::i16(skew.range(127,112));
    b8[r]=c2::i16(skew.range(143,128));
    b9[r]=c2::i16(skew.range(159,144));
    b10[r]=c2::i16(skew.range(175,160));
    b11[r]=c2::i16(skew.range(191,176));
    b12[r]=c2::i16(skew.range(207,192));
    b13[r]=c2::i16(skew.range(223,208));
    b14[r]=c2::i16(skew.range(239,224));
    b15[r]=c2::i16(skew.range(255,240));
   }
  }
 }
}
// Same scalar skew layout as c207mac::transpose_panels. Hold every read lock
// until all output writes finish; backpressure cannot release a live panel.
// The pair wrapper below still shares one instance across sequential A and B.
static void c209_transpose_operand(C209Panels &panels,unsigned rows,unsigned k,input_stream &out) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
    for(unsigned rp=0;rp<rows;rp+=32)for(unsigned kw=0;kw<k;kw+=16) {
        hls::read_lock<C209PanelBlock> b0(panels[0]);
        hls::read_lock<C209PanelBlock> b1(panels[1]);
        hls::read_lock<C209PanelBlock> b2(panels[2]);
        hls::read_lock<C209PanelBlock> b3(panels[3]);
        hls::read_lock<C209PanelBlock> b4(panels[4]);
        hls::read_lock<C209PanelBlock> b5(panels[5]);
        hls::read_lock<C209PanelBlock> b6(panels[6]);
        hls::read_lock<C209PanelBlock> b7(panels[7]);
        hls::read_lock<C209PanelBlock> b8(panels[8]);
        hls::read_lock<C209PanelBlock> b9(panels[9]);
        hls::read_lock<C209PanelBlock> b10(panels[10]);
        hls::read_lock<C209PanelBlock> b11(panels[11]);
        hls::read_lock<C209PanelBlock> b12(panels[12]);
        hls::read_lock<C209PanelBlock> b13(panels[13]);
        hls::read_lock<C209PanelBlock> b14(panels[14]);
        hls::read_lock<C209PanelBlock> b15(panels[15]);
        for(unsigned t=0;t<16&&kw+t<k;t++)for(unsigned half=0;half<2;half++) {
#pragma HLS PIPELINE II=1
            c2::bus_t p=0;
            p.range(15,0)=ap_uint<16>(b0[ap_uint<5>((half<<4)|((0u-t)&15u))]);
            p.range(31,16)=ap_uint<16>(b1[ap_uint<5>((half<<4)|((1u-t)&15u))]);
            p.range(47,32)=ap_uint<16>(b2[ap_uint<5>((half<<4)|((2u-t)&15u))]);
            p.range(63,48)=ap_uint<16>(b3[ap_uint<5>((half<<4)|((3u-t)&15u))]);
            p.range(79,64)=ap_uint<16>(b4[ap_uint<5>((half<<4)|((4u-t)&15u))]);
            p.range(95,80)=ap_uint<16>(b5[ap_uint<5>((half<<4)|((5u-t)&15u))]);
            p.range(111,96)=ap_uint<16>(b6[ap_uint<5>((half<<4)|((6u-t)&15u))]);
            p.range(127,112)=ap_uint<16>(b7[ap_uint<5>((half<<4)|((7u-t)&15u))]);
            p.range(143,128)=ap_uint<16>(b8[ap_uint<5>((half<<4)|((8u-t)&15u))]);
            p.range(159,144)=ap_uint<16>(b9[ap_uint<5>((half<<4)|((9u-t)&15u))]);
            p.range(175,160)=ap_uint<16>(b10[ap_uint<5>((half<<4)|((10u-t)&15u))]);
            p.range(191,176)=ap_uint<16>(b11[ap_uint<5>((half<<4)|((11u-t)&15u))]);
            p.range(207,192)=ap_uint<16>(b12[ap_uint<5>((half<<4)|((12u-t)&15u))]);
            p.range(223,208)=ap_uint<16>(b13[ap_uint<5>((half<<4)|((13u-t)&15u))]);
            p.range(239,224)=ap_uint<16>(b14[ap_uint<5>((half<<4)|((14u-t)&15u))]);
            p.range(255,240)=ap_uint<16>(b15[ap_uint<5>((half<<4)|((15u-t)&15u))]);
            out.write(c209_rotate_right_lanes(p,ap_uint<4>(t)));
        }
    }
}
static void c209_transpose_pair(unsigned rows,unsigned inner,C209Panels&panels,
 input_stream&raw) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION function instances=c209_transpose_operand limit=1
 // Sequential calls share one physical transpose datapath. A and B are still
 // produced in order into raw; the independent router selects their outputs.
 c209_transpose_operand(panels,rows,inner,raw);
 c209_transpose_operand(panels,128,inner,raw);
}
static void c209_route_operands(unsigned rows,unsigned inner,input_stream&raw,
 input_stream&ai,input_stream&bi) {
#pragma HLS INLINE off
 const unsigned a_beats=((rows+31u)/32u)*inner*2u;
 const unsigned b_beats=8u*inner;
 for(unsigned beat=0;beat<a_beats;beat++) {
#pragma HLS PIPELINE II=1
  ai.write(raw.read());
 }
 for(unsigned beat=0;beat<b_beats;beat++) {
#pragma HLS PIPELINE II=1
  bi.write(raw.read());
 }
}
static void c209_qk_inputs(unsigned rows,const KVBlock&data,cmd_stream&cmd,
 input_stream&ai,input_stream&bi) {
#pragma HLS INLINE off
 // Issue must precede panel production: the RTL cannot consume operands until
 // it has accepted a command. Local panel FIFOs are deliberately bounded.
 MacCommand d={rows,128,128,true,true,false};cmd.write(d);
 C209Panels panels;
 input_stream raw("c209_qk_raw");
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
#pragma HLS BIND_STORAGE variable=panels type=ram_2p impl=lutram
#pragma HLS STREAM variable=raw depth=8
#pragma HLS BIND_STORAGE variable=raw type=fifo impl=srl
#ifdef __SYNTHESIS__
#pragma HLS DATAFLOW
 c209_qk_load_panels(rows,128,data,panels);
 c209_transpose_pair(rows,128,panels,raw);
 c209_route_operands(rows,128,raw,ai,bi);
#else
 // Vendor native stream_of_blocks publishes at write_acquire, not release,
 // and has no capacity limit. Concurrent native panel readers can therefore
 // observe an unfinished block. Use an ordered numerical oracle here; only
 // the synthesis DATAFLOW above implements finite concurrent PIPO ownership.
 c209_qk_load_panels(rows,128,data,panels);
 c209_transpose_pair(rows,128,panels,raw);
 c209_route_operands(rows,128,raw,ai,bi);
#endif
}
static void c209_pv_inputs(unsigned rows,unsigned inner,const ProbBlock&probability,
 const C209VBlock&values,cmd_stream&cmd,input_stream&ai,input_stream&bi) {
#pragma HLS INLINE off
 MacCommand d={rows,128,inner,true,true,false};cmd.write(d);
 C209Panels panels;
 input_stream raw("c209_pv_raw");
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
#pragma HLS BIND_STORAGE variable=panels type=ram_2p impl=lutram
#pragma HLS STREAM variable=raw depth=8
#pragma HLS BIND_STORAGE variable=raw type=fifo impl=srl
#ifdef __SYNTHESIS__
#pragma HLS DATAFLOW
 c209_pv_load_panels(rows,inner,probability,values,panels);
 c209_transpose_pair(rows,inner,panels,raw);
 c209_route_operands(rows,inner,raw,ai,bi);
#else
 // Same native-only commit ordering workaround as QK; hardware stays DATAFLOW.
 c209_pv_load_panels(rows,inner,probability,values,panels);
 c209_transpose_pair(rows,inner,panels,raw);
 c209_route_operands(rows,inner,raw,ai,bi);
#endif
}

