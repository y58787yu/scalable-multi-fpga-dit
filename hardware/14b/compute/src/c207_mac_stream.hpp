#pragma once
#if defined(COSMOS_C207) && !defined(EXPLICIT_ACQUIRE_RELEASE)
#define EXPLICIT_ACQUIRE_RELEASE
#endif
#include <hls_streamofblocks.h>
// 16 narrow skew banks, 32 scalar addresses per bank and two bounded slots.
// bank=(row+column)&15, address=row: both row writes and column reads use
// exactly one 16-bit access per bank per cycle, not a 256-bit shallow PIPO.
namespace c207mac {
using PanelBlock=c2::i16[32];
using panel_blocks=hls::stream_of_blocks<PanelBlock,2>[16];
// Four fixed two-way stages; no dynamic wide slice or bank-index crossbar.
// Low word bits are lane 0. Left r maps input lane c to bank (r+c)&15.
static c2::bus_t rotate_left_lanes(const c2::bus_t &word,ap_uint<4> lanes) {
#pragma HLS INLINE
    c2::bus_t a=lanes[0]?c2::bus_t((word<<16)|(word>>240)):word;
    c2::bus_t b=lanes[1]?c2::bus_t((a<<32)|(a>>224)):a;
    c2::bus_t c=lanes[2]?c2::bus_t((b<<64)|(b>>192)):b;
    return lanes[3]?c2::bus_t((c<<128)|(c>>128)):c;
}
static c2::bus_t rotate_right_lanes(const c2::bus_t &word,ap_uint<4> lanes) {
#pragma HLS INLINE
    c2::bus_t a=lanes[0]?c2::bus_t((word>>16)|(word<<240)):word;
    c2::bus_t b=lanes[1]?c2::bus_t((a>>32)|(a<<224)):a;
    c2::bus_t c=lanes[2]?c2::bus_t((b>>64)|(b<<192)):b;
    return lanes[3]?c2::bus_t((c>>128)|(c<<128)):c;
}
static void transpose_panels(panel_blocks &panels,unsigned rows,unsigned k,input_stream &out) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
    for(unsigned rp=0;rp<rows;rp+=32)for(unsigned kw=0;kw<k;kw+=16) {
        hls::read_lock<PanelBlock> b0(panels[0]);
        hls::read_lock<PanelBlock> b1(panels[1]);
        hls::read_lock<PanelBlock> b2(panels[2]);
        hls::read_lock<PanelBlock> b3(panels[3]);
        hls::read_lock<PanelBlock> b4(panels[4]);
        hls::read_lock<PanelBlock> b5(panels[5]);
        hls::read_lock<PanelBlock> b6(panels[6]);
        hls::read_lock<PanelBlock> b7(panels[7]);
        hls::read_lock<PanelBlock> b8(panels[8]);
        hls::read_lock<PanelBlock> b9(panels[9]);
        hls::read_lock<PanelBlock> b10(panels[10]);
        hls::read_lock<PanelBlock> b11(panels[11]);
        hls::read_lock<PanelBlock> b12(panels[12]);
        hls::read_lock<PanelBlock> b13(panels[13]);
        hls::read_lock<PanelBlock> b14(panels[14]);
        hls::read_lock<PanelBlock> b15(panels[15]);
        for(unsigned t=0;t<16&&kw+t<k;t++)for(unsigned half=0;half<2;half++) {
#pragma HLS PIPELINE II=1
            // Every bank is static; only its five-bit scalar address varies.
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
            out.write(rotate_right_lanes(p,ap_uint<4>(t)));
        }
    }
}
template<unsigned ROWS,unsigned KS>
static void load_scratch_panels(const c2::i16 data[ROWS][KS],unsigned rows,unsigned k,panel_blocks &panels) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=data cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
    for(unsigned rp=0;rp<rows;rp+=32)for(unsigned kw=0;kw<k;kw+=16) {
        hls::write_lock<PanelBlock> b0(panels[0]);
        hls::write_lock<PanelBlock> b1(panels[1]);
        hls::write_lock<PanelBlock> b2(panels[2]);
        hls::write_lock<PanelBlock> b3(panels[3]);
        hls::write_lock<PanelBlock> b4(panels[4]);
        hls::write_lock<PanelBlock> b5(panels[5]);
        hls::write_lock<PanelBlock> b6(panels[6]);
        hls::write_lock<PanelBlock> b7(panels[7]);
        hls::write_lock<PanelBlock> b8(panels[8]);
        hls::write_lock<PanelBlock> b9(panels[9]);
        hls::write_lock<PanelBlock> b10(panels[10]);
        hls::write_lock<PanelBlock> b11(panels[11]);
        hls::write_lock<PanelBlock> b12(panels[12]);
        hls::write_lock<PanelBlock> b13(panels[13]);
        hls::write_lock<PanelBlock> b14(panels[14]);
        hls::write_lock<PanelBlock> b15(panels[15]);
        for(unsigned r=0;r<32;r++) {
#pragma HLS PIPELINE II=1
            c2::bus_t p=0;
            for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
                c2::i16 v=(rp+r<rows&&kw+l<k)?data[rp+r][kw+l]:c2::i16(0);
                p.range(16*l+15,16*l)=ap_uint<16>(v);
            }
            // All 32 addresses, including padded rows/columns, are written.
            c2::bus_t skew=rotate_left_lanes(p,ap_uint<4>(r));
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
template<unsigned ROWS,unsigned KS>
static void load_packed_panels(const c2::bus_t data[ROWS*(KS/16)],unsigned rows,unsigned k,panel_blocks &panels) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
    for(unsigned rp=0;rp<rows;rp+=32)for(unsigned kw=0;kw<k;kw+=16) {
        hls::write_lock<PanelBlock> b0(panels[0]);
        hls::write_lock<PanelBlock> b1(panels[1]);
        hls::write_lock<PanelBlock> b2(panels[2]);
        hls::write_lock<PanelBlock> b3(panels[3]);
        hls::write_lock<PanelBlock> b4(panels[4]);
        hls::write_lock<PanelBlock> b5(panels[5]);
        hls::write_lock<PanelBlock> b6(panels[6]);
        hls::write_lock<PanelBlock> b7(panels[7]);
        hls::write_lock<PanelBlock> b8(panels[8]);
        hls::write_lock<PanelBlock> b9(panels[9]);
        hls::write_lock<PanelBlock> b10(panels[10]);
        hls::write_lock<PanelBlock> b11(panels[11]);
        hls::write_lock<PanelBlock> b12(panels[12]);
        hls::write_lock<PanelBlock> b13(panels[13]);
        hls::write_lock<PanelBlock> b14(panels[14]);
        hls::write_lock<PanelBlock> b15(panels[15]);
        for(unsigned r=0;r<32;r++) {
#pragma HLS PIPELINE II=1
            c2::bus_t p=rp+r<rows?data[(rp+r)*(KS/16)+kw/16]:c2::bus_t(0);
            for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
                if(kw+l>=k)p.range(16*l+15,16*l)=0;
            }
            // All 32 addresses, including padded rows/columns, are written.
            c2::bus_t skew=rotate_left_lanes(p,ap_uint<4>(r));
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
template<unsigned ROWS,unsigned KS>
static void emit_panel(const c2::i16 data[ROWS][KS],unsigned rows,unsigned k,input_stream &out) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    panel_blocks panels;
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
#pragma HLS BIND_STORAGE variable=panels type=ram_2p impl=lutram
    load_scratch_panels<ROWS,KS>(data,rows,k,panels);
    transpose_panels(panels,rows,k,out);
}
template<unsigned ROWS,unsigned KS>
static void emit_packed_panel(const c2::bus_t data[ROWS*(KS/16)],unsigned rows,unsigned k,input_stream &out) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    panel_blocks panels;
#pragma HLS ARRAY_PARTITION variable=panels complete dim=1
#pragma HLS BIND_STORAGE variable=panels type=ram_2p impl=lutram
    load_packed_panels<ROWS,KS>(data,rows,k,panels);
    transpose_panels(panels,rows,k,out);
}
template<unsigned ROWS,unsigned COLS>
static void collect_result(c2::i48 c[ROWS][COLS],unsigned m,unsigned n,
                           bool commit,result_stream &results) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=c cyclic factor=16 dim=2
    if(commit)for(unsigned r=0;r<m;r++)for(unsigned nb=0;nb<n;nb+=16) {
#pragma HLS PIPELINE II=1
        ap_uint<768> p=results.read();
        for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
            if(nb+l<n)c[r][nb+l]=c2::i48(p.range(48*l+47,48*l));
        }
    }
}
template<unsigned AR,unsigned BC,unsigned KS>
static void tile_exchange(c2::i16 a[AR][KS],c2::i16 b[BC][KS],
                          c2::i48 c[AR][BC],unsigned m,unsigned n,unsigned k,
                          bool commit,input_stream &ain,input_stream &bin,
                          result_stream &results) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    emit_panel<AR,KS>(a,m,k,ain);
    emit_panel<BC,KS>(b,n,k,bin);
    collect_result<AR,BC>(c,m,n,commit,results);
}
}
