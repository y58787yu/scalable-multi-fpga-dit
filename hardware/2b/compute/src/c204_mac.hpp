#pragma once

// Two physically distinct 16x32 engines. They share an activation word and
// own alternating 32-column stripes of each logical 64-column issue group.
// This keeps all 1024 multipliers useful without doubling AXI activation
// traffic. Each engine has its own weight/accumulator banks and state machine.
namespace c204 {
static void distribute_mac(cmd_stream &commands,input_stream &ain,input_stream &bin,
                           cmd_stream &c0,cmd_stream &c1,cmd_stream &meta,
                           input_stream &a0,input_stream &a1,input_stream &b0,input_stream &b1) {
#pragma HLS INLINE off
    bool stop=false;
    while(!stop) {
        MacCommand d=commands.read();stop=d.stop;
        MacCommand local=d;local.n=d.n/2;
        c0.write(local);c1.write(local);meta.write(d);
        if(!stop) {
            for(unsigned n=0;n<d.n;n++)for(unsigned k=0;k<d.k;k+=16) {
#pragma HLS PIPELINE II=1
                bus_t word=bin.read();
                if((n/32)&1)b1.write(word);else b0.write(word);
            }
            for(unsigned nb=0;nb<d.n;nb+=64)
                for(unsigned r=0;r<d.m;r++)for(unsigned k=0;k<d.k;k+=16) {
#pragma HLS PIPELINE II=1
                    bus_t word=ain.read();a0.write(word);a1.write(word);
                }
        }
    }
}

template<unsigned ENGINE>
static void mac_engine(cmd_stream &commands,input_stream &ain,input_stream &bin,result_stream &results) {
#pragma HLS INLINE off
    i16 w[BN/2][BK];i48 c[MAC_ROWS][BN/2];
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=w type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=c cyclic factor=32 dim=2
#pragma HLS BIND_STORAGE variable=c type=ram_2p impl=bram
    // Four fixed shift-feedback registers keep the four-cycle recurrence
    // without a variable context/phase read-write crossbar. The first four
    // words of each row replace old values; short rows mask old tail lanes.
    i48 partial[4][32];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=0
    bool stop=false;
    while(!stop) {
        MacCommand d=commands.read();stop=d.stop;
        if(!stop) {
            if(d.reset)for(unsigned r=0;r<d.m;r++)for(unsigned n=0;n<d.n;n+=32) {
#pragma HLS PIPELINE II=1
                for(unsigned l=0;l<32;l++) {
#pragma HLS UNROLL
                    c[r][n+l]=0;
                }
            }
            for(unsigned n=0;n<d.n;n++)for(unsigned k=0;k<d.k;k+=16) {
#pragma HLS PIPELINE II=1
                bus_t packed=bin.read();
                for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
                    w[n][k+l]=i16(packed.range(16*l+15,16*l));
                }
            }
            const unsigned words=d.k/16;
            for(unsigned nb=0;nb<d.n;nb+=32) {
                unsigned r=0,t=0;
                C204_CONTINUOUS_ROWS:for(unsigned step=0;step<d.m*words;step++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=c inter false
                    bus_t packed=ain.read();i16 av[16];
#pragma HLS ARRAY_PARTITION variable=av complete
                    for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
                        av[l]=i16(packed.range(16*l+15,16*l));
                    }
                    for(unsigned n=0;n<32;n++) {
#pragma HLS UNROLL
                        ap_int<32> product[16];ap_int<33> sum8[8];ap_int<34> sum4[4];ap_int<35> sum2[2];
#pragma HLS ARRAY_PARTITION variable=product complete
#pragma HLS ARRAY_PARTITION variable=sum8 complete
#pragma HLS ARRAY_PARTITION variable=sum4 complete
#pragma HLS ARRAY_PARTITION variable=sum2 complete
                        for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
                            ap_int<32> p=av[l]*w[nb+n][t*16+l];
#pragma HLS BIND_OP variable=p op=mul impl=dsp
                            product[l]=p;
                        }
                        for(unsigned l=0;l<8;l++) {
#pragma HLS UNROLL
                            sum8[l]=ap_int<33>(product[2*l])+product[2*l+1];
                        }
                        for(unsigned l=0;l<4;l++) {
#pragma HLS UNROLL
                            sum4[l]=ap_int<34>(sum8[2*l])+sum8[2*l+1];
                        }
                        for(unsigned l=0;l<2;l++) {
#pragma HLS UNROLL
                            sum2[l]=ap_int<35>(sum4[2*l])+sum4[2*l+1];
                        }
                        ap_int<36> dot=ap_int<36>(sum2[0])+sum2[1];
                        i48 next=(t<4?i48(0):partial[0][n])+i48(dot);
                        if(t+1==words) {
                            i48 a=words>3?partial[1][n]:i48(0);
                            i48 b=words>2?partial[2][n]:i48(0);
                            i48 c_last=words>1?partial[3][n]:i48(0);
                            c[r][nb+n]+=(a+b)+(c_last+next);
                        }
                        partial[0][n]=partial[1][n];
                        partial[1][n]=partial[2][n];
                        partial[2][n]=partial[3][n];
                        partial[3][n]=next;
                    }
                    if(t+1==words){t=0;r++;}else t++;
                }
            }
            if(d.commit)for(unsigned r=0;r<d.m;r++)for(unsigned n=0;n<d.n;n+=16) {
#pragma HLS PIPELINE II=1
                ap_uint<768> packed=0;
                for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
                    packed.range(48*l+47,48*l)=ap_uint<48>(c[r][n+l]);
                }
                results.write(packed);
            }
        }
    }
}

static void merge_mac(cmd_stream &meta,result_stream &r0,result_stream &r1,result_stream &results) {
#pragma HLS INLINE off
    bool stop=false;
    while(!stop) {
        MacCommand d=meta.read();stop=d.stop;
        if(!stop&&d.commit)for(unsigned r=0;r<d.m;r++)for(unsigned n=0;n<d.n;n+=16) {
#pragma HLS PIPELINE II=1
            if((n/32)&1)results.write(r1.read());else results.write(r0.read());
        }
    }
}

static void dual_mac_service(cmd_stream &commands,input_stream &ain,input_stream &bin,result_stream &results) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    cmd_stream c0,c1,meta;input_stream a0,a1,b0,b1;result_stream r0,r1;
#pragma HLS STREAM variable=c0 depth=4
#pragma HLS STREAM variable=c1 depth=4
#pragma HLS STREAM variable=meta depth=8
#pragma HLS STREAM variable=a0 depth=64
#pragma HLS STREAM variable=a1 depth=64
#pragma HLS STREAM variable=b0 depth=64
#pragma HLS STREAM variable=b1 depth=64
#pragma HLS STREAM variable=r0 depth=32
#pragma HLS STREAM variable=r1 depth=32
    distribute_mac(commands,ain,bin,c0,c1,meta,a0,a1,b0,b1);
    mac_engine<0>(c0,a0,b0,r0);
    mac_engine<1>(c1,a1,b1,r1);
    merge_mac(meta,r0,r1,results);
}
}
