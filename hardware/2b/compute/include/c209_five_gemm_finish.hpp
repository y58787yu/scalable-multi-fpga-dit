#pragma once
// Included inside c209five after the five finite local result drains.
// Two local result/scaling regions share one DDR writer and one two-lane
// GELU service. Every invocation covers exactly one physical MAC group:
// neither region is allowed to request the next group while its tail is held.
using RegionalEncoded=hls::stream<ap_uint<128>>;

static void select_three_shards(unsigned count,unsigned pairs_per_shard,
 GemmPairs &a,GemmPairs &b,GemmPairs &c,GemmPairs &selected) {
#pragma HLS INLINE off
 for(unsigned shard=0;shard<count;shard++) {
#pragma HLS LOOP_FLATTEN off
  C209_REGION3_SELECT:for(unsigned pair=0;pair<pairs_per_shard;pair++) {
#pragma HLS PIPELINE II=1
   ap_uint<96> v=shard==0?a.read():shard==1?b.read():c.read();selected.write(v);
  }
 }
}
static void select_two_shards(unsigned count,unsigned pairs_per_shard,
 GemmPairs &a,GemmPairs &b,GemmPairs &selected) {
#pragma HLS INLINE off
 for(unsigned shard=0;shard<count;shard++) {
#pragma HLS LOOP_FLATTEN off
  C209_REGION2_SELECT:for(unsigned pair=0;pair<pairs_per_shard;pair++) {
#pragma HLS PIPELINE II=1
   ap_uint<96> v=shard==0?a.read():b.read();selected.write(v);
  }
 }
}
template<unsigned REGION,unsigned REGION_SHARDS> static void scale_region_pairs(unsigned rows,unsigned n,unsigned count,
 const c209gemmscale::Scale row_scale[TM],const c209gemmscale::Scale column_scale[REGION_SHARDS][TN],
 GemmPairs &selected,c2fast::linear_pair_stream &pairs) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=column_scale complete dim=1
#pragma HLS ARRAY_PARTITION variable=column_scale cyclic factor=2 dim=2
#pragma HLS BIND_STORAGE variable=column_scale type=ram_2p impl=lutram
 for(unsigned local=0;local<count;local++) {
#pragma HLS LOOP_FLATTEN off
  ap_uint<9> row=0;ap_uint<8> col=0;
  C209_REGIONAL_SCALE_PAIRS:for(unsigned word=0;word<rows*(n/2);word++) {
#pragma HLS PIPELINE II=1
   ap_uint<96> values=selected.read();ap_uint<64> packed=0;
   for(unsigned lane=0;lane<2;lane++) {
#pragma HLS UNROLL
    ap_int<48> dot=values.range(48*lane+47,48*lane);
    packed.range(32*lane+31,32*lane)=c209gemmscale::scaled_dot_bits(dot,row_scale[row],column_scale[local][col+lane]);
   }
   pairs.write(packed);col+=2;if(col==n){col=0;row++;}
  }
 }
}
template<unsigned REGION> static void encode_plain_region(unsigned count,unsigned format,bool partial,
 c2fast::linear_pair_stream &input,RegionalEncoded &output) {
#pragma HLS INLINE off
 ap_uint<128> pack=0;
 C209_REGION_ENCODE:for(unsigned pair=0;pair<count;pair++) {
#pragma HLS PIPELINE II=1
  ap_uint<64> value=input.read();
  if(format==F32) {
   // Sensitive F32 output keeps the Y3 half-rounded ordinary output ABI;
   // local-partial F32 remains the production TP reduction format.
   if(!partial)for(unsigned lane=0;lane<2;lane++) {
#pragma HLS UNROLL
    value.range(32*lane+31,32*lane)=fu(c209mp::widen(c209mp::narrow(uf(value.range(32*lane+31,32*lane).to_uint()))));
   }
   if(pair&1u){pack.range(127,64)=value;output.write(pack);}
   else pack.range(63,0)=value;
  } else {
   ap_uint<32> halves=0;
   for(unsigned lane=0;lane<2;lane++) {
#pragma HLS UNROLL
    ap_uint<32> bits=value.range(32*lane+31,32*lane);
    halves.range(16*lane+15,16*lane)=c209mp::bits(c209mp::narrow(uf(bits.to_uint())));
   }
   pack>>=32;pack.range(127,96)=halves;if((pair&3u)==3)output.write(pack);
  }
 }
}
static void merge_regional_gelu(unsigned rows,unsigned n,unsigned a_shards,unsigned b_shards,
 c2fast::linear_pair_stream &a,c2fast::linear_pair_stream &b,
 c2fast::gelu_command_stream &commands,c2fast::nonlinear_pair_stream &output) {
#pragma HLS INLINE off
#ifdef __SYNTHESIS__
 // The unchanged registered publisher counts pairs; columns stays a
 // multiple of8. One command owns the entire finite group, and one ACK.
 c2fast::send_registered_vector_gelu_command(commands,rows,n*(a_shards+b_shards),1);
#endif
 const unsigned b_pairs=rows*(n/2)*b_shards,total=rows*(n/2)*(a_shards+b_shards);
 C209_GELU_REGIONAL_MERGE:for(unsigned pair=0;pair<total;pair++) {
#pragma HLS PIPELINE II=1
  bool second=pair<(b_pairs<<1)&&(pair&1u);
  ap_uint<64> scaled=second?b.read():a.read();
  c2fast::NonlinearPair out;out.valid=3;out.values=0;
  for(unsigned lane=0;lane<2;lane++) {
#pragma HLS UNROLL
   out.values.range(16*lane+15,16*lane)=c209mp::bits(c209mp::narrow(uf(scaled.range(32*lane+31,32*lane).to_uint())));
  }
  output.write(out);
 }
}
static void split_regional_gelu(unsigned a_pairs,unsigned b_pairs,c2fast::pair_output_stream &input,
 RegionalEncoded &a,RegionalEncoded &b) {
#pragma HLS INLINE off
 ap_uint<128> pack[2]={0,0};ap_uint<2> position[2]={0,0};
#pragma HLS ARRAY_PARTITION variable=pack complete
#pragma HLS ARRAY_PARTITION variable=position complete
 C209_GELU_REGIONAL_SPLIT:for(unsigned pair=0;pair<a_pairs+b_pairs;pair++) {
#pragma HLS PIPELINE II=1
  bool second=pair<(b_pairs<<1)&&(pair&1u);unsigned region=second?1u:0u;
  c2fast::pair_word value=input.read();ap_uint<32> stored=0;
  stored.range(15,0)=c209fixed::auxiliary_detail::stored_half(c209mp::word(value.range(15,0)));
  stored.range(31,16)=c209fixed::auxiliary_detail::stored_half(c209mp::word(value.range(31,16)));
  ap_uint<128> next=pack[region]>>32;next.range(127,96)=stored;pack[region]=next;
  if(position[region]==3){if(second)b.write(next);else a.write(next);}
  position[region]++;
 }
}
static void regional_gelu_exchange(unsigned rows,unsigned n,unsigned a_shards,unsigned b_shards,
 c2fast::linear_pair_stream &a,c2fast::linear_pair_stream &b,RegionalEncoded &ao,RegionalEncoded &bo,
 c2fast::gelu_command_stream &gc,c2fast::nonlinear_pair_stream &gi,c2fast::pair_output_stream &go) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(merge_regional_gelu(rows,n,a_shards,b_shards,a,b,gc,gi));
#ifndef __SYNTHESIS__
 C209_FIXED_TASK(c2fast::shared_gelu_pair_math(gi,go,rows*(n/2)*(a_shards+b_shards)));
#endif
 C209_FIXED_TASK(split_regional_gelu(rows*(n/2)*a_shards,rows*(n/2)*b_shards,go,ao,bo));
}
static void regional_plain_exchange(unsigned a_pairs,unsigned b_pairs,unsigned format,bool partial,
 c2fast::linear_pair_stream &a,c2fast::linear_pair_stream &b,RegionalEncoded &ao,RegionalEncoded &bo) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(encode_plain_region<0>(a_pairs,format,partial,a,ao));
 C209_FIXED_TASK(encode_plain_region<1>(b_pairs,format,partial,b,bo));
}
static void encode_two_regions(const FixedJob &j,unsigned rows,unsigned n,unsigned a_shards,unsigned b_shards,
 c2fast::linear_pair_stream &a,c2fast::linear_pair_stream &b,RegionalEncoded &ao,RegionalEncoded &bo,
 c2fast::gelu_command_stream &gc,c2fast::nonlinear_pair_stream &gi,c2fast::pair_output_stream &go,c2fast::gelu_ack_stream &ga) {
#pragma HLS INLINE off
 if(j.op==GEMM_GELU_F16) {
  regional_gelu_exchange(rows,n,a_shards,b_shards,a,b,ao,bo,gc,gi,go);
  c2fast::end_shared_gelu(ga);
 } else regional_plain_exchange(rows*(n/2)*a_shards,rows*(n/2)*b_shards,j.fc,bool(j.flags&LOCAL_PARTIAL),a,b,ao,bo);
}
// Each region completes the packet/address/max bookkeeping locally. The
// only shared writer sees a complete256-bit data beat and its word offset.
// This removes the old dynamic region-indexed half-packet loop (actualII2).
using RegionalWrite=hls::stream<ap_uint<283>>;
using RegionalRowMaximum=hls::stream<ap_uint<16>>;
static WordOffset output_row_elements(unsigned row,unsigned columns) {
#pragma HLS INLINE
 WordOffset r=row;
 return columns==64?WordOffset(r<<6):columns==1024?WordOffset(r<<10):columns==4096?WordOffset(r<<12):WordOffset(r<<11);
}
template<unsigned REGION> static void pack_region_writes(const FixedJob &j,unsigned rb,unsigned nb,
 unsigned rows,unsigned n,unsigned shards,RegionalEncoded &input,RegionalWrite &output,RegionalRowMaximum &row_maxima) {
#pragma HLS INLINE off
 bool wide=j.fc==F32;unsigned per_word=wide?4u:8u;
 unsigned count=rows*(wide?(n>>2):(n>>3))*shards;
 ap_uint<8> col=0;ap_uint<9> row=0;ap_uint<3> shard=REGION==0?0:3;
 ap_uint<128> low=0;ap_uint<16> running=0;
 WordOffset row_base=output_row_elements(rb,j.lc),shard_base=nb+(REGION==0?0u:3u*TN);
 C209_REGION_PACK_WRITE:for(unsigned i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
  ap_uint<128> value=input.read();
  if(j.op==GEMM_GELU_F16)running=half_max_abs8(value,col==0?ap_uint<16>(0):running);
  if(wide?bool(col[2]):bool(col[3])) {
   ap_uint<283> packet=0;packet.range(127,0)=low;packet.range(255,128)=value;
   WordOffset element=row_base+shard_base+WordOffset(col-per_word);
   packet.range(282,256)=wide?WordOffset(element>>3):WordOffset(element>>4);
   output.write(packet);
  } else low=value;
  if(unsigned(col)+per_word==n) {
   if(j.op==GEMM_GELU_F16)row_maxima.write(running);
   col=0;
   if(unsigned(row)+1==rows){row=0;shard++;shard_base+=TN;row_base=output_row_elements(rb,j.lc);}
   else {row++;row_base+=j.lc;}
  } else col+=per_word;
 }
}
// The hot packet loop emits exactly one local maximum per completed GELU
// row. Merge with preceding shards/groups in this finite side actor, whose
// two-cycle ROW service is much faster than the sixteen encoded beats/row.
// Moving the real RAM read/modify/write out of the pack loop eliminates the
// native II2 distance1 constraint without a false DEPENDENCE assertion.
// Non-GELU producers and collectors both transfer zero maximum tokens.
template<unsigned REGION> static void collect_region_maximum(const FixedJob &j,unsigned rows,unsigned shards,
 RegionalRowMaximum &input,ap_uint<16> maximum[TM]) {
#pragma HLS INLINE off
 if(j.op!=GEMM_GELU_F16)return;
 for(unsigned shard=0;shard<shards;shard++) {
#pragma HLS LOOP_FLATTEN off
  C209_REGION_MAXIMUM_ROWS:for(unsigned row=0;row<rows;row++) {
#pragma HLS PIPELINE II=2
   ap_uint<16> local=input.read(),previous=maximum[row];
   maximum[row]=local>previous?local:previous;
  }
 }
}
using RegionalBurstAddress=hls::stream<WordOffset>;
using RegionalBurstData=hls::stream<bus_t>;
static void select_regional_bursts(unsigned a_words,unsigned b_words,RegionalWrite &a,RegionalWrite &b,
 RegionalBurstAddress &addresses,RegionalBurstData &data) {
#pragma HLS INLINE off
 // Four beats divide every admitted F16/F32 row (including N64). This
 // finite alternation is essential: shared GELU alternates A/B pairs, so
 // draining an entire region before the other can deadlock on B's queue.
 unsigned b_bursts=b_words>>2,total_bursts=(a_words+b_words)>>2;
 for(unsigned burst=0;burst<total_bursts;burst++) {
#pragma HLS LOOP_FLATTEN off
  bool second=burst<(b_bursts<<1)&&(burst&1u);
  WordOffset base=0;
  C209_FAIR_REGION_FOUR_BEATS:for(unsigned beat=0;beat<4;beat++) {
#pragma HLS PIPELINE II=1
   ap_uint<283> packet=second?b.read():a.read();
   if(beat==0){base=packet.range(282,256);addresses.write(base);}
#ifndef __SYNTHESIS__
   if(WordOffset(packet.range(282,256))!=WordOffset(base+beat))
    throw "GEMM burst crosses a non-contiguous row or shard";
#endif
   data.write(bus_t(packet.range(255,0)));
  }
 }
}
static void write_regional_bursts(bus_t *output,const FixedJob &j,unsigned bursts,
 RegionalBurstAddress &addresses,RegionalBurstData &data) {
#pragma HLS INLINE off
 // Pipeline complete four-beat transactions, including their B response.
 // The finite pipeline drains every outstanding response before returning.
 C209_BURST_TX_PIPE:for(unsigned burst=0;burst<bursts;burst++) {
#pragma HLS PIPELINE II=4
  WordOffset base=addresses.read();
  C209_DDR_CONTIGUOUS_FOUR_BEATS:for(unsigned beat=0;beat<4;beat++) {
   output[word_address(j.c,WordOffset(base+beat))]=data.read();
  }
 }
}
static void write_two_regions(bus_t *output,const FixedJob &j,unsigned rows,unsigned n,
 unsigned a_shards,unsigned b_shards,RegionalWrite &a,RegionalWrite &b) {
#pragma HLS INLINE off
 unsigned row_words=j.fc==F32?(n>>3):(n>>4);
 unsigned a_words=rows*row_words*a_shards,b_words=rows*row_words*b_shards;
 RegionalBurstAddress addresses;RegionalBurstData data;
#pragma HLS STREAM variable=addresses depth=2
#pragma HLS STREAM variable=data depth=8
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(select_regional_bursts(a_words,b_words,a,b,addresses,data));
 C209_FIXED_TASK(write_regional_bursts(output,j,(a_words+b_words)>>2,addresses,data));
}
static void finish_regional_group(bus_t *output,const FixedJob &j,unsigned rb,unsigned nb,unsigned rows,unsigned n,
 unsigned a_shards,unsigned b_shards,const c209gemmscale::Scale row_scale[TM],
 const c209gemmscale::Scale column_scale_a[3][TN],const c209gemmscale::Scale column_scale_b[2][TN],
 ap_uint<16> maximum_a[TM],ap_uint<16> maximum_b[TM],
 C209_GEMM_PAIR_PARAMS(input),c2fast::gelu_command_stream &gc,c2fast::nonlinear_pair_stream &gi,
 c2fast::pair_output_stream &go,c2fast::gelu_ack_stream &ga) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=column_scale_a complete dim=1
#pragma HLS ARRAY_PARTITION variable=column_scale_a cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=column_scale_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=column_scale_b cyclic factor=2 dim=2
#pragma HLS BIND_STORAGE variable=column_scale_a type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=column_scale_b type=ram_2p impl=lutram
 GemmPairs selected_a,selected_b;c2fast::linear_pair_stream scaled_a,scaled_b;RegionalEncoded encoded_a,encoded_b;RegionalWrite writes_a,writes_b;RegionalRowMaximum maxima_a,maxima_b;
#pragma HLS STREAM variable=selected_a depth=8
#pragma HLS STREAM variable=selected_b depth=8
#pragma HLS STREAM variable=scaled_a depth=16
#pragma HLS STREAM variable=scaled_b depth=16
#pragma HLS STREAM variable=encoded_a depth=8
#pragma HLS STREAM variable=encoded_b depth=8
#pragma HLS STREAM variable=writes_a depth=4
#pragma HLS STREAM variable=writes_b depth=4
#pragma HLS STREAM variable=maxima_a depth=2
#pragma HLS STREAM variable=maxima_b depth=2
#pragma HLS DATAFLOW
 C209_FIXED_GROUP;
 C209_FIXED_TASK(select_three_shards(a_shards,rows*(n/2),input0,input1,input2,selected_a));
 C209_FIXED_TASK(select_two_shards(b_shards,rows*(n/2),input3,input4,selected_b));
 C209_FIXED_TASK((scale_region_pairs<0,3>(rows,n,a_shards,row_scale,column_scale_a,selected_a,scaled_a)));
 C209_FIXED_TASK((scale_region_pairs<1,2>(rows,n,b_shards,row_scale,column_scale_b,selected_b,scaled_b)));
 C209_FIXED_TASK(encode_two_regions(j,rows,n,a_shards,b_shards,scaled_a,scaled_b,encoded_a,encoded_b,gc,gi,go,ga));
 C209_FIXED_TASK(pack_region_writes<0>(j,rb,nb,rows,n,a_shards,encoded_a,writes_a,maxima_a));
 C209_FIXED_TASK(pack_region_writes<1>(j,rb,nb,rows,n,b_shards,encoded_b,writes_b,maxima_b));
 C209_FIXED_TASK(collect_region_maximum<0>(j,rows,a_shards,maxima_a,maximum_a));
 C209_FIXED_TASK(collect_region_maximum<1>(j,rows,b_shards,maxima_b,maximum_b));
 C209_FIXED_TASK(write_two_regions(output,j,rows,n,a_shards,b_shards,writes_a,writes_b));
}
static void finish_groups(bus_t *output,bus_t *x,const FixedJob &j,C209_GEMM_PAIR_PARAMS(input),
 c2fast::gelu_command_stream &gc,c2fast::nonlinear_pair_stream &gi,c2fast::pair_output_stream &go,c2fast::gelu_ack_stream &ga) {
#pragma HLS INLINE off
 c209gemmscale::Scale row_scale[TM],column_scale_a[3][TN],column_scale_b[2][TN];
 ap_uint<16> maximum_a[TM],maximum_b[TM];
#pragma HLS ARRAY_PARTITION variable=column_scale_a complete dim=1
#pragma HLS ARRAY_PARTITION variable=column_scale_a cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=column_scale_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=column_scale_b cyclic factor=2 dim=2
#pragma HLS BIND_STORAGE variable=column_scale_a type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=column_scale_b type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=row_scale type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=maximum_a type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=maximum_b type=ram_2p impl=bram
 unsigned n=tile_columns(j);
 for(ap_uint<13> rb=0;rb<j.m;rb+=TM) {
#pragma HLS LOOP_FLATTEN off
  unsigned rows=tile_rows(j,rb);
  for(unsigned r=0;r<rows;r++) {
#pragma HLS PIPELINE II=1
   row_scale[r]=c209gemmscale::encode_scale_bits(fu(readf(x,j.s0,rb+r)));maximum_a[r]=0;maximum_b[r]=0;
  }
  for(ap_uint<13> nb=0;nb<j.n;nb+=SHARDS*TN) {
#pragma HLS LOOP_FLATTEN off
   unsigned active=0;
   for(unsigned shard=0;shard<SHARDS;shard++)if(nb+shard*TN<j.n) {
    active++;
    for(unsigned col=0;col<n;col++) {
#pragma HLS PIPELINE II=1
     c209gemmscale::Scale scale=c209gemmscale::encode_scale_bits(fu(readf(x,j.s1,nb+shard*TN+col)));
     if(shard<3)column_scale_a[shard][col]=scale;
     else column_scale_b[shard-3][col]=scale;
    }
   }
   unsigned a_shards=active<3?active:3,b_shards=active>3?active-3:0;
   finish_regional_group(output,j,rb,nb,rows,n,a_shards,b_shards,row_scale,column_scale_a,column_scale_b,maximum_a,maximum_b,
                         C209_FIVE_STREAM_ARGS(input),gc,gi,go,ga);
  }
  if(j.op==GEMM_GELU_F16) {
   bus_t scale_word=0;
   for(unsigned r=0;r<rows;r++) {
#pragma HLS PIPELINE II=1
    ap_uint<16> maximum=maximum_a[r]>maximum_b[r]?maximum_a[r]:maximum_b[r];
    float absolute=c209mp::widen(c209mp::from_bits(maximum));
    float scale=absolute==0.f?1.f:absolute/32760.f;
    scale_word>>=32;scale_word.range(255,224)=fu(scale);
    if((r&7u)==7)x[word_address(j.s2,WordOffset((rb+r)>>3))]=scale_word;
   }
  }
 }
}
