#pragma once
#include <ap_int.h>
#include <hls_stream.h>
// Fixed10-bit command: Mcode[1:0],N128[2],Kcode[4:3],reset5,
// commit6,STOP7,all5shards8,privateA9. Mcode0/1/2=32/128/256;
// Kcode0/1/2=16/64/128. Nbit0 gives16 for privateA,64 for sharedA;
// Nbit1 gives128 in either mode. Active mask is00001/11111.
// Shared mode consumes only a0 even when shard0 is inactive. Private mode consumes
// each active a stream. B and outputs are always per active shard. Panel32,K,half16
// operand order remains panel32,K,half16. Each original column16 result
// is now two 384-bit beats, lower8 INT48 first, then upper8 INT48.
// A non-reset command must continue an uncommitted C of the same M/N on every
// active shard. Commit closes those C contexts. STOP requires every C committed.
// FAST_GROUP_V1: elapsed,multiply,wait_A,wait_B,seed,store,output_blocked,other,
// eight mutually exclusive saturating fast-clock counters. Multiply is group
// issue cycles, NOT the sum over shards, useful MACs, DDR cycles or utilization.
// C numerical oracle has no clock model and returns all-zero perf explicitly.
void c209_five_mac_rtl(hls::stream<ap_uint<10>>& commands,
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
 hls::stream<ap_uint<256>>& perf);
