#pragma once
#include <ap_int.h>
#include <hls_stream.h>
// Two 200 MHz clients share one unchanged 300 MHz INT16/INT48 core.
// A GEMM keeps its lease across every partial-K command until commit drains.
// B is the attention PV client; each completed QK grants one PV credit (max2).
// perf lanes, low word first: elapsed_slow_cycles, owner_a_cycles,
// owner_b_cycles, source_empty_cycles, result_stall_cycles, a_commits,
// b_commits, no_runnable_request_cycles. Counters saturate at0xffffffff; this value is possibly clipped.
// elapsed excludes idle, final perf backpressure and DONE. Owner cycles count
// SEND/OPERANDS/WAIT_COMMAND/WAIT_RESULT, not the arbitration cycle.
// source_empty: an owned cycle waiting for the next command, or at least one
// still-owed operand input is empty; count at most once per slow clock.
// result_stall: a buffered result is valid but its selected client cannot take
// it. no_runnable: arbitration has no eligible command and is not joining STOP.
// Native C perf is structural only: zero cycle fields, actual commit counts.
void c209_shared_mac_rtl(hls::stream<ap_uint<99>>& commands_a,
 hls::stream<ap_uint<256>>& ain_a,hls::stream<ap_uint<256>>& bin_a,
 hls::stream<ap_uint<768>>& results_a,hls::stream<ap_uint<99>>& commands_b,
 hls::stream<ap_uint<256>>& ain_b,hls::stream<ap_uint<256>>& bin_b,
 hls::stream<ap_uint<768>>& results_b,hls::stream<ap_uint<256>>& perf);
