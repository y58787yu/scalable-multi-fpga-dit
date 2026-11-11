#pragma once
#include <ap_int.h>
#include <hls_stream.h>
// Data-dependent completion, including FIFO stalls, uses real ap_done.
void c207_mac_rtl_service(hls::stream<ap_uint<99>> &commands,
                         hls::stream<ap_uint<256>> &ain,
                         hls::stream<ap_uint<256>> &bin,
                         hls::stream<ap_uint<768>> &results);
