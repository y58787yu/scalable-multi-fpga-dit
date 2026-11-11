// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x000 : Control signals
//         bit 0  - ap_start (Read/Write/COH)
//         bit 1  - ap_done (Read/COR)
//         bit 2  - ap_idle (Read)
//         bit 3  - ap_ready (Read/COR)
//         bit 7  - auto_restart (Read/Write)
//         bit 9  - interrupt (Read)
//         others - reserved
// 0x004 : Global Interrupt Enable Register
//         bit 0  - Global Interrupt Enable (Read/Write)
//         others - reserved
// 0x008 : IP Interrupt Enable Register (Read/Write)
//         bit 0 - enable ap_done interrupt (Read/Write)
//         bit 1 - enable ap_ready interrupt (Read/Write)
//         others - reserved
// 0x00c : IP Interrupt Status Register (Read/TOW)
//         bit 0 - ap_done (Read/TOW)
//         bit 1 - ap_ready (Read/TOW)
//         others - reserved
// 0x010 : Data signal of act_offset
//         bit 31~0 - act_offset[31:0] (Read/Write)
// 0x014 : Data signal of act_offset
//         bit 31~0 - act_offset[63:32] (Read/Write)
// 0x018 : reserved
// 0x01c : Data signal of weights_offset
//         bit 31~0 - weights_offset[31:0] (Read/Write)
// 0x020 : Data signal of weights_offset
//         bit 31~0 - weights_offset[63:32] (Read/Write)
// 0x024 : reserved
// 0x028 : Data signal of output_r_offset
//         bit 31~0 - output_r_offset[31:0] (Read/Write)
// 0x02c : Data signal of output_r_offset
//         bit 31~0 - output_r_offset[63:32] (Read/Write)
// 0x030 : reserved
// 0x034 : Data signal of aux_offset
//         bit 31~0 - aux_offset[31:0] (Read/Write)
// 0x038 : Data signal of aux_offset
//         bit 31~0 - aux_offset[63:32] (Read/Write)
// 0x03c : reserved
// 0x048 : Data signal of rows
//         bit 31~0 - rows[31:0] (Read/Write)
// 0x04c : reserved
// 0x0c4 : Data signal of aux0
//         bit 31~0 - aux0[31:0] (Read/Write)
// 0x0c8 : Data signal of aux0
//         bit 31~0 - aux0[63:32] (Read/Write)
// 0x0cc : reserved
// 0x104 : Data signal of error
//         bit 31~0 - error[31:0] (Read)
// 0x108 : Control signal of error
//         bit 0  - error_ap_vld (Read/COR)
//         others - reserved
// 0x114 : Data signal of completed_tiles
//         bit 31~0 - completed_tiles[31:0] (Read)
// 0x118 : Control signal of completed_tiles
//         bit 0  - completed_tiles_ap_vld (Read/COR)
//         others - reserved
// 0x124 : Data signal of perf_elapsed_cycles
//         bit 31~0 - perf_elapsed_cycles[31:0] (Read)
// 0x128 : Control signal of perf_elapsed_cycles
//         bit 0  - perf_elapsed_cycles_ap_vld (Read/COR)
//         others - reserved
// 0x134 : Data signal of perf_owner_a_cycles
//         bit 31~0 - perf_owner_a_cycles[31:0] (Read)
// 0x138 : Control signal of perf_owner_a_cycles
//         bit 0  - perf_owner_a_cycles_ap_vld (Read/COR)
//         others - reserved
// 0x144 : Data signal of perf_owner_b_cycles
//         bit 31~0 - perf_owner_b_cycles[31:0] (Read)
// 0x148 : Control signal of perf_owner_b_cycles
//         bit 0  - perf_owner_b_cycles_ap_vld (Read/COR)
//         others - reserved
// 0x154 : Data signal of perf_source_empty_cycles
//         bit 31~0 - perf_source_empty_cycles[31:0] (Read)
// 0x158 : Control signal of perf_source_empty_cycles
//         bit 0  - perf_source_empty_cycles_ap_vld (Read/COR)
//         others - reserved
// 0x164 : Data signal of perf_result_stall_cycles
//         bit 31~0 - perf_result_stall_cycles[31:0] (Read)
// 0x168 : Control signal of perf_result_stall_cycles
//         bit 0  - perf_result_stall_cycles_ap_vld (Read/COR)
//         others - reserved
// 0x174 : Data signal of perf_a_commits
//         bit 31~0 - perf_a_commits[31:0] (Read)
// 0x178 : Control signal of perf_a_commits
//         bit 0  - perf_a_commits_ap_vld (Read/COR)
//         others - reserved
// 0x184 : Data signal of perf_b_commits
//         bit 31~0 - perf_b_commits[31:0] (Read)
// 0x188 : Control signal of perf_b_commits
//         bit 0  - perf_b_commits_ap_vld (Read/COR)
//         others - reserved
// 0x194 : Data signal of perf_no_request_cycles
//         bit 31~0 - perf_no_request_cycles[31:0] (Read)
// 0x198 : Control signal of perf_no_request_cycles
//         bit 0  - perf_no_request_cycles_ap_vld (Read/COR)
//         others - reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XCOSMOS2B_TILE_CONTROL_ADDR_AP_CTRL                       0x000
#define XCOSMOS2B_TILE_CONTROL_ADDR_GIE                           0x004
#define XCOSMOS2B_TILE_CONTROL_ADDR_IER                           0x008
#define XCOSMOS2B_TILE_CONTROL_ADDR_ISR                           0x00c
#define XCOSMOS2B_TILE_CONTROL_ADDR_ACT_OFFSET_DATA               0x010
#define XCOSMOS2B_TILE_CONTROL_BITS_ACT_OFFSET_DATA               64
#define XCOSMOS2B_TILE_CONTROL_ADDR_WEIGHTS_OFFSET_DATA           0x01c
#define XCOSMOS2B_TILE_CONTROL_BITS_WEIGHTS_OFFSET_DATA           64
#define XCOSMOS2B_TILE_CONTROL_ADDR_OUTPUT_R_OFFSET_DATA          0x028
#define XCOSMOS2B_TILE_CONTROL_BITS_OUTPUT_R_OFFSET_DATA          64
#define XCOSMOS2B_TILE_CONTROL_ADDR_AUX_OFFSET_DATA               0x034
#define XCOSMOS2B_TILE_CONTROL_BITS_AUX_OFFSET_DATA               64
#define XCOSMOS2B_TILE_CONTROL_ADDR_ROWS_DATA                     0x048
#define XCOSMOS2B_TILE_CONTROL_BITS_ROWS_DATA                     32
#define XCOSMOS2B_TILE_CONTROL_ADDR_AUX0_DATA                     0x0c4
#define XCOSMOS2B_TILE_CONTROL_BITS_AUX0_DATA                     64
#define XCOSMOS2B_TILE_CONTROL_ADDR_ERROR_DATA                    0x104
#define XCOSMOS2B_TILE_CONTROL_BITS_ERROR_DATA                    32
#define XCOSMOS2B_TILE_CONTROL_ADDR_ERROR_CTRL                    0x108
#define XCOSMOS2B_TILE_CONTROL_ADDR_COMPLETED_TILES_DATA          0x114
#define XCOSMOS2B_TILE_CONTROL_BITS_COMPLETED_TILES_DATA          32
#define XCOSMOS2B_TILE_CONTROL_ADDR_COMPLETED_TILES_CTRL          0x118
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_ELAPSED_CYCLES_DATA      0x124
#define XCOSMOS2B_TILE_CONTROL_BITS_PERF_ELAPSED_CYCLES_DATA      32
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_ELAPSED_CYCLES_CTRL      0x128
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_OWNER_A_CYCLES_DATA      0x134
#define XCOSMOS2B_TILE_CONTROL_BITS_PERF_OWNER_A_CYCLES_DATA      32
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_OWNER_A_CYCLES_CTRL      0x138
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_OWNER_B_CYCLES_DATA      0x144
#define XCOSMOS2B_TILE_CONTROL_BITS_PERF_OWNER_B_CYCLES_DATA      32
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_OWNER_B_CYCLES_CTRL      0x148
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_SOURCE_EMPTY_CYCLES_DATA 0x154
#define XCOSMOS2B_TILE_CONTROL_BITS_PERF_SOURCE_EMPTY_CYCLES_DATA 32
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_SOURCE_EMPTY_CYCLES_CTRL 0x158
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_RESULT_STALL_CYCLES_DATA 0x164
#define XCOSMOS2B_TILE_CONTROL_BITS_PERF_RESULT_STALL_CYCLES_DATA 32
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_RESULT_STALL_CYCLES_CTRL 0x168
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_A_COMMITS_DATA           0x174
#define XCOSMOS2B_TILE_CONTROL_BITS_PERF_A_COMMITS_DATA           32
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_A_COMMITS_CTRL           0x178
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_B_COMMITS_DATA           0x184
#define XCOSMOS2B_TILE_CONTROL_BITS_PERF_B_COMMITS_DATA           32
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_B_COMMITS_CTRL           0x188
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_NO_REQUEST_CYCLES_DATA   0x194
#define XCOSMOS2B_TILE_CONTROL_BITS_PERF_NO_REQUEST_CYCLES_DATA   32
#define XCOSMOS2B_TILE_CONTROL_ADDR_PERF_NO_REQUEST_CYCLES_CTRL   0x198

