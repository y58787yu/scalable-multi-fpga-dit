set compute $::env(COSMOS_HLS_SOURCE)
set output $::env(COSMOS_HLS_OUTPUT)
set ip_version $::env(COSMOS_HLS_IP_VERSION)
open_project [file join $output hls]
set_top cosmos2b_tile
set includes [file normalize [file join $compute include]]
set flags "-std=c++14 -ffp-contract=off -DCOSMOS_C205=1 -DCOSMOS_C207=1 -DCOSMOS_C209=1 -DC209_MIXED_PRECISION=1 -DC209_FIXED_PROFILE=1 -DC206_MAC_PO=16 -DC206_WEIGHT_BRAM=1 -DC206_NORM_SHARED=1 -DC206_CONSTANT_UNPACK=1 -I\"$includes\""
add_files [file join $compute src cosmos_fixed.cpp] -cflags $flags
foreach leaf {c209_five_mac_rtl c209_gelu_command_rtl c209_kv_broadcast_rtl} {
    add_files -blackbox [file join $output leaves "$leaf.json"]
}
open_solution solution1 -flow_target vivado
set_part xczu9eg-ffvb1156-2-e
create_clock -period 5.000 -name default
config_compile -unsafe_math_optimizations=false -no_signed_zeros=false
config_interface -m_axi_alignment_byte_size 32 -m_axi_latency 32 -m_axi_max_widen_bitwidth 256 -m_axi_addr64
config_rtl -register_reset_num 3
config_export -format ip_catalog -rtl verilog -version $ip_version -vivado_clock 5.000
csynth_design
export_design -format ip_catalog -rtl verilog
exit
