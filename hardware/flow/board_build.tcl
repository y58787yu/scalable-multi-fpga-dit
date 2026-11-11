if {$argc != 3} { error "board_build.tcl MODEL OUTPUT JOBS" }
lassign $argv model output jobs
if {![string match "2023.2*" [version -short]]} { error "Vivado 2023.2 is required" }
if {$model ni {2b 14b}} { error "Unknown hardware model $model" }
if {![string is integer -strict $jobs] || $jobs < 1} { error "Expected positive job count" }
set hw [file dirname [file dirname [file normalize [info script]]]]
set root [file join $hw $model shell]
set lib [file join $hw ethernet]
set build [file join $output project]
set kernel_repo [file join $output hls solution1 impl ip]
if {$model eq "2b"} {
    set c209_fixed_tag 0xC2092B10
    set c209_fixed_ip xilinx.com:hls:cosmos2b_tile:20.0
} else {
    set c209_fixed_tag 0xC2091000
    set c209_fixed_ip xilinx.com:hls:cosmos2b_tile:16.0
}
set artifacts [file join $output artifacts]
file mkdir $artifacts
create_project cosmos $build -part xczu9eg-ffvb1156-2-e
source [file join $hw $model board.tcl]
if {[get_property CONFIG.BIT_ID [get_bd_cells tp_batch]] != $c209_fixed_tag} {
    error "Batch controller and software hardware IDs do not agree"
}
generate_target all [get_ips {eth_xcvr_gt_full eth_xcvr_gt_channel c205_fp32_add c205_fp32_mul}]
set hwh [file join $build cosmos.gen sources_1 bd c206_cosmos_net_design hw_handoff c206_cosmos_net_design.hwh]
file copy $hwh [file join $artifacts cosmos_net.hwh]
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {![string match "*Complete*" [get_property STATUS [get_runs synth_1]]]} {
    error "Synthesis failed"
}
open_run synth_1
if {$model eq "2b"} { source [file join $root locate_gt.tcl] }
source [file join $root cdc.tcl]
set_property BITSTREAM.CONFIG.USR_ACCESS $c209_fixed_tag [current_design]
opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
report_route_status -file [file join $artifacts route.rpt]
report_timing_summary -file [file join $artifacts timing.rpt]
report_utilization -file [file join $artifacts utilization.rpt]
write_bitstream [file join $artifacts cosmos_net.bit]
exit
