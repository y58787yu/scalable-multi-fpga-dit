# Invoked by flow/board_build.tcl after opening the ZCU102 project.
set_property source_mgmt_mode All [current_project]
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
set_property board_part xilinx.com:zcu102:part0:3.4 [current_project]
set_property ip_repo_paths [list $kernel_repo] [current_project]
update_ip_catalog
foreach name {c206_tensor_link.sv c205_fp32_token.sv c206_stream_reducer.sv c205_ethernet_link.sv eth_xcvr_phy_wrapper.v} {add_files -norecurse [file join $root rtl $name]}
set c209_registered {c206_mesh_router.sv c209_router_output.sv c206_tensor_link.sv c205_fp32_token.sv c206_stream_reducer.sv c205_ethernet_link.sv eth_xcvr_phy_wrapper.v c206_cosmos_net_top.sv}
foreach path [lsort [concat [glob -nocomplain [file join $root rtl *.v]] [glob -nocomplain [file join $root rtl *.sv]]]] {
 if {[file tail $path] ni $c209_registered} {add_files -norecurse $path}
}
set_property file_type Verilog [get_files [list [file join $root rtl c206_tensor_link.sv] [file join $root rtl c206_stream_reducer.sv]]]
foreach name {eth_mac_10g_fifo eth_mac_10g axis_xgmii_rx_64 axis_xgmii_tx_64 eth_phy_10g eth_phy_10g_rx eth_phy_10g_rx_if eth_phy_10g_rx_frame_sync eth_phy_10g_rx_ber_mon eth_phy_10g_rx_watchdog eth_phy_10g_tx eth_phy_10g_tx_if xgmii_baser_dec_64 xgmii_baser_enc_64 lfsr} {add_files -norecurse [file join $lib rtl ${name}.v]}
foreach name {axis_fifo axis_async_fifo axis_async_fifo_adapter sync_reset} {add_files -norecurse [file join $lib lib axis rtl ${name}.v]}
foreach path [list [file join $root rtl c205_board.xdc] [file join $lib syn vivado eth_mac_fifo.tcl] [file join $lib lib axis syn vivado axis_async_fifo.tcl] [file join $lib lib axis syn vivado sync_reset.tcl]] {add_files -fileset constrs_1 -norecurse $path}
source [file join $root scripts create_c205_gt.tcl]
source [file join $root scripts create_c205_fp_vivado.tcl]
update_compile_order -fileset sources_1
create_bd_design c206_cosmos_net_design
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells ps]
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 0 CONFIG.PSU__USE__S_AXI_GP0 1 CONFIG.PSU__USE__S_AXI_GP1 1 CONFIG.PSU__USE__S_AXI_GP2 1 CONFIG.PSU__USE__S_AXI_GP3 1 CONFIG.PSU__USE__S_AXI_GP4 1 CONFIG.PSU__USE__S_AXI_GP5 0 CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ 200] [get_bd_cells ps]
set logic_hz [get_property CONFIG.FREQ_HZ [get_bd_pins ps/pl_clk0]]
if {$logic_hz eq ""} {error "PS preset did not resolve an actual PL clock"}
set c209_ps_reference_hz $logic_hz
set c206_compute_ip_version [lindex [split $c209_fixed_ip :] 3]
set candidates [get_ipdefs -all [format "*:hls:cosmos2b_tile:%s" $c206_compute_ip_version]]
if {[llength $candidates]!=1} {error "Expected one exported compute cosmos2b_tile:$c206_compute_ip_version definition"}
create_bd_cell -type ip -vlnv [get_property VLNV [lindex $candidates 0]] cosmos2b_tile_0
set_property CONFIG.FREQ_HZ.VALUE_SRC DEFAULT [get_bd_pins cosmos2b_tile_0/mac_fast_clk]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_control
set_property -dict [list CONFIG.C_EXT_RESET_HIGH 0 CONFIG.C_AUX_RESET_HIGH 0] [get_bd_cells reset_control]
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 one
set_property CONFIG.CONST_VAL 1 [get_bd_cells one]
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 zero
set_property CONFIG.CONST_VAL 0 [get_bd_cells zero]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins reset_control/ext_reset_in]
connect_bd_net [get_bd_pins one/dout] [get_bd_pins reset_control/aux_reset_in]
connect_bd_net [get_bd_pins zero/dout] [get_bd_pins reset_control/mb_debug_sys_rst]
if {$c209_ps_reference_hz != 187481262} {error "Actual frozen PS clock reference changed"}
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 mac_fast_clock
set_property -dict [list CONFIG.PRIM_IN_FREQ [expr {$c209_ps_reference_hz/1000000.0}] CONFIG.NUM_OUT_CLKS 2 CONFIG.CLKOUT1_REQUESTED_OUT_FREQ 200.000 CONFIG.CLKOUT1_USED true CONFIG.CLKOUT2_REQUESTED_OUT_FREQ 300.000 CONFIG.CLKOUT2_USED true CONFIG.USE_LOCKED true CONFIG.USE_RESET true CONFIG.RESET_TYPE ACTIVE_HIGH CONFIG.PRIM_SOURCE No_buffer] [get_bd_cells mac_fast_clock]
create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 mac_clock_reset_not
set_property -dict [list CONFIG.C_OPERATION not CONFIG.C_SIZE 1] [get_bd_cells mac_clock_reset_not]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_mac_fast
set_property -dict [list CONFIG.C_EXT_RESET_HIGH 0 CONFIG.C_AUX_RESET_HIGH 0] [get_bd_cells reset_mac_fast]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins mac_fast_clock/clk_in1]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins mac_clock_reset_not/Op1]
connect_bd_net [get_bd_pins mac_clock_reset_not/Res] [get_bd_pins mac_fast_clock/reset]
connect_bd_net [get_bd_pins mac_fast_clock/clk_out1] [get_bd_pins reset_control/slowest_sync_clk]
connect_bd_net [get_bd_pins mac_fast_clock/clk_out2] [get_bd_pins reset_mac_fast/slowest_sync_clk] [get_bd_pins cosmos2b_tile_0/mac_fast_clk]
connect_bd_net [get_bd_pins mac_fast_clock/locked] [get_bd_pins reset_mac_fast/dcm_locked] [get_bd_pins reset_control/dcm_locked]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins reset_mac_fast/ext_reset_in]
connect_bd_net [get_bd_pins one/dout] [get_bd_pins reset_mac_fast/aux_reset_in]
connect_bd_net [get_bd_pins zero/dout] [get_bd_pins reset_mac_fast/mb_debug_sys_rst]
connect_bd_net [get_bd_pins reset_mac_fast/peripheral_reset] [get_bd_pins cosmos2b_tile_0/mac_fast_rst]
set logic_hz [get_property CONFIG.FREQ_HZ [get_bd_pins mac_fast_clock/clk_out1]]
set c209_fast_hz [get_property CONFIG.FREQ_HZ [get_bd_pins mac_fast_clock/clk_out2]]
if {$logic_hz eq "" || $c209_fast_hz eq "" || abs($logic_hz-200000000)>500000 || abs($c209_fast_hz-300000000)>750000} {error "Clock wizard actual output frequency absent/outside target tolerance"}

proc c206_smart {name si mi} {
 create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 $name
 set_property -dict [list CONFIG.NUM_SI $si CONFIG.NUM_MI $mi] [get_bd_cells $name]
}
proc c206_if {cell wanted} {
 foreach p [get_bd_intf_pins $cell/*] {if {[string equal -nocase [file tail $p] $wanted]} {return $p}}
 error "Missing $cell/$wanted; available [get_bd_intf_pins $cell/*]"
}
c206_smart control_bus 2 9
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins control_bus/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_bus/M00_AXI] [get_bd_intf_pins cosmos2b_tile_0/s_axi_control]
create_bd_cell -type module -reference c209_tp4_batch_controller tp_batch
set_property CONFIG.BIT_ID $c209_fixed_tag [get_bd_cells tp_batch]
if {[get_property CONFIG.BIT_ID [get_bd_cells tp_batch]] != $c209_fixed_tag} {
 error "TP batch BIT_ID does not match the current link image"
}
set_property CONFIG.FREQ_HZ.VALUE_SRC DEFAULT [get_bd_pins tp_batch/aclk]
connect_bd_intf_net [c206_if tp_batch m_axi] [get_bd_intf_pins control_bus/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins control_bus/M08_AXI] [c206_if tp_batch s_axi]
foreach {master slave} {act HP0 output_r HP1 weights HPC0} {
 c206_smart data_$master 1 1
 connect_bd_intf_net [get_bd_intf_pins cosmos2b_tile_0/m_axi_$master] [get_bd_intf_pins data_$master/S00_AXI]
 connect_bd_intf_net [get_bd_intf_pins data_$master/M00_AXI] [get_bd_intf_pins ps/S_AXI_${slave}_FPD]
}
set control_index 1
foreach lane {1 2} {
 set dma dma$lane;set residual residual_dma$lane;set link link$lane;set reduce reducer$lane
 create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 $dma
 set_property -dict [list CONFIG.c_include_sg 0 CONFIG.c_addr_width 64 CONFIG.c_sg_length_width 23 CONFIG.c_m_axi_mm2s_data_width 128 CONFIG.c_m_axi_s2mm_data_width 128 CONFIG.c_m_axis_mm2s_tdata_width 64 CONFIG.c_s_axis_s2mm_tdata_width 64 CONFIG.c_include_mm2s_dre 0 CONFIG.c_include_s2mm_dre 0 CONFIG.c_mm2s_burst_size 64 CONFIG.c_s2mm_burst_size 64] [get_bd_cells $dma]
 if {$lane==1} {
  create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 $residual
  set_property -dict [list CONFIG.c_include_sg 0 CONFIG.c_include_s2mm 0 CONFIG.c_addr_width 64 CONFIG.c_sg_length_width 23 CONFIG.c_m_axi_mm2s_data_width 128 CONFIG.c_m_axis_mm2s_tdata_width 64 CONFIG.c_include_mm2s_dre 0 CONFIG.c_mm2s_burst_size 64] [get_bd_cells $residual]
 }
 create_bd_cell -type module -reference c206_tensor_link $link
 set_property CONFIG.LANE_ID $lane [get_bd_cells $link]
 create_bd_cell -type module -reference c206_stream_reducer $reduce
 # TP2 uses one P1 FUSED reduction. P2 is the one full-duplex SP2 RAW
 # endpoint; it remains idle in SP1. P0/P3 have no instantiated datapath.
 set_property CONFIG.BULK_ONLY [expr {$lane==1?0:1}] [get_bd_cells $reduce]
 set_property CONFIG.SUM_ONLY 0 [get_bd_cells $reduce]
 set lane_control [list $dma S_AXI_LITE]
 if {$lane==1} {lappend lane_control $residual S_AXI_LITE}
 lappend lane_control $link s_axi $reduce s_axi
 foreach {cell interface} $lane_control {
  connect_bd_intf_net [get_bd_intf_pins [format "control_bus/M%02d_AXI" $control_index]] [c206_if $cell $interface]
  incr control_index
 }
 create_bd_cell -type ip -vlnv xilinx.com:ip:axis_broadcaster:1.1 broadcast$lane
 set_property -dict [list CONFIG.NUM_MI 2 CONFIG.S_TDATA_NUM_BYTES 8 CONFIG.M_TDATA_NUM_BYTES 8 CONFIG.HAS_TKEEP 1 CONFIG.HAS_TLAST 1] [get_bd_cells broadcast$lane]
 if {$lane==1} {
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 local_fifo$lane
  set_property -dict [list CONFIG.FIFO_DEPTH 1024 CONFIG.TDATA_NUM_BYTES 8 CONFIG.HAS_TKEEP 1 CONFIG.HAS_TLAST 1 CONFIG.FIFO_MODE 1] [get_bd_cells local_fifo$lane]
 }
 connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins broadcast$lane/S_AXIS]
 connect_bd_intf_net [get_bd_intf_pins broadcast$lane/M00_AXIS] [c206_if $link s_axis_mm2s]
 if {$lane==1} {
  connect_bd_intf_net [get_bd_intf_pins broadcast$lane/M01_AXIS] [get_bd_intf_pins local_fifo$lane/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins local_fifo$lane/M_AXIS] [c206_if $reduce s_axis_local]
 } else {
  # RAW-only P2 needs no residual producer or local replay FIFO.
  connect_bd_intf_net [get_bd_intf_pins broadcast$lane/M01_AXIS] [c206_if $reduce s_axis_local]
 }
 connect_bd_intf_net [c206_if $link m_axis_s2mm] [c206_if $reduce s_axis_peer]
 if {$lane==1} {connect_bd_intf_net [get_bd_intf_pins $residual/M_AXIS_MM2S] [c206_if $reduce s_axis_residual]}
 connect_bd_intf_net [c206_if $reduce m_axis_result] [get_bd_intf_pins $dma/S_AXIS_S2MM]
 # Preserve the independent PS memory paths. HP2 is physically 128 bits wide.
 # Narrow only the 256-bit aux ingress BEFORE the shared three-input fabric;
 # otherwise SmartConnect widens both 128-bit P2 DMA paths to 256 bits and
 # narrows it again at HP2. Read/write arbitration remains independent and the
 # existing outstanding-request capacity is preserved for DDR latency hiding.
 set mem [lindex {HP3 HPC1 HP2} $lane]
 # P1: TX/RX/residual. P2: TX/RX/compute aux. No P0/P3 master slots.
 c206_smart lane_memory$lane 3 1
 connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins lane_memory$lane/S00_AXI]
 connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins lane_memory$lane/S01_AXI]
 if {$lane==1} {connect_bd_intf_net [get_bd_intf_pins $residual/M_AXI_MM2S] [get_bd_intf_pins lane_memory$lane/S02_AXI]}
 if {$lane==2} {
  create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dwidth_converter:2.1 aux_to_hp2_128
  set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.READ_WRITE_MODE READ_WRITE \
   CONFIG.ADDR_WIDTH 64 CONFIG.SI_DATA_WIDTH 256 CONFIG.MI_DATA_WIDTH 128 \
   CONFIG.SI_ID_WIDTH 1 CONFIG.MAX_SPLIT_BEATS 256 CONFIG.FIFO_MODE 0 CONFIG.ACLK_ASYNC 0] [get_bd_cells aux_to_hp2_128]
  # The native downsizer has independent read/write command queues, each with
  # 32 entries. Keep the existing 16-request, one-thread interface contract;
  # FIFO_MODE=0 selects the synchronous downsizer, not one outstanding request.
  foreach interface {S_AXI M_AXI} {
   set_property -dict [list CONFIG.NUM_READ_OUTSTANDING 16 CONFIG.NUM_WRITE_OUTSTANDING 16 \
    CONFIG.NUM_READ_THREADS 1 CONFIG.NUM_WRITE_THREADS 1] [get_bd_intf_pins aux_to_hp2_128/$interface]
  }
  connect_bd_net [get_bd_pins mac_fast_clock/clk_out1] [get_bd_pins aux_to_hp2_128/s_axi_aclk]
  connect_bd_net [get_bd_pins reset_control/peripheral_aresetn] [get_bd_pins aux_to_hp2_128/s_axi_aresetn]
  connect_bd_intf_net [get_bd_intf_pins cosmos2b_tile_0/m_axi_aux] [get_bd_intf_pins aux_to_hp2_128/S_AXI]
  connect_bd_intf_net [get_bd_intf_pins aux_to_hp2_128/M_AXI] [get_bd_intf_pins lane_memory$lane/S02_AXI]
 }
 connect_bd_intf_net [get_bd_intf_pins lane_memory$lane/M00_AXI] [get_bd_intf_pins ps/S_AXI_${mem}_FPD]
 foreach {name mode cellif} {TX Master m_axis_net RX Slave s_axis_net} {
  set port MAC${lane}_$name
  create_bd_intf_port -mode $mode -vlnv xilinx.com:interface:axis_rtl:1.0 $port
  set_property CONFIG.FREQ_HZ $logic_hz [get_bd_intf_ports $port]
  if {$mode eq "Slave"} {set_property -dict [list CONFIG.TDATA_NUM_BYTES 8 CONFIG.HAS_TKEEP 1 CONFIG.HAS_TLAST 1 CONFIG.TUSER_WIDTH 1] [get_bd_intf_ports $port]}
  connect_bd_intf_net [get_bd_intf_ports $port] [c206_if $link $cellif]
 }
 foreach {name direction width} {phy_status I 8 mac_errors I 8 rx_polarity O 1} {
  create_bd_port -dir $direction -from [expr {$width-1}] -to 0 ${name}$lane
  connect_bd_net [get_bd_ports ${name}$lane] [get_bd_pins $link/$name]
 }
}
if {$control_index != 8} {error "Unexpected TP2/two-port control interface count"}
foreach bits {8 64} {
 create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 sp_zero$bits
 set_property -dict [list CONFIG.CONST_WIDTH $bits CONFIG.CONST_VAL 0] [get_bd_cells sp_zero$bits]
}
connect_bd_net [get_bd_pins sp_zero64/dout] [get_bd_pins reducer2/s_axis_residual_tdata]
connect_bd_net [get_bd_pins sp_zero8/dout] [get_bd_pins reducer2/s_axis_residual_tkeep]
connect_bd_net [get_bd_pins zero/dout] [get_bd_pins reducer2/s_axis_residual_tvalid] [get_bd_pins reducer2/s_axis_residual_tlast]
foreach pin [get_bd_pins -of_objects [get_bd_cells] -filter {DIR == I && TYPE == clk}] {
 if {![llength [get_bd_nets -of_objects $pin]]} {connect_bd_net [get_bd_pins mac_fast_clock/clk_out1] $pin}
}
foreach pin [get_bd_pins -of_objects [get_bd_cells] -filter {DIR == I && TYPE == rst}] {
 if {![llength [get_bd_nets -of_objects $pin]] && ([string match *aresetn $pin]||[string match *rst_n $pin])} {connect_bd_net [get_bd_pins reset_control/peripheral_aresetn] $pin}
}
create_bd_port -dir O -type clk logic_clk
set_property -dict [list CONFIG.FREQ_HZ $logic_hz CONFIG.ASSOCIATED_BUSIF {MAC1_TX:MAC1_RX:MAC2_TX:MAC2_RX} CONFIG.ASSOCIATED_RESET logic_resetn] [get_bd_ports logic_clk]
connect_bd_net [get_bd_ports logic_clk] [get_bd_pins mac_fast_clock/clk_out1]
create_bd_port -dir O -type rst logic_resetn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports logic_resetn]
connect_bd_net [get_bd_ports logic_resetn] [get_bd_pins reset_control/peripheral_aresetn]
create_bd_port -dir I -from 31 -to 0 user_tag
foreach lane {1 2} {connect_bd_net [get_bd_ports user_tag] [get_bd_pins link$lane/user_tag]}
create_bd_port -dir O -from 7 -to 0 node_id
connect_bd_net [get_bd_ports node_id] [get_bd_pins link1/node_id]
foreach name {router_dropped router_forwarded router_delivered} {
 create_bd_port -dir I -from 31 -to 0 $name
 foreach lane {1 2} {connect_bd_net [get_bd_ports $name] [get_bd_pins link$lane/$name]}
}
assign_bd_address
proc c206_address {cell address range} {
 set seg [get_bd_addr_segs -quiet $cell/*/Reg]
 if {![llength $seg]} {set seg [get_bd_addr_segs -quiet $cell/*]}
 if {[llength $seg]!=1} {error "Expected one control segment for $cell: $seg"}
 assign_bd_address -force -offset $address -range $range -target_address_space [get_bd_addr_spaces ps/Data] [lindex $seg 0]
}
c206_address cosmos2b_tile_0 0xA0000000 0x1000
foreach lane {1 2} {
 c206_address dma$lane [format 0x%X [expr {0xA0010000+$lane*0x40000}]] 0x10000
 if {$lane==1} {c206_address residual_dma$lane [format 0x%X [expr {0xA0110000+$lane*0x20000}]] 0x10000}
 c206_address link$lane [format 0x%X [expr {0xA0020000+$lane*0x1000}]] 0x1000
 c206_address reducer$lane [format 0x%X [expr {0xA0200000+$lane*0x1000}]] 0x1000
}
set batch_slave_seg [get_bd_addr_segs -of_objects [c206_if tp_batch s_axi]]
if {[llength $batch_slave_seg] != 1} {error "Expected one TP batch doorbell segment"}
assign_bd_address -force -offset 0xA0210000 -range 0x1000 -target_address_space [get_bd_addr_spaces ps/Data] [lindex $batch_slave_seg 0]
set batch_space [get_bd_addr_spaces -of_objects [c206_if tp_batch m_axi]]
if {[llength $batch_space] != 1} {error "Expected one TP batch AXI-Lite address space"}
foreach {cell offset range} {
 dma1 0xA0050000 0x10000 link1 0xA0021000 0x1000 reducer1 0xA0201000 0x1000
 residual_dma1 0xA0130000 0x10000
} {
 set seg [get_bd_addr_segs -quiet $cell/*/Reg]
 if {![llength $seg]} {set seg [get_bd_addr_segs -quiet $cell/*]}
 if {[llength $seg] != 1} {error "Ambiguous TP batch register segment $cell"}
 assign_bd_address -force -offset $offset -range $range -target_address_space $batch_space [lindex $seg 0]
}
foreach {cell interface} {cosmos2b_tile_0 s_axi_control dma2 S_AXI_LITE link2 s_axi reducer2 s_axi tp_batch s_axi} {
 foreach seg [get_bd_addr_segs -of_objects [c206_if $cell $interface]] {
  exclude_bd_addr_seg -target_address_space $batch_space $seg
 }
}
foreach removed {dma0 residual_dma0 link0 reducer0 broadcast0 local_fifo0 lane_memory0 dma3 residual_dma3 link3 reducer3 broadcast3 local_fifo3} {
 if {[llength [get_bd_cells -quiet $removed]]} {error "Removed two-port shell block still present: $removed"}
}
if {[get_property CONFIG.NUM_MI [get_bd_cells control_bus]] != 9 || [get_property CONFIG.NUM_SI [get_bd_cells lane_memory2]] != 3} {error "Two-port AXI fabric was not physically reduced"}
foreach {pattern expected} {dma* {dma1 dma2} residual_dma* {residual_dma1} link* {link1 link2} reducer* {reducer1 reducer2} broadcast* {broadcast1 broadcast2} local_fifo* {local_fifo1} lane_memory* {lane_memory1 lane_memory2}} {
 set actual [list]
 foreach cell [get_bd_cells -quiet $pattern] {lappend actual [file tail $cell]}
 if {[lsort $actual] ne [lsort $expected]} {error "Two-port BD cell set differs for $pattern: $actual != $expected"}
}
foreach {master slave} {
 control_bus/M08_AXI tp_batch/s_axi
 aux_to_hp2_128/M_AXI lane_memory2/S02_AXI
 lane_memory2/M00_AXI ps/S_AXI_HP2_FPD
} {
 set master_net [get_bd_intf_nets -of_objects [c206_if [file dirname $master] [file tail $master]]]
 set slave_net [get_bd_intf_nets -of_objects [c206_if [file dirname $slave] [file tail $slave]]]
 if {[llength $master_net] != 1 || $master_net ne $slave_net} {error "Two-port required interface connection missing: $master -> $slave"}
}

validate_bd_design
save_bd_design
generate_target all [get_files c206_cosmos_net_design.bd]
add_files -norecurse [make_wrapper -files [get_files c206_cosmos_net_design.bd] -top]
add_files -norecurse [file join $root rtl c206_cosmos_net_top.sv]
set_property top c206_cosmos_net_top [current_fileset]
update_compile_order -fileset sources_1
