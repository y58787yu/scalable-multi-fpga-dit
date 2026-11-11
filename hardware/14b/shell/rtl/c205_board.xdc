# XDC constraints for the Xilinx ZCU102 board
# part: xczu9eg-ffvb1156-2-e

# General configuration
set_property BITSTREAM.GENERAL.COMPRESS true           [current_design]

# System clocks
# 125 MHz
set_property -dict {LOC G21  IOSTANDARD LVDS_25} [get_ports clk_125mhz_p]
set_property -dict {LOC F21  IOSTANDARD LVDS_25} [get_ports clk_125mhz_n]
create_clock -period 8.000 -name clk_125mhz [get_ports clk_125mhz_p]

# LEDs
set_property -dict {LOC AG14 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports {led[0]}]
set_property -dict {LOC AF13 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports {led[1]}]
set_property -dict {LOC AE13 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports {led[2]}]
set_property -dict {LOC AJ14 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports {led[3]}]
set_property -dict {LOC AJ15 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports {led[4]}]
set_property -dict {LOC AH13 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports {led[5]}]
set_property -dict {LOC AH14 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports {led[6]}]
set_property -dict {LOC AL12 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports {led[7]}]

set_false_path -to [get_ports {led[*]}]
set_output_delay 0 [get_ports {led[*]}]

# Reset button
set_property -dict {LOC AM13 IOSTANDARD LVCMOS33} [get_ports reset]

set_false_path -from [get_ports {reset}]
set_input_delay 0 [get_ports {reset}]

# Push buttons


# DIP switches


# UART



# I2C interfaces
#set_property -dict {LOC J10  IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports i2c0_scl]
#set_property -dict {LOC J11  IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports i2c0_sda]
#set_property -dict {LOC K20  IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports i2c1_scl]
#set_property -dict {LOC L20  IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports i2c1_sda]

#set_false_path -to [get_ports {i2c1_sda i2c1_scl}]
#set_output_delay 0 [get_ports {i2c1_sda i2c1_scl}]
#set_false_path -from [get_ports {i2c1_sda i2c1_scl}]
#set_input_delay 0 [get_ports {i2c1_sda i2c1_scl}]

# SFP+ Interface
set_property -dict {LOC D2  } [get_ports sfp0_rx_p] ;# MGTHRXP0_230 GTHE4_CHANNEL_X1Y12 / GTHE4_COMMON_X1Y3
set_property -dict {LOC D1  } [get_ports sfp0_rx_n] ;# MGTHRXN0_230 GTHE4_CHANNEL_X1Y12 / GTHE4_COMMON_X1Y3
set_property -dict {LOC E4  } [get_ports sfp0_tx_p] ;# MGTHTXP0_230 GTHE4_CHANNEL_X1Y12 / GTHE4_COMMON_X1Y3
set_property -dict {LOC E3  } [get_ports sfp0_tx_n] ;# MGTHTXN0_230 GTHE4_CHANNEL_X1Y12 / GTHE4_COMMON_X1Y3
set_property -dict {LOC C8  } [get_ports sfp_mgt_refclk_0_p] ;# MGTREFCLK0P_230 from U56 SI570 via U51 SI53340
set_property -dict {LOC C7  } [get_ports sfp_mgt_refclk_0_n] ;# MGTREFCLK0N_230 from U56 SI570 via U51 SI53340
#set_property -dict {LOC B10 } [get_ports sfp_mgt_refclk_1_p] ;# MGTREFCLK1P_230 from U20 CKOUT2 SI5328
#set_property -dict {LOC B9  } [get_ports sfp_mgt_refclk_1_n] ;# MGTREFCLK1N_230 from U20 CKOUT2 SI5328
#set_property -dict {LOC R10  IOSTANDARD LVDS} [get_ports sfp_recclk_p] ;# to U20 CKIN1 SI5328
#set_property -dict {LOC R9   IOSTANDARD LVDS} [get_ports sfp_recclk_n] ;# to U20 CKIN1 SI5328
set_property -dict {LOC A12  IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports sfp0_tx_disable_b]

# Additional ZCU102 SFP channels in the same GT quad/common.
set_property LOC C4 [get_ports sfp1_rx_p]
set_property LOC C3 [get_ports sfp1_rx_n]
set_property LOC D6 [get_ports sfp1_tx_p]
set_property LOC D5 [get_ports sfp1_tx_n]
set_property LOC B2 [get_ports sfp2_rx_p]
set_property LOC B1 [get_ports sfp2_rx_n]
set_property LOC B6 [get_ports sfp2_tx_p]
set_property LOC B5 [get_ports sfp2_tx_n]
set_property -dict {LOC A13 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports sfp1_tx_disable_b]
set_property -dict {LOC B13 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports sfp2_tx_disable_b]
# P3 pins from the checked-in verilog-ethernet ZCU102 board fpga.xdc,
# lines91-94/104 (sfp10g_test/external/verilog-ethernet/example/ZCU102/fpga).
# MGTH channel3 in quad230: GTHE4_CHANNEL_X1Y15 / GTHE4_COMMON_X1Y3.
set_property LOC A4 [get_ports sfp3_rx_p]
set_property LOC A3 [get_ports sfp3_rx_n]
set_property LOC A8 [get_ports sfp3_tx_p]
set_property LOC A7 [get_ports sfp3_tx_n]
set_property -dict {LOC C13 IOSTANDARD LVCMOS33 SLEW SLOW DRIVE 8} [get_ports sfp3_tx_disable_b]
# 156.25 MHz MGT reference clock
create_clock -period 6.400 -name sfp_mgt_refclk_0 [get_ports sfp_mgt_refclk_0_p]


set_false_path -to [get_cells -hier -filter {NAME =~ *phy_sync1_reg*}]
set_false_path -to [get_pins -hier -filter {REF_PIN_NAME == RXPOLARITY}]
