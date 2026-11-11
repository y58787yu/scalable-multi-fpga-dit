# ZCU102 single GT quad230: retained SFP1 owns COMMON; SFP2 is channel-only.
# The same full/channel definitions are reused; DISABLE_LOC_XDC allows the
# full instance to follow SFP1 pad locations without retaining unused SFP0.
# Based on the MIT-licensed verilog-ethernet ZCU102 reference GT settings.
set base_name eth_xcvr_gt
set c205_gt_config [list \
 CONFIG.preset GTH-10GBASE-R \
 CONFIG.TX_LINE_RATE 10.3125 CONFIG.RX_LINE_RATE 10.3125 \
 CONFIG.TX_REFCLK_FREQUENCY 156.25 CONFIG.RX_REFCLK_FREQUENCY 156.25 \
 CONFIG.TX_QPLL_FRACN_NUMERATOR 0 CONFIG.RX_QPLL_FRACN_NUMERATOR 0 \
 CONFIG.TX_USER_DATA_WIDTH 64 CONFIG.RX_USER_DATA_WIDTH 64 \
 CONFIG.TX_INT_DATA_WIDTH 32 CONFIG.RX_INT_DATA_WIDTH 32 \
 CONFIG.LOCATE_RESET_CONTROLLER CORE CONFIG.LOCATE_TX_USER_CLOCKING CORE \
 CONFIG.LOCATE_RX_USER_CLOCKING CORE CONFIG.LOCATE_USER_DATA_WIDTH_SIZING CORE \
 CONFIG.FREERUN_FREQUENCY 125 CONFIG.DISABLE_LOC_XDC 1]
# 10.3125GHz / 2 / 156.25MHz = 33; integral QPLL multiplier.
foreach {variant common optional} {full CORE {qpll0lock_out rxpolarity_in} channel EXAMPLE_DESIGN {rxpolarity_in}} {
 create_ip -name gtwizard_ultrascale -vendor xilinx.com -library ip -module_name ${base_name}_${variant}
 set_property -dict [concat $c205_gt_config [list CONFIG.LOCATE_COMMON $common CONFIG.ENABLE_OPTIONAL_PORTS $optional]] [get_ips ${base_name}_${variant}]
}
