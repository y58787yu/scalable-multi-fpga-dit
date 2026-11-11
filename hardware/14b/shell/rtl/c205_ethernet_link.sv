/* Copyright (c) 2020-2023 Alex Forencich
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
// C209 TP4/PP3: four independent SFP+ MACs with one GT quad common.
// Only port0 owns COMMON. Ports1..3 use its QPLL and cannot reset it on loss
// of RX lock. P3 cable absence has no input to global logic/clock resets.
module c205_ethernet_link(
 input wire clk_125mhz_p,clk_125mhz_n,reset,
 input wire [3:0] sfp_rx_p,sfp_rx_n,output wire [3:0] sfp_tx_p,sfp_tx_n,sfp_tx_disable_b,
 input wire sfp_mgt_refclk_0_p,sfp_mgt_refclk_0_n,
 input wire logic_clk,logic_resetn,input wire [3:0] rx_polarity,
 input wire [255:0] tx_data,input wire [31:0] tx_keep,input wire [3:0] tx_valid,output wire [3:0] tx_ready,input wire [3:0] tx_last,tx_user,
 output wire [255:0] rx_data,output wire [31:0] rx_keep,output wire [3:0] rx_valid,input wire [3:0] rx_ready,output wire [3:0] rx_last,rx_user,
 output wire [31:0] phy_status,mac_errors,output wire [7:0] led);
// Clock and reset

wire clk_125mhz_ibufg;
wire clk_125mhz_bufg;

// Internal 125 MHz clock
wire clk_125mhz_mmcm_out;
wire clk_125mhz_int;
wire rst_125mhz_int;



wire mmcm_rst = reset;
wire mmcm_locked;
wire mmcm_clkfb;

IBUFGDS #(
   .DIFF_TERM("FALSE"),
   .IBUF_LOW_PWR("FALSE")   
)
clk_125mhz_ibufg_inst (
   .O   (clk_125mhz_ibufg),
   .I   (clk_125mhz_p),
   .IB  (clk_125mhz_n) 
);

BUFG
clk_125mhz_bufg_in_inst (
    .I(clk_125mhz_ibufg),
    .O(clk_125mhz_bufg)
);

// MMCM instance
// 125 MHz in, 125 MHz out
// PFD range: 10 MHz to 500 MHz
// VCO range: 800 MHz to 1600 MHz
// M = 8, D = 1 sets Fvco = 1000 MHz (in range)
// Divide by 8 to get output frequency of 125 MHz
MMCME4_BASE #(
    .BANDWIDTH("OPTIMIZED"),
    .CLKOUT0_DIVIDE_F(8),
    .CLKOUT0_DUTY_CYCLE(0.5),
    .CLKOUT0_PHASE(0),
    .CLKOUT1_DIVIDE(1),
    .CLKOUT1_DUTY_CYCLE(0.5),
    .CLKOUT1_PHASE(0),
    .CLKOUT2_DIVIDE(1),
    .CLKOUT2_DUTY_CYCLE(0.5),
    .CLKOUT2_PHASE(0),
    .CLKOUT3_DIVIDE(1),
    .CLKOUT3_DUTY_CYCLE(0.5),
    .CLKOUT3_PHASE(0),
    .CLKOUT4_DIVIDE(1),
    .CLKOUT4_DUTY_CYCLE(0.5),
    .CLKOUT4_PHASE(0),
    .CLKOUT5_DIVIDE(1),
    .CLKOUT5_DUTY_CYCLE(0.5),
    .CLKOUT5_PHASE(0),
    .CLKOUT6_DIVIDE(1),
    .CLKOUT6_DUTY_CYCLE(0.5),
    .CLKOUT6_PHASE(0),
    .CLKFBOUT_MULT_F(8),
    .CLKFBOUT_PHASE(0),
    .DIVCLK_DIVIDE(1),
    .REF_JITTER1(0.010),
    .CLKIN1_PERIOD(8.0),
    .STARTUP_WAIT("FALSE"),
    .CLKOUT4_CASCADE("FALSE")
)
clk_mmcm_inst (
    .CLKIN1(clk_125mhz_bufg),
    .CLKFBIN(mmcm_clkfb),
    .RST(mmcm_rst),
    .PWRDWN(1'b0),
    .CLKOUT0(clk_125mhz_mmcm_out),
    .CLKOUT0B(),
    .CLKOUT1(),
    .CLKOUT1B(),
    .CLKOUT2(),
    .CLKOUT2B(),
    .CLKOUT3(),
    .CLKOUT3B(),
    .CLKOUT4(),
    .CLKOUT5(),
    .CLKOUT6(),
    .CLKFBOUT(mmcm_clkfb),
    .CLKFBOUTB(),
    .LOCKED(mmcm_locked)
);

BUFG
clk_125mhz_bufg_inst (
    .I(clk_125mhz_mmcm_out),
    .O(clk_125mhz_int)
);

sync_reset #(
    .N(4)
)
sync_reset_125mhz_inst (
    .clk(clk_125mhz_int),
    .rst(~mmcm_locked),
    .out(rst_125mhz_int)
);


 wire refclk,qpll_lock,qpll_clk,qpll_refclk;
 IBUFDS_GTE4 refclk_buf(.I(sfp_mgt_refclk_0_p),.IB(sfp_mgt_refclk_0_n),.CEB(1'b0),.O(refclk),.ODIV2());
 assign sfp_tx_disable_b=4'b1111;
 genvar g;
 generate for(g=0;g<4;g=g+1)begin:ports
  wire rx_clk,rx_rst,tx_clk,tx_rst,power_good,block_lock,high_ber,rx_status,rx_bad_block;
  wire [6:0] rx_error_count;wire [63:0] xgmii_txd,xgmii_rxd;wire [7:0] xgmii_txc,xgmii_rxc;
  wire pll_lock_out,pll_clk_out,pll_refclk_out;
  if(g==0)begin:common
   assign qpll_lock=pll_lock_out;assign qpll_clk=pll_clk_out;assign qpll_refclk=pll_refclk_out;
  end
  eth_xcvr_phy_wrapper #(.HAS_COMMON(g==0)) phy(
   .xcvr_ctrl_clk(clk_125mhz_int),.xcvr_ctrl_rst(rst_125mhz_int),.xcvr_rx_polarity(rx_polarity[g]),
   .xcvr_gtpowergood_out(power_good),.xcvr_gtrefclk00_in(refclk),
   .xcvr_qpll0lock_out(pll_lock_out),.xcvr_qpll0clk_out(pll_clk_out),.xcvr_qpll0refclk_out(pll_refclk_out),
   .xcvr_qpll0lock_in(qpll_lock),.xcvr_qpll0reset_out(),.xcvr_qpll0clk_in(qpll_clk),.xcvr_qpll0refclk_in(qpll_refclk),
   .xcvr_txp(sfp_tx_p[g]),.xcvr_txn(sfp_tx_n[g]),.xcvr_rxp(sfp_rx_p[g]),.xcvr_rxn(sfp_rx_n[g]),
   .phy_tx_clk(tx_clk),.phy_tx_rst(tx_rst),.phy_xgmii_txd(xgmii_txd),.phy_xgmii_txc(xgmii_txc),
   .phy_rx_clk(rx_clk),.phy_rx_rst(rx_rst),.phy_xgmii_rxd(xgmii_rxd),.phy_xgmii_rxc(xgmii_rxc),
   .phy_tx_bad_block(),.phy_rx_error_count(rx_error_count),.phy_rx_bad_block(rx_bad_block),
   .phy_rx_sequence_error(),.phy_rx_block_lock(block_lock),.phy_rx_high_ber(high_ber),.phy_rx_status(rx_status),
   .phy_cfg_tx_prbs31_enable(1'b0),.phy_cfg_rx_prbs31_enable(1'b0));
  wire mac_rx_reset,mac_tx_reset;
  sync_reset #(.N(4)) rx_release(.clk(rx_clk),.rst(rx_rst||!logic_resetn),.out(mac_rx_reset));
  sync_reset #(.N(4)) tx_release(.clk(tx_clk),.rst(tx_rst||!logic_resetn),.out(mac_tx_reset));
  eth_mac_10g_fifo #(.TX_FIFO_DEPTH(16384),.RX_FIFO_DEPTH(16384),.RX_FRAME_FIFO(1),.RX_DROP_BAD_FRAME(1)) mac(
   .rx_clk(rx_clk),.rx_rst(mac_rx_reset),.tx_clk(tx_clk),.tx_rst(mac_tx_reset),.logic_clk(logic_clk),.logic_rst(!logic_resetn),.ptp_sample_clk(1'b0),
   .tx_axis_tdata(tx_data[64*g+:64]),.tx_axis_tkeep(tx_keep[8*g+:8]),.tx_axis_tvalid(tx_valid[g]),.tx_axis_tready(tx_ready[g]),.tx_axis_tlast(tx_last[g]),.tx_axis_tuser(tx_user[g]),
   .m_axis_tx_ptp_ts_96(),.m_axis_tx_ptp_ts_tag(),.m_axis_tx_ptp_ts_valid(),.m_axis_tx_ptp_ts_ready(1'b1),
   .rx_axis_tdata(rx_data[64*g+:64]),.rx_axis_tkeep(rx_keep[8*g+:8]),.rx_axis_tvalid(rx_valid[g]),.rx_axis_tready(rx_ready[g]),.rx_axis_tlast(rx_last[g]),.rx_axis_tuser(rx_user[g]),
   .xgmii_rxd(xgmii_rxd),.xgmii_rxc(xgmii_rxc),.xgmii_txd(xgmii_txd),.xgmii_txc(xgmii_txc),
   .tx_error_underflow(mac_errors[8*g]),.tx_fifo_overflow(mac_errors[8*g+1]),.tx_fifo_bad_frame(mac_errors[8*g+2]),.tx_fifo_good_frame(),
   .rx_error_bad_frame(mac_errors[8*g+3]),.rx_error_bad_fcs(mac_errors[8*g+4]),.rx_fifo_overflow(mac_errors[8*g+5]),.rx_fifo_bad_frame(mac_errors[8*g+6]),.rx_fifo_good_frame(),
   .ptp_ts_96(96'd0),.ptp_ts_step(1'b0),.cfg_ifg(8'd12),.cfg_tx_enable(1'b1),.cfg_rx_enable(1'b1));
  assign mac_errors[8*g+7]=1'b0;
  assign phy_status[8*g+:8]={1'b0,rx_status,high_ber,block_lock,rx_rst,tx_rst,qpll_lock,power_good};
 end endgenerate
 assign led=phy_status[7:0];
endmodule
