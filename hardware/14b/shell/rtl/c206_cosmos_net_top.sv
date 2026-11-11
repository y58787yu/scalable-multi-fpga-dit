`timescale 1ns/1ps
module c206_cosmos_net_top(
 input wire clk_125mhz_p,clk_125mhz_n,reset,
 input wire sfp0_rx_p,sfp0_rx_n,output wire sfp0_tx_p,sfp0_tx_n,sfp0_tx_disable_b,
 input wire sfp1_rx_p,sfp1_rx_n,output wire sfp1_tx_p,sfp1_tx_n,sfp1_tx_disable_b,
 input wire sfp2_rx_p,sfp2_rx_n,output wire sfp2_tx_p,sfp2_tx_n,sfp2_tx_disable_b,
 input wire sfp3_rx_p,sfp3_rx_n,output wire sfp3_tx_p,sfp3_tx_n,sfp3_tx_disable_b,
 input wire sfp_mgt_refclk_0_p,sfp_mgt_refclk_0_n,output wire [7:0] led);
 wire logic_clk,logic_resetn;wire [3:0] rx_polarity;wire [31:0] user_tag;
 wire [31:0] phy_status,mac_errors,tx_keep,rx_keep;wire [255:0] tx_data,rx_data;
 wire [3:0] tx_valid,tx_ready,tx_last,tx_user,rx_valid,rx_ready,rx_last,rx_user;
 wire [7:0] node_id;
 wire [31:0] router_dropped,router_forwarded,router_delivered;
 // Direct physical endpoints: P0/P1 TP rank xor1/xor2; P2/P3 next/prev
 // sequence group at the same rank. Header/context checks remain in each link.
 // Keep the legacy CSR fields readable, with no fabric router instantiated.
 assign router_dropped=32'd0;assign router_forwarded=32'd0;assign router_delivered=32'd0;
 USR_ACCESSE2 identity(.CFGCLK(),.DATA(user_tag),.DATAVALID());
 c206_cosmos_net_design_wrapper system(.logic_clk(logic_clk),.logic_resetn(logic_resetn),.user_tag(user_tag),.node_id(node_id),
 .router_dropped(router_dropped),.router_forwarded(router_forwarded),.router_delivered(router_delivered),
 .MAC0_TX_tdata(tx_data[0+:64]),
 .MAC0_TX_tkeep(tx_keep[0+:8]),
 .MAC0_TX_tvalid(tx_valid[0]),
 .MAC0_TX_tready(tx_ready[0]),
 .MAC0_TX_tlast(tx_last[0]),
 .MAC0_TX_tuser(tx_user[0]),
 .MAC0_RX_tdata(rx_data[0+:64]),
 .MAC0_RX_tkeep(rx_keep[0+:8]),
 .MAC0_RX_tvalid(rx_valid[0]),
 .MAC0_RX_tready(rx_ready[0]),
 .MAC0_RX_tlast(rx_last[0]),
 .MAC0_RX_tuser(rx_user[0]),
 .phy_status0(phy_status[0+:8]),
 .mac_errors0(mac_errors[0+:8]),
 .rx_polarity0(rx_polarity[0]),
 .MAC1_TX_tdata(tx_data[64+:64]),
 .MAC1_TX_tkeep(tx_keep[8+:8]),
 .MAC1_TX_tvalid(tx_valid[1]),
 .MAC1_TX_tready(tx_ready[1]),
 .MAC1_TX_tlast(tx_last[1]),
 .MAC1_TX_tuser(tx_user[1]),
 .MAC1_RX_tdata(rx_data[64+:64]),
 .MAC1_RX_tkeep(rx_keep[8+:8]),
 .MAC1_RX_tvalid(rx_valid[1]),
 .MAC1_RX_tready(rx_ready[1]),
 .MAC1_RX_tlast(rx_last[1]),
 .MAC1_RX_tuser(rx_user[1]),
 .phy_status1(phy_status[8+:8]),
 .mac_errors1(mac_errors[8+:8]),
 .rx_polarity1(rx_polarity[1]),
 .MAC2_TX_tdata(tx_data[128+:64]),
 .MAC2_TX_tkeep(tx_keep[16+:8]),
 .MAC2_TX_tvalid(tx_valid[2]),
 .MAC2_TX_tready(tx_ready[2]),
 .MAC2_TX_tlast(tx_last[2]),
 .MAC2_TX_tuser(tx_user[2]),
 .MAC2_RX_tdata(rx_data[128+:64]),
 .MAC2_RX_tkeep(rx_keep[16+:8]),
 .MAC2_RX_tvalid(rx_valid[2]),
 .MAC2_RX_tready(rx_ready[2]),
 .MAC2_RX_tlast(rx_last[2]),
 .MAC2_RX_tuser(rx_user[2]),
 .phy_status2(phy_status[16+:8]),
 .mac_errors2(mac_errors[16+:8]),
 .rx_polarity2(rx_polarity[2]),
 // P3 is an independent SP previous-group endpoint. Its on-wire lane ID
 // is2, matching the peer P2; physical DMA/CSR index remains3.
 .MAC3_TX_tdata(tx_data[192+:64]),
 .MAC3_TX_tkeep(tx_keep[24+:8]),
 .MAC3_TX_tvalid(tx_valid[3]),
 .MAC3_TX_tready(tx_ready[3]),
 .MAC3_TX_tlast(tx_last[3]),
 .MAC3_TX_tuser(tx_user[3]),
 .MAC3_RX_tdata(rx_data[192+:64]),
 .MAC3_RX_tkeep(rx_keep[24+:8]),
 .MAC3_RX_tvalid(rx_valid[3]),
 .MAC3_RX_tready(rx_ready[3]),
 .MAC3_RX_tlast(rx_last[3]),
 .MAC3_RX_tuser(rx_user[3]),
 .phy_status3(phy_status[24+:8]),
 .mac_errors3(mac_errors[24+:8]),
 .rx_polarity3(rx_polarity[3]));
 c205_ethernet_link links(
 .clk_125mhz_p(clk_125mhz_p),.clk_125mhz_n(clk_125mhz_n),.reset(reset),
 .sfp_rx_p({sfp3_rx_p,sfp2_rx_p,sfp1_rx_p,sfp0_rx_p}),.sfp_rx_n({sfp3_rx_n,sfp2_rx_n,sfp1_rx_n,sfp0_rx_n}),
 .sfp_tx_p({sfp3_tx_p,sfp2_tx_p,sfp1_tx_p,sfp0_tx_p}),.sfp_tx_n({sfp3_tx_n,sfp2_tx_n,sfp1_tx_n,sfp0_tx_n}),.sfp_tx_disable_b({sfp3_tx_disable_b,sfp2_tx_disable_b,sfp1_tx_disable_b,sfp0_tx_disable_b}),
 .sfp_mgt_refclk_0_p(sfp_mgt_refclk_0_p),.sfp_mgt_refclk_0_n(sfp_mgt_refclk_0_n),
 .logic_clk(logic_clk),.logic_resetn(logic_resetn),.rx_polarity(rx_polarity),
 .tx_data(tx_data),.tx_keep(tx_keep),.tx_valid(tx_valid),.tx_ready(tx_ready),.tx_last(tx_last),.tx_user(tx_user),
 .rx_data(rx_data),.rx_keep(rx_keep),.rx_valid(rx_valid),.rx_ready(rx_ready),.rx_last(rx_last),.rx_user(rx_user),
 .phy_status(phy_status),.mac_errors(mac_errors),.led(led));
endmodule
