`timescale 1ns/1ps
// Four parallel 192-bit column-bank payloads plus a replicated STOP marker.
// Depth counts ORIGINAL 16-lane result packets, not serialized quarters.
// Every write/read is atomic across all banks, including reset-busy recovery.
(* keep_hierarchy="yes" *)
module c209_mac_result_cdc #(parameter DEPTH=32)(
 input wire wr_clk,rd_clk,rst,
 input wire [767:0] din,
 input wire marker_in,wr_en,
 output wire full_n,
 output wire [767:0] dout,
 output wire marker_out,empty_n,
 input wire rd_en
);
 wire [3:0] bank_full_n,bank_empty_n,bank_marker;
 wire write_fire=wr_en&&full_n;
 wire read_fire=rd_en&&empty_n;
 // The write-domain reset asserts the read gate asynchronously; release is
 // synchronized to rd_clk and ALSO waits for every XPM rd_rst_busy to clear.
 (* ASYNC_REG="TRUE" *)reg [2:0] read_reset_pipe;
 always @(posedge rd_clk or posedge rst)
  if(rst)read_reset_pipe<=3'b111;
  else read_reset_pipe<={read_reset_pipe[1:0],1'b0};
 wire markers_match=(&bank_marker)||(~|bank_marker);
 assign full_n=!rst&&(&bank_full_n);
 // read_reset_pipe already asserts asynchronously and releases on rd_clk.
 // Do not feed raw write-domain reset into read-domain FIFO/SRL enables.
 assign empty_n=!read_reset_pipe[2]&&(&bank_empty_n)&&markers_match;
 assign marker_out=bank_marker[0];
 genvar g;
 generate for(g=0;g<4;g=g+1)begin:column_banks
  wire [192:0] packet;
  c209_mac_async_fifo #(.WIDTH(193),.DEPTH(DEPTH)) fifo(
   .wr_clk(wr_clk),.rd_clk(rd_clk),.rst(rst),
   .din({marker_in,din[g*192+:192]}),.wr_en(write_fire),.full_n(bank_full_n[g]),
   .dout(packet),.rd_en(read_fire),.empty_n(bank_empty_n[g]));
  assign dout[g*192+:192]=packet[191:0];
  assign bank_marker[g]=packet[192];
 end endgenerate
 // synthesis translate_off
 always @(posedge rd_clk)begin
  if(!rst&&!read_reset_pipe[2]&&(&bank_empty_n)&&!markers_match)begin
   $display("C209 result CDC marker mismatch: banks lost packet alignment");
   $stop;
  end
 end
 // synthesis translate_on
endmodule

// Five-shard production transport. Each island reads its C banks directly as
// ordered low/high384-bit beats; no768-bit packet crosses the island boundary.
// DEPTH64 half-packets retains the old32x768 payload capacity. The read-domain
// reset hierarchy is unchanged. Core completion follows its last HIGH beat;
// wrapper result credits still drain every half before STOP/performance.
(* keep_hierarchy="yes" *)
module c209_five_result_cdc384 #(parameter DEPTH=64)(
 input wire wr_clk,rd_clk,rst,
 input wire [383:0] din,
 input wire marker_in,wr_en,
 output wire full_n,
 output wire [383:0] dout,
 output wire marker_out,empty_n,
 input wire rd_en
);
 wire [1:0] bank_full_n,bank_empty_n,bank_marker;
 wire write_fire=wr_en&&full_n;
 wire read_fire=rd_en&&empty_n;
 assign full_n=!rst&&(&bank_full_n);
 (* ASYNC_REG="TRUE" *)reg [2:0] read_reset_pipe;
 always @(posedge rd_clk or posedge rst)
  if(rst)read_reset_pipe<=3'b111;
  else read_reset_pipe<={read_reset_pipe[1:0],1'b0};
 wire markers_match=(&bank_marker)||(~|bank_marker);
 assign empty_n=!read_reset_pipe[2]&&(&bank_empty_n)&&markers_match;
 assign marker_out=bank_marker[0];
 genvar g;
 generate for(g=0;g<2;g=g+1)begin:column_banks
  wire [192:0] received;
  c209_mac_async_fifo #(.WIDTH(193),.DEPTH(DEPTH)) fifo(
   .wr_clk(wr_clk),.rd_clk(rd_clk),.rst(rst),
   .din({marker_in,din[g*192+:192]}),.wr_en(write_fire),.full_n(bank_full_n[g]),
   .dout(received),.rd_en(read_fire),.empty_n(bank_empty_n[g]));
  assign dout[g*192+:192]=received[191:0];
  assign bank_marker[g]=received[192];
 end endgenerate
endmodule
