`timescale 1ns/1ps
// Standard XPM Gray-pointer asynchronous FIFO, not a multicycle bus crossing.
// Both clocks must run through coordinated reset. rst is local wr-domain reset.
module c209_mac_async_fifo #(parameter WIDTH=256,DEPTH=32)(
 input wire wr_clk,rd_clk,rst,
 input wire [WIDTH-1:0] din,input wire wr_en,output wire full_n,
 output wire [WIDTH-1:0] dout,input wire rd_en,output wire empty_n
);
 wire full,empty,wr_busy,rd_busy;
 assign full_n=!full&&!wr_busy;
 assign empty_n=!empty&&!rd_busy;
 xpm_fifo_async #(
  .CDC_SYNC_STAGES(3),.DOUT_RESET_VALUE("0"),.ECC_MODE("no_ecc"),
  .FIFO_MEMORY_TYPE("auto"),.FIFO_READ_LATENCY(0),.FIFO_WRITE_DEPTH(DEPTH),
  .FULL_RESET_VALUE(0),.PROG_EMPTY_THRESH(10),.PROG_FULL_THRESH(10),
  .RD_DATA_COUNT_WIDTH($clog2(DEPTH)+1),.READ_DATA_WIDTH(WIDTH),
  .READ_MODE("fwft"),.RELATED_CLOCKS(0),.SIM_ASSERT_CHK(1),
  .USE_ADV_FEATURES("0000"),.WAKEUP_TIME(0),
  .WRITE_DATA_WIDTH(WIDTH),.WR_DATA_COUNT_WIDTH($clog2(DEPTH)+1)
 ) fifo (
  .rst(rst),.wr_clk(wr_clk),.rd_clk(rd_clk),.din(din),
  .wr_en(wr_en&&full_n),.rd_en(rd_en&&empty_n),.dout(dout),
  .full(full),.empty(empty),.wr_rst_busy(wr_busy),.rd_rst_busy(rd_busy),
  .sleep(1'b0),.injectdbiterr(1'b0),.injectsbiterr(1'b0)
 );
endmodule
