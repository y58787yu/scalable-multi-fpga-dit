`timescale 1ns/1ps
// Shared slow-domain 32xK panel transpose, K=16..128 in multiples of16.
// Two slots, sixteen explicit 512x16 SDP BRAM banks. Row-major DDR words
// enter once; one vector of16 rows leaves each cycle after pipeline fill.
// The ten instances serve GEMM/QK/PV, replacing their private HLS transposers.
(* keep_hierarchy="yes" *)
module c209_panel_transposer(
 input wire clk,rst,
 input wire cfg_valid,output wire cfg_ready,input wire [8:0] cfg_rows,cfg_k,
 input wire [255:0] input_data,input wire input_valid,output wire input_ready,
 output wire [255:0] output_data,output wire output_valid,input wire output_ready
);
 wire cfg_full,cfg_empty,cfg_wr_busy,cfg_rd_busy,cfg_pop;
 wire [17:0] cfg_data;
 assign cfg_ready=!cfg_full&&!cfg_wr_busy&&!rst;
 reg job_valid;
 reg [3:0] panels_left,words_per_row;
 reg [8:0] job_k;
 assign cfg_pop=!rst&&!job_valid&&!cfg_empty&&!cfg_rd_busy;
 xpm_fifo_sync #(
  .FIFO_MEMORY_TYPE("distributed"),.FIFO_WRITE_DEPTH(16),
  .WRITE_DATA_WIDTH(18),.READ_DATA_WIDTH(18),.READ_MODE("fwft"),.FIFO_READ_LATENCY(0),
  .WR_DATA_COUNT_WIDTH(5),.RD_DATA_COUNT_WIDTH(5),.USE_ADV_FEATURES("0000"),
  .DOUT_RESET_VALUE("0"),.FULL_RESET_VALUE(0),.PROG_FULL_THRESH(10),.PROG_EMPTY_THRESH(10),
  .ECC_MODE("no_ecc"),.SIM_ASSERT_CHK(1),.WAKEUP_TIME(0)
 ) geometry_fifo(
  .wr_clk(clk),.rst(rst),.din({cfg_rows,cfg_k}),.wr_en(cfg_valid&&cfg_ready),.full(cfg_full),
  .dout(cfg_data),.rd_en(cfg_pop),.empty(cfg_empty),.wr_rst_busy(cfg_wr_busy),.rd_rst_busy(cfg_rd_busy),
  .sleep(1'b0),.injectdbiterr(1'b0),.injectsbiterr(1'b0)
 );

 reg [1:0] ready_slots;
 reg write_slot,read_slot,read_half;
 reg [4:0] write_row;
 reg [3:0] write_word;
 reg [6:0] read_k;
 reg [8:0] slot_k[0:1];
 reg raw_valid,out_valid,raw_last,out_last,raw_slot,out_slot;
 reg [3:0] raw_rotation;
 reg [255:0] raw_data,out_data;
 wire out_advance=!out_valid||output_ready;
 wire raw_advance=!raw_valid||out_advance;
 wire read_issue=!rst&&ready_slots[read_slot]&&raw_advance;
 wire read_last=read_half&&({2'b00,read_k}+9'd1==slot_k[read_slot]);
 assign input_ready=!rst&&job_valid&&!ready_slots[write_slot];
 wire write_fire=input_valid&&input_ready;
 wire write_last=write_row==31&&write_word+1'b1==words_per_row;
 assign output_data=out_data;
 assign output_valid=out_valid&&!rst;

 function [255:0] rotate_left;
  input [255:0] word;input [3:0] amount;
  reg [255:0] a,b,c;
  begin
   a=amount[0]?{word[239:0],word[255:240]}:word;
   b=amount[1]?{a[223:0],a[255:224]}:a;
   c=amount[2]?{b[191:0],b[255:192]}:b;
   rotate_left=amount[3]?{c[127:0],c[255:128]}:c;
  end
 endfunction
 function [255:0] rotate_right;
  input [255:0] word;input [3:0] amount;
  reg [255:0] a,b,c;
  begin
   a=amount[0]?{word[15:0],word[255:16]}:word;
   b=amount[1]?{a[31:0],a[255:32]}:a;
   c=amount[2]?{b[63:0],b[255:64]}:b;
   rotate_right=amount[3]?{c[127:0],c[255:128]}:c;
  end
 endfunction
 wire [255:0] skew=rotate_left(input_data,write_row[3:0]);
 wire [8:0] write_addr={write_slot,write_word[2:0],write_row};
 genvar g;
 generate for(g=0;g<16;g=g+1)begin:banks
  localparam [3:0] BANK=g;
  wire [3:0] row_low=BANK-read_k[3:0];
  wire [8:0] read_addr={read_slot,read_k[6:4],read_half,row_low};
  (* ram_style="block" *)reg [15:0] memory[0:511];
  // Canonical one-write/one-read form; no reset on BRAM payload.
  always @(posedge clk)begin
   if(write_fire)memory[write_addr]<=skew[g*16+:16];
   if(read_issue)raw_data[g*16+:16]<=memory[read_addr];
  end
 end endgenerate

 always @(posedge clk)begin
  if(rst)begin
   job_valid<=0;panels_left<=0;words_per_row<=0;job_k<=0;
   ready_slots<=0;write_slot<=0;read_slot<=0;write_row<=0;write_word<=0;
   read_k<=0;read_half<=0;raw_valid<=0;out_valid<=0;
   raw_last<=0;out_last<=0;raw_slot<=0;out_slot<=0;raw_rotation<=0;
   slot_k[0]<=0;slot_k[1]<=0;
  end else begin
   if(cfg_pop)begin
    job_valid<=1;panels_left<=(cfg_data[17:9]+9'd31)>>5;
    words_per_row<=(cfg_data[8:0]+9'd15)>>4;job_k<=cfg_data[8:0];
   end
   if(write_fire)begin
    if(write_word+1'b1==words_per_row)begin write_word<=0;write_row<=write_row+1'b1;end
    else write_word<=write_word+1'b1;
    if(write_last)begin
     ready_slots[write_slot]<=1;slot_k[write_slot]<=job_k;write_slot<=!write_slot;
     write_row<=0;write_word<=0;panels_left<=panels_left-1'b1;
     if(panels_left==1)job_valid<=0;
    end
   end
   if(raw_advance)begin
    raw_valid<=read_issue;
    if(read_issue)begin
     raw_rotation<=read_k[3:0];raw_last<=read_last;raw_slot<=read_slot;
     read_half<=!read_half;
     if(read_half)begin
      if(read_last)begin read_k<=0;read_slot<=!read_slot;end
      else read_k<=read_k+1'b1;
     end
    end
   end
   if(out_advance)begin
    out_valid<=raw_valid;
    if(raw_valid)begin out_data<=rotate_right(raw_data,raw_rotation);out_last<=raw_last;out_slot<=raw_slot;end
   end
   if(out_valid&&output_ready&&out_last)ready_slots[out_slot]<=0;
  end
 end
endmodule
