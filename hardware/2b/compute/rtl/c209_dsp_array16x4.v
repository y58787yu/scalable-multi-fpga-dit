`timescale 1ns/1ps
// One physical column bank. CHAIN_ROWS=4 gives four local chains per column;
// the default retains the legacy sixteen-row interface for other users.
// Four banks run together; A RAM is shared in the parent, never replicated.
// Operand distribution retains exactly the original one-cycle latency.
(* keep_hierarchy="yes" *)
module c209_dsp_array16x4 #(parameter PIPELINED_CONTROL=0,parameter CHAIN_ROWS=16)(
 input wire clk,rst,clear,seed_valid,multiply_valid,shift,
 // In the fixed five-shard path these describe the operation at the NEXT
 // P-update edge. OPMODE and each column's CEP register capture together.
 input wire next_clear,next_seed,next_multiply,next_shift,
 input wire [255:0] a,
 input wire [63:0] b,
 input wire [(16/CHAIN_ROWS)*192-1:0] seed,
 output wire [(16/CHAIN_ROWS)*192-1:0] bottom
);
 wire [47:0] cascade[0:63];
 wire [29:0] local_a[0:15];
 wire [17:0] local_b[0:15];
 wire [3:0] column_reset,column_enable;
 genvar r,c,g;
 generate
  for(r=0;r<16;r=r+1)begin:a_distribute
   // Keep physical bank copies, but store the 16 information bits only.
   // Sign extension is local wiring into the DSP, not fourteen extra FFs.
   (* DONT_TOUCH="yes" *) reg [15:0] a_reg;
   always @(posedge clk)a_reg<=a[r*16+:16];
   assign local_a[r]={{14{a_reg[15]}},a_reg};
  end
  for(c=0;c<4;c=c+1)begin:b_distribute
   for(g=0;g<4;g=g+1)begin:groups
    (* DONT_TOUCH="yes" *) reg [15:0] b_reg;
    always @(posedge clk)b_reg<=b[c*16+:16];
    assign local_b[c*4+g]={{2{b_reg[15]}},b_reg};
   end
   if(PIPELINED_CONTROL)begin:local_control
    // Synchronous local reset leaves the top reset driving only local roots.
    // Startup is held by the bank sequencer before any valid operation.
    // Retain the per-column registered operation boundary, while permitting
    // synchronous reset/CE replication for nearby DSP pins. These are not
    // CDC synchronizers and need no physical-optimization DONT_TOUCH barrier.
    (* KEEP="yes",MAX_FANOUT=64 *) reg reset_q=1'b1;
    (* KEEP="yes" *) reg enable_q=1'b0;
    always @(posedge clk)begin
     reset_q<=rst;
     if(reset_q)enable_q<=0;
     else enable_q<=next_clear||next_seed||next_multiply||next_shift;
    end
    assign column_reset[c]=reset_q;
    assign column_enable[c]=enable_q;
   end else begin:legacy_control
    assign column_reset[c]=rst;
    assign column_enable[c]=clear||seed_valid||multiply_valid||shift;
   end
  end
  for(r=0;r<16;r=r+1)begin:rows
   for(c=0;c<4;c=c+1)begin:cols
    // Each chain head injects C; the other rows shift OLD PCOUT. Supplying
    // chain rows in reverse order restores each independent P accumulator.
    wire mode_clear=PIPELINED_CONTROL?next_clear:clear;
    wire mode_seed=PIPELINED_CONTROL?next_seed:seed_valid;
    wire mode_shift=PIPELINED_CONTROL?next_shift:shift;
    wire [8:0] mode=mode_clear?9'h000:
      mode_seed?(r%CHAIN_ROWS==0?9'h030:9'h010):mode_shift?9'h010:9'h025;
    wire [47:0] previous,seed_input,fabric_result;
    if(r%CHAIN_ROWS==0)begin:first_row
     assign previous=48'd0;
     assign seed_input=seed[(r/CHAIN_ROWS)*192+c*48+:48];
    end else begin:following_row
     assign previous=cascade[(r-1)*4+c];
     assign seed_input=48'd0;
    end
    // CREG, OPMODE and CEP retain the original operation alignment.
    c209_dsp_pe #(.C_PIPELINE(r%CHAIN_ROWS==0?1:0),.OPMODE_PIPELINE(PIPELINED_CONTROL)) pe(
     .clk(clk),.rst(column_reset[c]),.p_enable(column_enable[c]),
     .a(local_a[r]),.b(local_b[c*4+r/4]),.seed(seed_input),
     .pcin(previous),.opmode(mode),.p(fabric_result),.pcout(cascade[r*4+c]));
    if(r%CHAIN_ROWS==CHAIN_ROWS-1)
     assign bottom[(r/CHAIN_ROWS)*192+c*48+:48]=fabric_result;
   end
  end
 endgenerate
endmodule
