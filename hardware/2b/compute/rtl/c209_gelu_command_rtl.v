`timescale 1ns/1ps
// A bounded, single-transaction command publisher in the 200 MHz domain.
// No data-path token passes through this module. Existing GELU data queues
// and the GEMM command publisher are unchanged. Count is unsigned32 modulo
// 2^32: rows * ((columns + 7) modulo 2^32 >> 3) * 4.
// Narrow shift/add multiplication avoids an unregistered wide multiply chain.
// With request/output ready and CE=1: the output transfer is 36 rising edges
// after start acceptance; DONE follows that transfer, never precedes it.
(* keep_hierarchy="yes", use_dsp="no" *)
module c209_gelu_command_rtl(
 input wire ap_clk,ap_rst,ap_ce,ap_start,ap_continue,
 output wire ap_done,ap_idle,ap_ready,
 input wire [64:0] requests_dout,
 input wire requests_empty_n,
 output wire requests_read,
 output wire [33:0] commands_din,
 input wire commands_full_n,
 output wire commands_write
);
 localparam IDLE=0,REQUEST=1,ROUND=2,INITIALIZE=3,MULTIPLY=4,PUBLISH=5,DONE=6;
 reg [2:0] state;
 reg [31:0] rows,columns,rounded_columns,multiplicand,multiplier,accumulator,count;
 reg source;
 reg [5:0] bit_index;
 wire [31:0] addend=multiplier[0]?multiplicand:32'd0;
 wire [31:0] next_accumulator=accumulator+addend;
 assign requests_read=!ap_rst&&ap_ce&&(state==REQUEST)&&requests_empty_n;
 assign commands_din={1'b0,source,count};
 assign commands_write=!ap_rst&&ap_ce&&(state==PUBLISH)&&commands_full_n;
 assign ap_done=!ap_rst&&(state==DONE);
 assign ap_idle=!ap_rst&&(state==IDLE);
 assign ap_ready=!ap_rst&&ap_ce&&(state==DONE)&&ap_continue;
 always @(posedge ap_clk)begin
  if(ap_rst)begin
   state<=IDLE;rows<=0;columns<=0;rounded_columns<=0;
   multiplicand<=0;multiplier<=0;accumulator<=0;count<=0;
   source<=0;bit_index<=0;
  end else if(ap_ce)begin
   case(state)
    IDLE:if(ap_start)state<=REQUEST;
    REQUEST:if(requests_read)begin
     rows<=requests_dout[31:0];columns<=requests_dout[63:32];
     source<=requests_dout[64];state<=ROUND;
    end
    ROUND:begin rounded_columns<=columns+32'd7;state<=INITIALIZE;end
    INITIALIZE:begin
     multiplicand<=rows<<2;multiplier<={3'd0,rounded_columns[31:3]};
     accumulator<=0;bit_index<=0;state<=MULTIPLY;
    end
    MULTIPLY:begin
     accumulator<=next_accumulator;
     multiplicand<=multiplicand<<1;multiplier<=multiplier>>1;
     if(bit_index==31)begin count<=next_accumulator;state<=PUBLISH;end
     else bit_index<=bit_index+1;
    end
    PUBLISH:if(commands_write)state<=DONE;
    // Held start alone cannot duplicate a transaction while DONE is held.
    // A start coincident with continue accepts the NEXT invocation. Its FIFO
    // request is consumed only in REQUEST, independently of upstream stalls.
    DONE:if(ap_continue)state<=ap_start?REQUEST:IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule
