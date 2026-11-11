`timescale 1ns/1ps
// 200MHz bounded multicast. REQUEST latches length, LOAD latches one256-bit
// word, SEND independently retires its five copies. Output data has no
// input bypass. Every accepted lane is disabled until the next LOAD.
// For N>0 and no stalls, the last write is2*N+1 edges after start acceptance;
// N=0 completes one edge after start. DONE always follows actual retirement.
(* keep_hierarchy="yes", use_dsp="no" *)
module c209_kv_broadcast_rtl(
 input wire ap_clk,ap_rst,ap_ce,ap_start,ap_continue,
 output wire ap_done,ap_idle,ap_ready,
 input wire [11:0] requests_dout,
 input wire requests_empty_n,
 output wire requests_read,
 input wire [255:0] payload_dout,
 input wire payload_empty_n,
 output wire payload_read,
 output wire [255:0] output0_din,output1_din,output2_din,output3_din,output4_din,
 input wire output0_full_n,output1_full_n,output2_full_n,output3_full_n,output4_full_n,
 output wire output0_write,output1_write,output2_write,output3_write,output4_write
);
 localparam IDLE=3'd0,REQUEST=3'd1,LOAD=3'd2,SEND=3'd3,DONE=3'd4;
 reg [2:0] state;
 reg [11:0] remaining;
 reg [255:0] held;
 reg [4:0] pending;
 wire [4:0] space={output4_full_n,output3_full_n,output2_full_n,output1_full_n,output0_full_n};
 wire [4:0] accepted=pending&space;
 wire [4:0] next_pending=pending&~space;
 wire sending=!ap_rst&&ap_ce&&(state==SEND);
 assign requests_read=!ap_rst&&ap_ce&&(state==REQUEST)&&requests_empty_n;
 assign payload_read=!ap_rst&&ap_ce&&(state==LOAD)&&payload_empty_n;
 assign output0_din=held;
 assign output1_din=held;
 assign output2_din=held;
 assign output3_din=held;
 assign output4_din=held;
 assign output0_write=sending&&accepted[0];
 assign output1_write=sending&&accepted[1];
 assign output2_write=sending&&accepted[2];
 assign output3_write=sending&&accepted[3];
 assign output4_write=sending&&accepted[4];
 assign ap_done=!ap_rst&&(state==DONE);
 assign ap_idle=!ap_rst&&(state==IDLE);
 assign ap_ready=!ap_rst&&ap_ce&&(state==DONE)&&ap_continue;
 always @(posedge ap_clk)begin
  if(ap_rst)begin
   state<=IDLE;remaining<=0;held<=0;pending<=0;
  end else if(ap_ce)begin
   case(state)
    IDLE:if(ap_start)state<=REQUEST;
    REQUEST:if(requests_read)begin
     remaining<=requests_dout;pending<=0;
     state<=requests_dout==0?DONE:LOAD;
    end
    LOAD:if(payload_read)begin
     held<=payload_dout;remaining<=remaining-1'b1;pending<=5'b11111;state<=SEND;
    end
    SEND:begin
     pending<=next_pending;
     if(next_pending==0)state<=remaining==0?DONE:LOAD;
    end
    // A held start cannot duplicate a completed request. Chained continue
    // accepts the next invocation but never reads its FIFO in this state.
    DONE:if(ap_continue)state<=ap_start?REQUEST:IDLE;
    default:begin state<=IDLE;pending<=0;remaining<=0;end
   endcase
  end
 end
endmodule
