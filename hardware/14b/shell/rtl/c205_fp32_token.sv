`timescale 1ns/1ps
// Elastic wrapper around Vivado floating_point. No arithmetic is implemented here.
// Operand handshakes are tracked independently; each sidecar is queued exactly once.
module c205_fp32_token #(parameter OP=0, SIDE_WIDTH=1, DEPTH=64)(
 input wire aclk,aresetn,
 input wire [31:0] a,b,input wire [SIDE_WIDTH-1:0] side,
 input wire s_valid,output wire s_ready,
 output wire [31:0] result,output wire [SIDE_WIDTH-1:0] result_side,
 output wire m_valid,input wire m_ready);
 reg [31:0] held_a=0,held_b=0;
 reg pending_a=0,pending_b=0;
 wire a_ready,b_ready,ip_valid,ip_ready;
 wire [31:0] ip_result;
 localparam PTR=$clog2(DEPTH);
 reg [SIDE_WIDTH-1:0] metadata[0:DEPTH-1];
 reg [PTR-1:0] wr=0,rd=0;
 reg [PTR:0] count=0;
 assign s_ready=(!pending_a||a_ready)&&(!pending_b||b_ready)&&(count<DEPTH);
 assign m_valid=ip_valid&&(count!=0);
 assign result=ip_result;
 assign result_side=metadata[rd];
 assign ip_ready=m_ready&&(count!=0);
 wire push=s_valid&&s_ready,pop=m_valid&&m_ready;
 always @(posedge aclk)begin
  if(!aresetn)begin pending_a<=0;pending_b<=0;wr<=0;rd<=0;count<=0;end
  else begin
   if(pending_a&&a_ready)pending_a<=0;
   if(pending_b&&b_ready)pending_b<=0;
   if(push)begin held_a<=a;held_b<=b;pending_a<=1;pending_b<=1;metadata[wr]<=side;wr<=wr+1;end
   if(pop)rd<=rd+1;
   case({push,pop})2'b10:count<=count+1;2'b01:count<=count-1;default:count<=count;endcase
  end
 end
 generate if(OP==0)begin:add
  c205_fp32_add fp(.aclk(aclk),.aresetn(aresetn),
   .s_axis_a_tdata(held_a),.s_axis_a_tvalid(pending_a),.s_axis_a_tready(a_ready),
   .s_axis_b_tdata(held_b),.s_axis_b_tvalid(pending_b),.s_axis_b_tready(b_ready),
   .m_axis_result_tdata(ip_result),.m_axis_result_tvalid(ip_valid),.m_axis_result_tready(ip_ready));
 end else begin:mul
  c205_fp32_mul fp(.aclk(aclk),.aresetn(aresetn),
   .s_axis_a_tdata(held_a),.s_axis_a_tvalid(pending_a),.s_axis_a_tready(a_ready),
   .s_axis_b_tdata(held_b),.s_axis_b_tvalid(pending_b),.s_axis_b_tready(b_ready),
   .m_axis_result_tdata(ip_result),.m_axis_result_tvalid(ip_valid),.m_axis_result_tready(ip_ready));
 end endgenerate
endmodule
