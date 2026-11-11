`timescale 1ns/1ps
// Independent two-entry AXI-Lite AW and W queues.  The CSR owner consumes only
// registered queue heads: neither WDATA nor WSTRB bypasses into its write mux.
// Interface reset clears accepted work; a software CLEAR command must not.
module c209_link_write_capture(
 input wire aclk,aresetn,
 input wire [11:0] awaddr,input wire awvalid,output wire awready,
 input wire [31:0] wdata,input wire [3:0] wstrb,input wire wvalid,output wire wready,
 output wire command_valid,input wire command_ready,
 output wire [11:0] command_address,output wire [31:0] command_data,
 output wire [3:0] command_strobe
);
 reg [11:0] address_q[0:1];
 reg [31:0] data_q[0:1];
 reg [3:0] strobe_q[0:1];
 reg aw_write=0,aw_read=0,w_write=0,w_read=0;
 reg [1:0] aw_count=0,w_count=0;
 assign awready=aresetn&&(aw_count<2);
 assign wready=aresetn&&(w_count<2);
 assign command_valid=aresetn&&(aw_count!=0)&&(w_count!=0);
 assign command_address=address_q[aw_read];
 assign command_data=data_q[w_read];
 assign command_strobe=strobe_q[w_read];
 wire aw_push=awvalid&&awready,w_push=wvalid&&wready;
 wire pop=command_valid&&command_ready;
 always @(posedge aclk)begin
  if(!aresetn)begin
   aw_write<=0;aw_read<=0;w_write<=0;w_read<=0;aw_count<=0;w_count<=0;
  end else begin
   if(aw_push)begin address_q[aw_write]<=awaddr;aw_write<=!aw_write;end
   if(w_push)begin data_q[w_write]<=wdata;strobe_q[w_write]<=wstrb;w_write<=!w_write;end
   if(pop)begin aw_read<=!aw_read;w_read<=!w_read;end
   case({aw_push,pop})2'b10:aw_count<=aw_count+1;2'b01:aw_count<=aw_count-1;default:aw_count<=aw_count;endcase
   case({w_push,pop})2'b10:w_count<=w_count+1;2'b01:w_count<=w_count-1;default:w_count<=w_count;endcase
  end
 end
endmodule
