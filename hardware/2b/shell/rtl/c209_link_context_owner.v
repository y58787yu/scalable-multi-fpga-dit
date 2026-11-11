`timescale 1ns/1ps
// All five complete collective tuples stay inside this owner.  Its external
// data interfaces are one 32-bit CSR word and one 64-bit header word, not a
// broadcast of the 224-bit arrays into the link's status/control datapath.
// Separate owners preserve the original capture edge and CLEAR retention.
// There is no hash, shortened identity, or speculative READY credit.
module c209_link_context_owner #(parameter LANE_ID=0)(
 input wire aclk,aresetn,clear,
 input wire cfg_write,input wire [11:0] cfg_address,
 input wire [31:0] cfg_data,input wire [3:0] cfg_strobe,
 input wire [11:0] csr_address,output reg [31:0] csr_data,
 input wire tx_capture,rx_capture,ready_capture,
 input wire [15:0] tx_index,output reg [63:0] tx_header,
 output wire [7:0] tx_lane,
 input wire frame_take,input wire [15:0] frame_index,input wire [7:0] frame_kind,
 input wire [63:0] frame_data,output reg header_bad,
 output wire ready_matches_rx,frame_matches_rx,pending_matches_frame,
 output wire same_namespace,frame_older
);
 localparam [31:0] LOGICAL_LANE=LANE_ID;
 wire [31:0] context_cfg[0:6],group_cfg;
 reg [31:0] tx_context[0:6],tx_group=0;
 wire [31:0] rx_context[0:6],rx_group;
 reg [31:0] frame_context[0:6],frame_group=0;
 reg [31:0] ready_context[0:6],ready_group=0;
 // One CE per complete word. Byte strobes retain old bytes on D; they must
 // not become four independent eight-FF control sets after synthesis/CE8.
 c209_link_cfg_word cfg_group_word(.aclk(aclk),.aresetn(aresetn),
  .word_write(cfg_write&&cfg_address[11:2]==39),.write_data(cfg_data),
  .write_strobe(cfg_strobe),.reset_value(32'd0),.value(group_cfg));
 genvar c;
 generate for(c=0;c<7;c=c+1)begin:cfg_words
  c209_link_cfg_word cfg_word(.aclk(aclk),.aresetn(aresetn),
   .word_write(cfg_write&&cfg_address[11:2]==40+c),.write_data(cfg_data),
   .write_strobe(cfg_strobe),.reset_value(c==6?LOGICAL_LANE:32'd0),.value(context_cfg[c]));
 end endgenerate
 // Local banks publish every word on the same accepted event.  Unused data
 // registers are not reset by CLEAR; valid/armed state lives in the link owner.
 always @(posedge aclk)begin:tx_owner
  if(aresetn&&!clear&&tx_capture)begin
   tx_group<=group_cfg;
   tx_context[0]<=context_cfg[0];tx_context[1]<=context_cfg[1];tx_context[2]<=context_cfg[2];
   tx_context[3]<=context_cfg[3];tx_context[4]<=context_cfg[4];tx_context[5]<=context_cfg[5];tx_context[6]<=context_cfg[6];
  end
 end
 // Cold reset remains observable as a zero ARM snapshot. CLEAR retains it.
 c209_link_cfg_word rx_group_word(.aclk(aclk),.aresetn(aresetn),
  .word_write(!clear&&rx_capture),.write_data(group_cfg),.write_strobe(4'hf),
  .reset_value(32'd0),.value(rx_group));
 generate for(c=0;c<7;c=c+1)begin:rx_words
  c209_link_cfg_word rx_word(.aclk(aclk),.aresetn(aresetn),
   .word_write(!clear&&rx_capture),.write_data(context_cfg[c]),.write_strobe(4'hf),
   .reset_value(32'd0),.value(rx_context[c]));
 end endgenerate
 always @(posedge aclk)begin:frame_owner
  if(aresetn&&!clear&&frame_take)begin
   case(frame_index)
    6:begin frame_context[0]<=frame_data[31:0];frame_context[1]<=frame_data[63:32];end
    7:begin frame_context[2]<=frame_data[31:0];frame_context[3]<=frame_data[63:32];end
    8:begin frame_context[4]<=frame_data[31:0];frame_context[5]<=frame_data[63:32];end
    9:begin frame_context[6]<=frame_data[31:0];frame_group<=frame_data[63:32];end
   endcase
  end
 end
 always @(posedge aclk)begin:ready_owner
  if(aresetn&&!clear&&ready_capture)begin
   ready_group<=frame_group;
   ready_context[0]<=frame_context[0];ready_context[1]<=frame_context[1];ready_context[2]<=frame_context[2];
   ready_context[3]<=frame_context[3];ready_context[4]<=frame_context[4];ready_context[5]<=frame_context[5];ready_context[6]<=frame_context[6];
  end
 end
 assign tx_lane=tx_context[6][7:0];
 always @*begin
  tx_header=0;
  case(tx_index)
   6:tx_header={tx_context[1],tx_context[0]};
   7:tx_header={tx_context[3],tx_context[2]};
   8:tx_header={tx_context[5],tx_context[4]};
   9:tx_header={tx_group,tx_context[6]};
  endcase
 end
 always @*begin
  csr_data=0;
  case(csr_address[11:2])
   39:csr_data=group_cfg;
   40:csr_data=context_cfg[0];41:csr_data=context_cfg[1];42:csr_data=context_cfg[2];
   43:csr_data=context_cfg[3];44:csr_data=context_cfg[4];45:csr_data=context_cfg[5];46:csr_data=context_cfg[6];
   51:csr_data=ready_context[5];52:csr_data=ready_context[0];
  endcase
 end
 // Per-header-word validation is combinational on the same beat as before.
 // Data never escapes before the last identity word has been checked.
 always @*begin
  header_bad=0;
  case(frame_index)
   6:begin
    if(frame_kind<2)header_bad=frame_data!={rx_context[1],rx_context[0]};
    else if(frame_kind!=2)header_bad=frame_data!={context_cfg[1],context_cfg[0]};
   end
   7:begin
    if(frame_kind<2)header_bad=frame_data!={rx_context[3],rx_context[2]};
    else if(frame_kind!=2)header_bad=frame_data!={context_cfg[3],context_cfg[2]};
   end
   8:begin
    if(frame_kind<2)header_bad=frame_data!={rx_context[5],rx_context[4]};
    else if(frame_kind!=2)header_bad=frame_data!={context_cfg[5],context_cfg[4]};
   end
   9:begin
    if(frame_kind<2)header_bad=frame_data!={rx_group,rx_context[6]};
    else if(frame_kind==2)header_bad=frame_data[31:0]!=LOGICAL_LANE||frame_data[63:32]!=frame_context[2];
    else header_bad=frame_data!={group_cfg,context_cfg[6]};
   end
  endcase
 end
 assign ready_matches_rx=ready_group==rx_group&&
  ready_context[0]==rx_context[0]&&ready_context[1]==rx_context[1]&&ready_context[2]==rx_context[2]&&
  ready_context[3]==rx_context[3]&&ready_context[4]==rx_context[4]&&ready_context[5]==rx_context[5]&&ready_context[6]==rx_context[6];
 assign frame_matches_rx=frame_group==rx_group&&
  frame_context[0]==rx_context[0]&&frame_context[1]==rx_context[1]&&frame_context[2]==rx_context[2]&&
  frame_context[3]==rx_context[3]&&frame_context[4]==rx_context[4]&&frame_context[5]==rx_context[5]&&frame_context[6]==rx_context[6];
 assign pending_matches_frame=ready_group==frame_group&&
  ready_context[0]==frame_context[0]&&ready_context[1]==frame_context[1]&&ready_context[2]==frame_context[2]&&
  ready_context[3]==frame_context[3]&&ready_context[4]==frame_context[4]&&ready_context[5]==frame_context[5]&&ready_context[6]==frame_context[6];
 assign same_namespace=frame_group==rx_group&&frame_context[2]==rx_context[2]&&
  (frame_group!=32'h80||(frame_context[3]==rx_context[3]&&frame_context[4]==rx_context[4]));
 assign frame_older=(frame_context[0]<rx_context[0])||
  (frame_context[0]==rx_context[0]&&frame_context[1]<rx_context[1])||
  (frame_context[0]==rx_context[0]&&frame_context[1]==rx_context[1]&&frame_context[3]<rx_context[3])||
  (frame_context[0]==rx_context[0]&&frame_context[1]==rx_context[1]&&frame_context[3]==rx_context[3]&&frame_context[4]<rx_context[4])||
  (frame_context[0]==rx_context[0]&&frame_context[1]==rx_context[1]&&frame_context[3]==rx_context[3]&&frame_context[4]==rx_context[4]&&frame_context[5]<rx_context[5]);
endmodule

// A local CSR/snapshot word, not a global reset or enable transform. The
// synchronous cold-reset assignment is deliberately data, selected by one
// word CE. No reset latency is added and software CLEAR never reaches here.
// DIRECT_ENABLE selects the outer word event instead of a byte-strobe event;
// EXTRACT_RESET keeps reset-value selection on D. Native OOC pin auditing is
// required to prove the resulting C/CE/R/S topology; attributes are not proof.
(* keep_hierarchy="yes" *)
module c209_link_cfg_word #(parameter WIDTH=32,parameter [WIDTH-1:0] INIT=0)(
 input wire aclk,aresetn,word_write,
 input wire [WIDTH-1:0] write_data,reset_value,
 input wire [3:0] write_strobe,
 output wire [WIDTH-1:0] value
);
 (* direct_enable="yes" *) wire word_update=!aresetn||word_write;
 (* extract_reset="no" *) reg [WIDTH-1:0] stored=INIT;
 wire [WIDTH-1:0] byte_merged;
 genvar b;
 generate for(b=0;b<WIDTH/8;b=b+1)begin:bytes
  assign byte_merged[b*8+:8]=write_strobe[b]?write_data[b*8+:8]:stored[b*8+:8];
 end endgenerate
 always @(posedge aclk)begin
  if(word_update)stored<=aresetn?byte_merged:reset_value;
 end
 assign value=stored;
endmodule
