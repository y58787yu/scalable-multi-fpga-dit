`timescale 1ns/1ps
`default_nettype none
// Fixed 2B TP2 CSR sequencer (legacy module/file name retained for BD reuse).
// A single P1 FUSED round performs (partial + peer) * gate + residual exactly
// once. No SUM producer, scratch tensor, or second-round retirement dependency.
// No payload RAM, arithmetic, DMA replacement,
// extra packet credit, or changes to the existing link/reducer packet ABI.
// CPU owns the target CSRs only while BUSY=0. ERROR is NOT a drain receipt.
module c209_tp4_batch_controller #(
 parameter [31:0] BIT_ID=32'hC2092B10, parameter integer CLOCK_MHZ=200
)(
 (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER="XIL_INTERFACENAME aclk, ASSOCIATED_BUSIF s_axi:m_axi, ASSOCIATED_RESET aresetn, FREQ_HZ 200000000" *) input wire aclk,
 (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER="XIL_INTERFACENAME aresetn, POLARITY ACTIVE_LOW" *) input wire aresetn,
 (* X_INTERFACE_PARAMETER="XIL_INTERFACENAME s_axi, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 10" *) (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi AWADDR" *) input wire [9:0] s_axi_awaddr,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi AWPROT" *) input wire [2:0] s_axi_awprot,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi AWVALID" *) input wire s_axi_awvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi AWREADY" *) output wire s_axi_awready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi WDATA" *) input wire [31:0] s_axi_wdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi WSTRB" *) input wire [3:0] s_axi_wstrb,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi WVALID" *) input wire s_axi_wvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi WREADY" *) output wire s_axi_wready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi BRESP" *) output reg [1:0] s_axi_bresp,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi BVALID" *) output reg s_axi_bvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi BREADY" *) input wire s_axi_bready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi ARADDR" *) input wire [9:0] s_axi_araddr,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi ARPROT" *) input wire [2:0] s_axi_arprot,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi ARVALID" *) input wire s_axi_arvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi ARREADY" *) output wire s_axi_arready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi RDATA" *) output reg [31:0] s_axi_rdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi RRESP" *) output reg [1:0] s_axi_rresp,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi RVALID" *) output reg s_axi_rvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 s_axi RREADY" *) input wire s_axi_rready,
 (* X_INTERFACE_PARAMETER="XIL_INTERFACENAME m_axi, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 32" *) (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi AWADDR" *) output reg [31:0] m_axi_awaddr,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi AWPROT" *) output wire [2:0] m_axi_awprot,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi AWVALID" *) output reg m_axi_awvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi AWREADY" *) input wire m_axi_awready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi WDATA" *) output reg [31:0] m_axi_wdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi WSTRB" *) output wire [3:0] m_axi_wstrb,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi WVALID" *) output reg m_axi_wvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi WREADY" *) input wire m_axi_wready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi BRESP" *) input wire [1:0] m_axi_bresp,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi BVALID" *) input wire m_axi_bvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi BREADY" *) output wire m_axi_bready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi ARADDR" *) output reg [31:0] m_axi_araddr,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi ARPROT" *) output wire [2:0] m_axi_arprot,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi ARVALID" *) output reg m_axi_arvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi ARREADY" *) input wire m_axi_arready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi RDATA" *) input wire [31:0] m_axi_rdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi RRESP" *) input wire [1:0] m_axi_rresp,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi RVALID" *) input wire m_axi_rvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 m_axi RREADY" *) output wire m_axi_rready
);
 reg [31:0] cfg[0:63];
 reg busy,complete,failed,cancel_pending,start_lanes;
 reg [31:0] error_code;
 reg [9:0] aw_hold;reg aw_pending,w_pending;reg [31:0] w_hold;reg [3:0] strb_hold;
 wire [1:0] lane_done;wire [31:0] lane_error0,lane_error1;wire [9:0] retired0,retired1;
 wire [7:0] phase0,phase1;wire [31:0] elapsed0,elapsed1;
 wire [1:0] req_valid,req_write;wire [31:0] req_addr0,req_addr1,req_data0,req_data1;
 reg [1:0] req_ready,response_valid;reg [31:0] response_data;reg [1:0] response_code;
 reg master_busy,master_write,master_lane,round_robin;
 wire choose_lane=req_valid[round_robin]?round_robin:~round_robin;
 assign m_axi_awprot=0;assign m_axi_arprot=0;assign m_axi_wstrb=4'hf;
 assign m_axi_bready=master_busy&&master_write&&!m_axi_awvalid&&!m_axi_wvalid;
 assign m_axi_rready=master_busy&&!master_write&&!m_axi_arvalid;
 assign s_axi_awready=!aw_pending&&!s_axi_bvalid;
 assign s_axi_wready=!w_pending&&!s_axi_bvalid;
 assign s_axi_arready=!s_axi_rvalid;
 wire [63:0] partial_addr={cfg[17],cfg[16]};
 wire [63:0] state_addr={cfg[21],cfg[20]},dest_addr={cfg[23],cfg[22]},gate_addr={cfg[25],cfg[24]};
 wire descriptor_valid=(cfg[5]==128||cfg[5]==256)&&cfg[6][6:0]==0&&cfg[6]<=3968&&cfg[6]+cfg[5]<=4096&&
    cfg[7]>=1&&cfg[7]<=2&&cfg[8]!=0&&cfg[9]<19&&cfg[10]<2&&cfg[11]<28&&cfg[12]<3&&cfg[13]!=0&&
    partial_addr!=0&&state_addr!=0&&dest_addr!=0&&gate_addr!=0&&
    partial_addr[5:0]==0&&state_addr[5:0]==0&&dest_addr[5:0]==0&&gate_addr[5:0]==0&&
    cfg[32]==32'hA0021000&&cfg[33]==32'hA0050000&&cfg[34]==32'hA0201000&&cfg[35]==32'hA0130000;
 integer k,b;
 always @(posedge aclk)begin
  if(!aresetn)begin
   busy<=0;complete<=0;failed<=0;error_code<=0;cancel_pending<=0;start_lanes<=0;
   aw_pending<=0;w_pending<=0;s_axi_bvalid<=0;s_axi_bresp<=0;s_axi_rvalid<=0;s_axi_rdata<=0;s_axi_rresp<=0;
   aw_hold<=0;w_hold<=0;strb_hold<=0;
   for(k=0;k<64;k=k+1)cfg[k]<=0;
  end else begin
   start_lanes<=0;
   if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
   if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
   if(s_axi_awvalid&&s_axi_awready)begin aw_hold<=s_axi_awaddr;aw_pending<=1;end
   if(s_axi_wvalid&&s_axi_wready)begin w_hold<=s_axi_wdata;strb_hold<=s_axi_wstrb;w_pending<=1;end
   if(aw_pending&&w_pending&&!s_axi_bvalid)begin
    aw_pending<=0;w_pending<=0;s_axi_bvalid<=1;s_axi_bresp<=0;
    if(aw_hold[1:0]!=0||aw_hold>=256||strb_hold!=4'hf)s_axi_bresp<=2;
    else if(aw_hold==8)begin
     if(w_hold[2]&&busy)cancel_pending<=1;
     if(w_hold[1]&&!busy)begin complete<=0;failed<=0;error_code<=0;cancel_pending<=0;end
     if(w_hold[0])begin
      if(busy)s_axi_bresp<=2;
      else if(!descriptor_valid)begin complete<=1;failed<=1;error_code<=5;end
      else begin busy<=1;complete<=0;failed<=0;error_code<=0;cancel_pending<=0;start_lanes<=1;end
     end
    end else if(busy)s_axi_bresp<=2;
    else if(aw_hold>=20&&aw_hold<=140)cfg[aw_hold[7:2]]<=w_hold;
    else s_axi_bresp<=2;
   end
   if(s_axi_arvalid&&s_axi_arready)begin
    s_axi_rvalid<=1;s_axi_rresp<=0;
    case(s_axi_araddr)
     0:s_axi_rdata<=32'hC209B201;
     4:s_axi_rdata<={28'd0,(complete&&!failed),failed,complete,busy};
     12:s_axi_rdata<=error_code;
     56:s_axi_rdata<={22'd0,retired0};60:s_axi_rdata<={22'd0,retired1};
     144:s_axi_rdata<={16'd0,phase1,phase0};
     148:s_axi_rdata<=elapsed0;152:s_axi_rdata<=elapsed1;
     default:begin
      if(s_axi_araddr<256&&s_axi_araddr[1:0]==0)s_axi_rdata<=cfg[s_axi_araddr[7:2]];
      else begin s_axi_rdata<=0;s_axi_rresp<=2;end
     end
    endcase
   end
   if(busy&&!start_lanes&&(lane_error0!=0||lane_error1!=0))begin
    cancel_pending<=1;
    if(!failed)begin failed<=1;error_code<=lane_error0!=0?lane_error0:lane_error1;end
   end
   if(busy&&(&lane_done)&&!master_busy&&!start_lanes)begin
    busy<=0;complete<=1;
    if(lane_error0!=0||lane_error1!=0)begin failed<=1;if(!failed)error_code<=lane_error0!=0?lane_error0:lane_error1;end
   end
  end
 end
 // One outstanding AXI-Lite transaction. AW/W are independent and remain
 // stable until individually accepted. Fairness is at transaction boundaries.
 always @(posedge aclk)begin
  if(!aresetn)begin
   master_busy<=0;master_write<=0;master_lane<=0;round_robin<=0;
   req_ready<=0;response_valid<=0;response_data<=0;response_code<=0;
   m_axi_awvalid<=0;m_axi_wvalid<=0;m_axi_arvalid<=0;
   m_axi_awaddr<=0;m_axi_wdata<=0;m_axi_araddr<=0;
  end else begin
   req_ready<=0;response_valid<=0;
   if(m_axi_awvalid&&m_axi_awready)m_axi_awvalid<=0;
   if(m_axi_wvalid&&m_axi_wready)m_axi_wvalid<=0;
   if(m_axi_arvalid&&m_axi_arready)m_axi_arvalid<=0;
   if(!master_busy&&busy&&(|req_valid)&&!(|req_ready)&&!(|response_valid))begin
    master_busy<=1;master_lane<=choose_lane;master_write<=req_write[choose_lane];req_ready[choose_lane]<=1;
    if(req_write[choose_lane])begin
     m_axi_awaddr<=choose_lane?req_addr1:req_addr0;m_axi_wdata<=choose_lane?req_data1:req_data0;
     m_axi_awvalid<=1;m_axi_wvalid<=1;
    end else begin m_axi_araddr<=choose_lane?req_addr1:req_addr0;m_axi_arvalid<=1;end
   end
   if((m_axi_bvalid&&m_axi_bready)||(m_axi_rvalid&&m_axi_rready))begin
    response_valid[master_lane]<=1;response_data<=master_write?0:m_axi_rdata;response_code<=master_write?m_axi_bresp:m_axi_rresp;
    master_busy<=0;round_robin<=~master_lane;
   end
  end
 end
 // Keep the CSR layout, but statically remove the unused SUM sequencer. Its
 // completion is permanently true, not dependent on an unconfigured port.
 assign lane_done[0]=1'b1;assign lane_error0=32'd0;assign retired0=10'd0;
 assign phase0=8'd0;assign elapsed0=32'd0;
 assign req_valid[0]=1'b0;assign req_write[0]=1'b0;assign req_addr0=32'd0;assign req_data0=32'd0;
 c209_tp4_batch_lane #(.BIT_ID(BIT_ID),.CLOCK_MHZ(CLOCK_MHZ)) fused_lane(
  .clk(aclk),.resetn(aresetn),.start(start_lanes),.cancel(cancel_pending),.rows(cfg[5]),.first_row(cfg[6]),
  .node(cfg[7]),.epoch(cfg[8]),.step(cfg[9]),.condition(cfg[10]),.layer(cfg[11]),.operator_id(cfg[12]),.timeout_us(cfg[13]),
  .source(partial_addr),.destination(dest_addr),.residual(state_addr),.gate(gate_addr),
  .link_base(cfg[32]),.dma_base(cfg[33]),.reducer_base(cfg[34]),.residual_base(cfg[35]),
  .done(lane_done[1]),.error(lane_error1),.retired(retired1),.phase(phase1),.elapsed_us(elapsed1),
  .request_valid(req_valid[1]),.request_write(req_write[1]),.request_addr(req_addr1),.request_data(req_data1),
  .request_ready(req_ready[1]),.response_valid(response_valid[1]),.response_data(response_data),.response_code(response_code));
endmodule

module c209_tp4_batch_lane #(parameter [31:0] BIT_ID=32'hC2092B10,parameter integer CLOCK_MHZ=200)(
 input wire clk,resetn,start,cancel,input wire [31:0] rows,first_row,node,epoch,step,condition,layer,operator_id,timeout_us,
 input wire [63:0] source,destination,residual,gate,input wire [31:0] link_base,dma_base,reducer_base,residual_base,
 output reg done,output reg [31:0] error,output reg [9:0] retired,output wire [7:0] phase,output reg [31:0] elapsed_us,
 output wire request_valid,output reg request_write,output reg [31:0] request_addr,request_data,input wire request_ready,
 input wire response_valid,input wire [31:0] response_data,input wire [1:0] response_code
);
 reg [7:0] pc;reg waiting,gate_loaded;reg [8:0] row;
 reg [31:0] saved,age_us;reg [63:0] tx_before;reg [15:0] tick;
 // Source/destination/residual are tile-local base addresses. first_row is
 // carried only in packet identity; it must not be added to the pointers twice.
 wire [31:0] bytes=32'd8192;
 wire [63:0] offset=(64'd8192*row);
 wire [63:0] src=source+offset,dst=destination+offset,res=residual+offset,gat=gate;
 wire [31:0] chunk=first_row+row;
 wire [31:0] packet_id=32'h40000000|chunk;
 wire [31:0] peer=((node-1)^32'd1)+1;
 assign phase=pc;
 assign request_valid=!done&&!waiting&&pc!=0&&pc!=83&&!cancel;
 task fail;input [31:0] code;begin done<=1;error<=code;waiting<=0;end endtask
 always @*begin
  request_write=1;request_addr=link_base;request_data=0;
  case(pc)
   1:begin request_addr=reducer_base+8;request_data=2;end
   2:begin request_addr=reducer_base+36;request_data=0;end
   3:begin request_addr=reducer_base+12;request_data=8192;end
   4:begin request_addr=reducer_base+8;request_data=4;end
   5:begin request_addr=residual_base+4;request_data=4096;end
   6:begin request_addr=residual_base+24;request_data=gat[31:0];end
   7:begin request_addr=residual_base+28;request_data=gat[63:32];end
   8:begin request_addr=residual_base+40;request_data=8192;end
   9:begin request_write=0;request_addr=residual_base+4;end
   10:begin request_write=0;request_addr=reducer_base+4;end
   20:begin request_addr=reducer_base+8;request_data=2;end
   21:begin request_addr=reducer_base+36;request_data=0;end
   22:begin request_addr=reducer_base+12;request_data=bytes;end
   23:begin request_addr=reducer_base+16;request_data=0;end
   24:begin request_addr=reducer_base+20;request_data=1;end
   25:begin request_addr=dma_base+52;request_data=4096;end
   26:begin request_addr=dma_base+72;request_data=dst[31:0];end
   27:begin request_addr=dma_base+76;request_data=dst[63:32];end
   28:begin request_addr=dma_base+88;request_data=bytes;end
   29:begin request_addr=link_base+156;request_data=condition;end
   30:begin request_addr=link_base+160;request_data=epoch;end
   31:begin request_addr=link_base+164;request_data=step;end
   32:begin request_addr=link_base+168;request_data=condition;end
   33:begin request_addr=link_base+172;request_data=layer;end
   34:begin request_addr=link_base+176;request_data=operator_id;end
   35:begin request_addr=link_base+184;request_data=1;end
   36:begin request_addr=link_base+180;request_data=chunk;end
   37:begin request_addr=link_base+32;request_data=packet_id;end
   38:begin request_addr=link_base+36;request_data=bytes;end
   39:begin request_addr=link_base+40;request_data=1;end
   40:begin request_addr=link_base+44;request_data=bytes>>2;end
   41:begin request_addr=link_base+4;request_data=4;end
   42:begin request_addr=reducer_base+8;request_data=1;end
   43:begin request_addr=residual_base+4;request_data=4096;end
   44:begin request_addr=residual_base+24;request_data=res[31:0];end
   45:begin request_addr=residual_base+28;request_data=res[63:32];end
   46:begin request_addr=residual_base+40;request_data=bytes;end
   47:begin request_write=0;request_addr=link_base+4;end
   48,54,61,67:begin request_write=0;request_addr=link_base+104;end
   49,62:begin request_addr=link_base+16;request_data=packet_id;end
   50:begin request_addr=link_base+20;request_data=0;end
   51:begin request_addr=link_base+24;request_data=2;end
   52:begin request_addr=link_base+28;request_data=0;end
   53:begin request_addr=link_base+4;request_data=16;end
   55,56,76:begin request_write=0;request_addr=link_base+4;end
   57:begin request_write=0;request_addr=link_base+128;end
   58:begin request_write=0;request_addr=link_base+48;end
   59:begin request_write=0;request_addr=link_base+52;end
   60:begin request_addr=link_base+132;request_data=1;end
   63:begin request_addr=link_base+20;request_data=bytes;end
   64:begin request_addr=link_base+24;request_data=1;end
   65:begin request_addr=link_base+28;request_data=bytes>>2;end
   66:begin request_addr=link_base+4;request_data=1;end
   68:begin request_addr=dma_base+4;request_data=4096;end
   69:begin request_addr=dma_base+24;request_data=src[31:0];end
   70:begin request_addr=dma_base+28;request_data=src[63:32];end
   71:begin request_addr=dma_base+40;request_data=bytes;end
   72:begin request_write=0;request_addr=dma_base+4;end
   73:begin request_write=0;request_addr=dma_base+52;end
   74:begin request_write=0;request_addr=residual_base+4;end
   75:begin request_write=0;request_addr=reducer_base+4;end
   77:begin request_write=0;request_addr=link_base+64;end
   78:begin request_write=0;request_addr=reducer_base+28;end
   79:begin request_write=0;request_addr=reducer_base+32;end
   80:begin request_write=0;request_addr=link_base+76;end
   81:begin request_write=0;request_addr=link_base+80;end
   82:begin request_write=0;request_addr=link_base+212;end
   100:begin request_write=0;request_addr=link_base;end
   101:begin request_write=0;request_addr=reducer_base;end
   102:begin request_write=0;request_addr=link_base+124;end
   103:begin request_write=0;request_addr=link_base+4;end
   104:begin request_write=0;request_addr=reducer_base+4;end
   105:begin request_write=0;request_addr=dma_base+40;end
   106:begin request_write=0;request_addr=dma_base+4;end
   107:begin request_write=0;request_addr=dma_base+88;end
   108:begin request_write=0;request_addr=dma_base+52;end
   109:begin request_write=0;request_addr=residual_base+40;end
   110:begin request_write=0;request_addr=residual_base+4;end
   111:begin request_addr=link_base+144;request_data=node;end
   112:begin request_addr=link_base+148;request_data=peer;end
   // Preserve Y3's full64-bit TX-start timestamp. A low-word-only compare
   // aliases when two starts are exactly2^32 ticks apart. Snapshot both
   // halves before issuing the local START. No other writer owns this link
   // while the batch runs; the timestamp is therefore stable for the read.
   // Poll the high half only when the low half has not changed.
   113,114,115,116:begin request_write=0;request_addr=link_base+108;end
   default:begin request_write=0;request_addr=link_base;end
  endcase
 end
 always @(posedge clk)begin
  if(!resetn)begin done<=1;error<=0;retired<=0;pc<=0;waiting<=0;gate_loaded<=0;row<=0;saved<=0;tx_before<=0;age_us<=0;elapsed_us<=0;tick<=0;end
  else if(start)begin done<=0;error<=0;retired<=0;pc<=100;waiting<=0;gate_loaded<=0;row<=0;age_us<=0;elapsed_us<=0;tick<=0;end
  else if(!done)begin
   if(tick==CLOCK_MHZ-1)begin tick<=0;age_us<=age_us+1;if(elapsed_us!=32'hffffffff)elapsed_us<=elapsed_us+1;end else tick<=tick+1;
   if(request_ready)waiting<=1;
   if(cancel)fail(15);
   else if(age_us>=timeout_us)fail(1);
   else if(pc==0&&!waiting)begin
    if(!gate_loaded)begin pc<=1;age_us<=0;end
    else begin pc<=20;age_us<=0;end
   end else if(pc==83)begin
    retired<=retired+1;
    if(row+1==rows)begin row<=0;gate_loaded<=0;done<=1;pc<=0;end
    else begin row<=row+1;pc<=0;end
   end else if(response_valid&&waiting)begin
    waiting<=0;
    if(response_code!=0)fail(3);
    else case(pc)
     9:if(response_data&32'h70)fail(3);else if(response_data&4096)pc<=10;
     10:if(response_data&4)fail(3);else if(response_data&8)begin gate_loaded<=1;pc<=0;age_us<=0;end
     20:pc<=row==0?21:25;
     28:pc<=row==0?29:36;
     42:pc<=43;
     47:if(response_data&8)fail(3);else if(!(response_data&1))pc<=48;
     48:begin tx_before[31:0]<=response_data;pc<=113;end
     54:if(response_data!=tx_before[31:0])pc<=55;else pc<=114;
     55:if(response_data&8)fail(3);else if((response_data&3)==2)pc<=56;
     56:if(response_data&8)fail(3);else if(response_data&64)pc<=57;
     57:pc<=response_data!=0?58:56;
     58:if(response_data!=2)fail(7);else pc<=59;
     59:if(response_data!=packet_id)fail(7);else pc<=60;
     61:begin tx_before[31:0]<=response_data;pc<=115;end
     67:if(response_data!=tx_before[31:0])pc<=68;else pc<=116;
     72:if(response_data&32'h70)fail(3);else if(response_data&4096)pc<=73;
     73:if(response_data&32'h70)fail(3);else if(response_data&4096)pc<=74;
     74:if(response_data&32'h70)fail(3);else if(response_data&4096)pc<=75;
     75:if(response_data&4)fail(3);else if(response_data&2)pc<=76;
     76:if(response_data&8)fail(3);else if(!(response_data&1)&&(response_data&4))pc<=77;
     77,78,79:if(response_data!=bytes)fail(3);else pc<=pc+1;
     80,81,82:if(response_data!=0)fail(3);else pc<=pc+1;
     100:if(response_data!=32'hC206E001)fail(12);else pc<=101;
     101:if(response_data!=32'hC206F001)fail(12);else pc<=102;
     102:if(response_data!=BIT_ID)fail(12);else pc<=103;
     103:if(response_data&9)fail(4);else if(!(response_data&32))fail(2);else pc<=104;
     104:if(response_data&5)fail(4);else pc<=105;
     105,107,109:begin saved<=response_data;pc<=pc+1;end
     106:if((response_data&32'h70)||(saved!=0&&!(response_data&3)))fail(4);else pc<=107;
     108:if((response_data&32'h70)||(saved!=0&&!(response_data&3)))fail(4);else pc<=109;
     110:if((response_data&32'h70)||(saved!=0&&!(response_data&3)))fail(4);else pc<=111;
     112:pc<=0;
     113:begin tx_before[63:32]<=response_data;pc<=49;end
     114:if(response_data!=tx_before[63:32])pc<=55;else pc<=54;
     115:begin tx_before[63:32]<=response_data;pc<=62;end
     116:if(response_data!=tx_before[63:32])pc<=68;else pc<=67;
     default:pc<=pc+1;
    endcase
   end
  end
 end
endmodule
`default_nettype wire
