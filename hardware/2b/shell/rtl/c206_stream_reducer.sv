`timescale 1ns/1ps
// One independent 64-bit lane. Two FP32 elements per token:
// round32(round32(local+peer)*gate[channel])+residual (three distinct operations).
// Gate BRAM is loaded once per logical projection through S_AXIS_RESIDUAL and
// retained across row-tile jobs/CLEAR. No peer DDR staging.
// SUM_ONLY keeps the first TP4 sum and raw transfer, with no gate/residual path.
// BULK_ONLY removes unused FP/gate hardware from reserved lanes2/3.
// It retains raw byte transfer, both-leg completion and the existing CSR map.
module c206_stream_reducer #(parameter BULK_ONLY=0,parameter SUM_ONLY=0)(
 (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF s_axi:s_axis_local:s_axis_peer:s_axis_residual:m_axis_result, ASSOCIATED_RESET aresetn" *)
 input wire aclk,
 (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *)
 input wire aresetn,
 input wire [11:0] s_axi_awaddr,input wire [2:0] s_axi_awprot,input wire s_axi_awvalid,output wire s_axi_awready,
 input wire [31:0] s_axi_wdata,input wire [3:0] s_axi_wstrb,input wire s_axi_wvalid,output wire s_axi_wready,
 output reg [1:0] s_axi_bresp=0,output reg s_axi_bvalid=0,input wire s_axi_bready,
 input wire [11:0] s_axi_araddr,input wire [2:0] s_axi_arprot,input wire s_axi_arvalid,output wire s_axi_arready,
 output reg [31:0] s_axi_rdata=0,output reg [1:0] s_axi_rresp=0,output reg s_axi_rvalid=0,input wire s_axi_rready,
 input wire [63:0] s_axis_local_tdata,input wire [7:0] s_axis_local_tkeep,input wire s_axis_local_tvalid,output wire s_axis_local_tready,input wire s_axis_local_tlast,
 input wire [63:0] s_axis_peer_tdata,input wire [7:0] s_axis_peer_tkeep,input wire s_axis_peer_tvalid,output wire s_axis_peer_tready,input wire s_axis_peer_tlast,
 input wire [63:0] s_axis_residual_tdata,input wire [7:0] s_axis_residual_tkeep,input wire s_axis_residual_tvalid,output wire s_axis_residual_tready,input wire s_axis_residual_tlast,
 output wire [63:0] m_axis_result_tdata,output wire [7:0] m_axis_result_tkeep,output wire m_axis_result_tvalid,input wire m_axis_result_tready,output wire m_axis_result_tlast);
 reg aw_pending=0,w_pending=0;
 reg [11:0] aw_address=0;
 reg [31:0] w_value=0;reg [3:0] w_strobe=0;
 assign s_axi_awready=!aw_pending&&!s_axi_bvalid;
 assign s_axi_wready=!w_pending&&!s_axi_bvalid;
 assign s_axi_arready=!s_axi_rvalid;
 wire aw_take=s_axi_awvalid&&s_axi_awready,w_take=s_axi_wvalid&&s_axi_wready;
 wire write_fire=(aw_pending||aw_take)&&(w_pending||w_take)&&!s_axi_bvalid;
 wire [11:0] write_address=aw_pending?aw_address:s_axi_awaddr;
 wire [31:0] write_value=w_pending?w_value:s_axi_wdata;
 wire [3:0] write_strobe=w_pending?w_strobe:s_axi_wstrb;
 reg [31:0] bytes_cfg=0,base_cfg=0,mode_cfg=SUM_ONLY?0:1,raw_cfg=0;
 reg raw=0;reg [31:0] local_drained=0;
 reg start=0,clear=0,preload=0;
 reg busy=0,done=0,gate_loaded=0,loading=0;
 reg [31:0] error_bits=0,accepted=0,emitted=0,length=0;
 reg fused=SUM_ONLY?0:1;
 reg [9:0] gate_index=0,load_index=0;
 wire [63:0] gate_pair;
 function automatic [31:0] merged(input [31:0] old,input [31:0] value,input [3:0] strobe);
 integer k;begin for(k=0;k<4;k=k+1)merged[k*8+:8]=strobe[k]?value[k*8+:8]:old[k*8+:8];end endfunction
 always @(posedge aclk)begin
  if(!aresetn)begin aw_pending<=0;w_pending<=0;s_axi_bvalid<=0;s_axi_rvalid<=0;bytes_cfg<=0;base_cfg<=0;mode_cfg<=SUM_ONLY?0:1;raw_cfg<=0;start<=0;clear<=0;preload<=0;end
  else begin
   start<=0;clear<=0;preload<=0;
   if(aw_take)begin aw_pending<=1;aw_address<=s_axi_awaddr;end
   if(w_take)begin w_pending<=1;w_value<=s_axi_wdata;w_strobe<=s_axi_wstrb;end
   if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
   if(write_fire)begin
    aw_pending<=0;w_pending<=0;s_axi_bvalid<=1;s_axi_bresp<=0;
    case(write_address[11:2])
     2:if(write_strobe[0])begin start<=write_value[0];clear<=write_value[1];preload<=write_value[2];end
     3:if(!busy)bytes_cfg<=merged(bytes_cfg,write_value,write_strobe);else s_axi_bresp<=2;
     4:if(!busy)base_cfg<=merged(base_cfg,write_value,write_strobe);else s_axi_bresp<=2;
     5:if(!busy)mode_cfg<=merged(mode_cfg,write_value,write_strobe);else s_axi_bresp<=2;
     9:if(!busy)raw_cfg<=merged(raw_cfg,write_value,write_strobe);else s_axi_bresp<=2;
     default:s_axi_bresp<=2;
    endcase
   end
   if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
   if(s_axi_arvalid&&s_axi_arready)begin
    s_axi_rvalid<=1;s_axi_rresp<=0;
    case(s_axi_araddr[11:2])
     0:s_axi_rdata<=32'hc206f001;
     1:s_axi_rdata<={27'd0,loading,gate_loaded,(error_bits!=0),done,busy};
     3:s_axi_rdata<=bytes_cfg;4:s_axi_rdata<=base_cfg;5:s_axi_rdata<=mode_cfg;
     6:s_axi_rdata<=error_bits;7:s_axi_rdata<=accepted;8:s_axi_rdata<=emitted;
     9:s_axi_rdata<=raw_cfg;10:s_axi_rdata<=local_drained;
     default:s_axi_rdata<=0;
    endcase
   end
  end
 end
 wire [1:0] in_ready,sum_valid,sum_ready,mul_valid,mul_ready,residual_ready,out_valid,out_ready;
 wire [31:0] sum_data[0:1],mul_data[0:1],out_data[0:1];
 wire [64:0] sum_side[0:1];wire [32:0] mul_side[0:1];wire out_last[0:1];
 wire expected_last=(accepted+8==length);
 reg [1:0] pipe_reset_count=3;
 always @(posedge aclk)begin
  if(!aresetn||clear)pipe_reset_count<=3;
  else if(pipe_reset_count!=0)pipe_reset_count<=pipe_reset_count-1;
 end
 wire pipe_resetn=aresetn&&!clear&&(pipe_reset_count==0);
 wire input_valid=!BULK_ONLY&&!raw&&pipe_resetn&&busy&&!loading&&(accepted<length)&&s_axis_local_tvalid&&s_axis_peer_tvalid&&(!fused||s_axis_residual_tvalid);
 wire joined=input_valid&&(&in_ready);
 // One synchronous RAM read expression permits BRAM mapping.
 // Preserve the original NBA priority: joined overrides start if both occur.
 // The first gate is still available immediately after the start clock edge;
 // each accepted pair fetches the next gate, with no extra pipeline cycle.
 wire gate_read_enable=aresetn&&!clear&&((start&&!busy)||joined);
 wire [9:0] gate_read_address=joined?(gate_index+10'd1):base_cfg[10:1];
 generate if(!BULK_ONLY&&!SUM_ONLY)begin:gate_cache
  (* ram_style="block" *)reg [63:0] gate_memory[0:1023];
  reg [63:0] gate_pair_reg=0;
  always @(posedge aclk)begin
   if(aresetn&&!clear&&loading&&s_axis_residual_tvalid&&s_axis_residual_tready)gate_memory[load_index]<=s_axis_residual_tdata;
   if(gate_read_enable)gate_pair_reg<=gate_memory[gate_read_address];
  end
  assign gate_pair=gate_pair_reg;
 end else begin:no_gate_cache
  assign gate_pair=64'd0;
 end endgenerate
 wire raw_active=raw&&pipe_resetn&&busy&&!loading;
 wire raw_take=raw_active&&(accepted<length)&&s_axis_peer_tvalid&&m_axis_result_tready;
 wire local_take=raw_active&&s_axis_local_tvalid&&s_axis_local_tready;
 assign s_axis_local_tready=raw?(raw_active&&local_drained<length):pipe_resetn&&busy&&!loading&&(accepted<length)&&s_axis_peer_tvalid&&(!fused||s_axis_residual_tvalid)&&(&in_ready);
 assign s_axis_peer_tready=raw?(raw_active&&(accepted<length)&&m_axis_result_tready):pipe_resetn&&busy&&!loading&&(accepted<length)&&s_axis_local_tvalid&&(!fused||s_axis_residual_tvalid)&&(&in_ready);
 assign s_axis_residual_tready=!BULK_ONLY&&!SUM_ONLY&&(loading||(!raw&&pipe_resetn&&busy&&(accepted<length)&&fused&&s_axis_local_tvalid&&s_axis_peer_tvalid&&(&in_ready)));
 genvar i,b;
 generate for(i=0;i<(BULK_ONLY?0:2);i=i+1)begin:elements
  if(SUM_ONLY)begin:sum_only
   wire sum_last;
   // P0's exact first-round FP32 add; do not carry unused gate/residual
   // payload through its pipeline. The two sum lanes retain atomic flow.
   c205_fp32_token #(.OP(0),.SIDE_WIDTH(1)) sum(
    .aclk(aclk),.aresetn(pipe_resetn),.a(s_axis_local_tdata[32*i+:32]),.b(s_axis_peer_tdata[32*i+:32]),
    .side(expected_last),.s_valid(input_valid&&in_ready[1-i]),.s_ready(in_ready[i]),
    .result(sum_data[i]),.result_side(sum_last),.m_valid(sum_valid[i]),.m_ready(sum_ready[i]));
   assign sum_side[i]={sum_last,64'd0};
   assign mul_ready[i]=1'b0;assign mul_valid[i]=1'b0;assign residual_ready[i]=1'b0;
   assign out_valid[i]=1'b0;assign mul_data[i]=32'd0;assign out_data[i]=32'd0;
   assign mul_side[i]=33'd0;assign out_last[i]=1'b0;
  end else begin:fused_capable
  c205_fp32_token #(.OP(0),.SIDE_WIDTH(65)) sum(
   .aclk(aclk),.aresetn(pipe_resetn),.a(s_axis_local_tdata[32*i+:32]),.b(s_axis_peer_tdata[32*i+:32]),
   .side({expected_last,s_axis_residual_tdata[32*i+:32],gate_pair[32*i+:32]}),
   .s_valid(input_valid&&in_ready[1-i]),.s_ready(in_ready[i]),.result(sum_data[i]),.result_side(sum_side[i]),.m_valid(sum_valid[i]),.m_ready(sum_ready[i]));
  c205_fp32_token #(.OP(1),.SIDE_WIDTH(33)) mul(
   .aclk(aclk),.aresetn(pipe_resetn),.a(sum_data[i]),.b(sum_side[i][31:0]),.side(sum_side[i][64:32]),
   .s_valid(sum_valid[i]&&fused),.s_ready(mul_ready[i]),.result(mul_data[i]),.result_side(mul_side[i]),.m_valid(mul_valid[i]),.m_ready(residual_ready[i]));
  c205_fp32_token #(.OP(0),.SIDE_WIDTH(1)) residual(
   .aclk(aclk),.aresetn(pipe_resetn),.a(mul_data[i]),.b(mul_side[i][31:0]),.side(mul_side[i][32]),
   .s_valid(mul_valid[i]),.s_ready(residual_ready[i]),.result(out_data[i]),.result_side(out_last[i]),.m_valid(out_valid[i]),.m_ready(out_ready[i]));
  end
  assign sum_ready[i]=fused?mul_ready[i]:(m_axis_result_tready&&sum_valid[1-i]);
 end endgenerate
 generate if(BULK_ONLY)begin:bulk_constants
  assign in_ready=0;assign sum_valid=0;assign mul_valid=0;assign out_valid=0;
  for(b=0;b<2;b=b+1)begin:unused_fp
   assign sum_data[b]=0;assign mul_data[b]=0;assign out_data[b]=0;
   assign sum_side[b]=0;assign mul_side[b]=0;assign out_last[b]=0;
  end
 end endgenerate
 // SUM mode bypasses multiply/residual; the mode remains frozen until output drains.
 assign m_axis_result_tdata=(BULK_ONLY||raw)?s_axis_peer_tdata:fused?{out_data[1],out_data[0]}:{sum_data[1],sum_data[0]};
 assign m_axis_result_tvalid=(BULK_ONLY||raw)?(raw_active&&(accepted<length)&&s_axis_peer_tvalid):fused?(&out_valid):(&sum_valid);
 assign m_axis_result_tkeep=8'hff;
 assign m_axis_result_tlast=(BULK_ONLY||raw)?expected_last:fused?(out_last[0]&&out_last[1]):(sum_side[0][64]&&sum_side[1][64]);
 assign out_ready[0]=fused&&m_axis_result_tready&&out_valid[1];
 assign out_ready[1]=fused&&m_axis_result_tready&&out_valid[0];
 always @(posedge aclk)begin
  if(!aresetn)begin busy<=0;done<=0;gate_loaded<=0;loading<=0;error_bits<=0;accepted<=0;emitted<=0;length<=0;gate_index<=0;load_index<=0;fused<=SUM_ONLY?0:1;raw<=0;local_drained<=0;end
  else if(clear)begin busy<=0;done<=0;loading<=0;error_bits<=0;accepted<=0;emitted<=0;local_drained<=0;end
  else begin
   if(preload&&!busy)begin
    busy<=!BULK_ONLY&&!SUM_ONLY;loading<=!BULK_ONLY&&!SUM_ONLY;gate_loaded<=0;done<=0;
    error_bits<=(BULK_ONLY||SUM_ONLY)?32'h80:0;load_index<=0;accepted<=0;emitted<=0;
   end
   if(start&&!busy)begin
    done<=0;error_bits<=0;accepted<=0;emitted<=0;length<=bytes_cfg;fused<=SUM_ONLY?1'b0:mode_cfg[0];raw<=raw_cfg[0];local_drained<=0;gate_index<=base_cfg[10:1];
    // Bulk-only endpoints explicitly reject PRELOAD and non-raw START. The
    // AXI command write is acknowledged; CSR error bit7 reports unsupported.
    if((BULK_ONLY&&(preload||!raw_cfg[0]))||(SUM_ONLY&&(preload||(!raw_cfg[0]&&mode_cfg[0]))))error_bits<=32'h80;
    else if(bytes_cfg==0||bytes_cfg>8192||bytes_cfg[2:0]!=0||base_cfg[0]||mode_cfg>1||raw_cfg>1||(!raw_cfg[0]&&mode_cfg[0]&&!gate_loaded))error_bits<=1;
    else busy<=1;
   end
   if(loading&&s_axis_residual_tvalid&&s_axis_residual_tready)begin
    load_index<=load_index+1;accepted<=accepted+8;
    if(s_axis_residual_tkeep!=8'hff||s_axis_residual_tlast!=(load_index==1023))begin error_bits<=error_bits|2;busy<=0;loading<=0;end
    else if(load_index==1023)begin gate_loaded<=1;done<=1;busy<=0;loading<=0;end
   end
   if(joined)begin
    accepted<=accepted+8;gate_index<=gate_index+1;
    if(s_axis_local_tkeep!=8'hff||s_axis_peer_tkeep!=8'hff||(fused&&s_axis_residual_tkeep!=8'hff))error_bits<=error_bits|4;
    if(s_axis_local_tlast!=expected_last||s_axis_peer_tlast!=expected_last||(fused&&s_axis_residual_tlast!=expected_last))error_bits<=error_bits|8;
   end
   if(local_take)begin
    local_drained<=local_drained+8;
    if(s_axis_local_tkeep!=8'hff||s_axis_local_tlast!=(local_drained+8==length))error_bits<=error_bits|32;
   end
   if(raw_take)begin
    accepted<=accepted+8;
    if(s_axis_peer_tkeep!=8'hff||s_axis_peer_tlast!=expected_last)error_bits<=error_bits|64;
   end
   if(raw&&busy&&(emitted+(raw_take?8:0)==length)&&(local_drained+(local_take?8:0)==length))begin busy<=0;done<=1;end
   if(m_axis_result_tvalid&&m_axis_result_tready)begin
    emitted<=emitted+8;
    if(!raw&&((fused&&(out_last[0]!=out_last[1]))||(!fused&&(sum_side[0][64]!=sum_side[1][64]))))error_bits<=error_bits|16;
    if(!raw&&emitted+8==length)begin busy<=0;done<=1;end
   end
  end
 end
endmodule
