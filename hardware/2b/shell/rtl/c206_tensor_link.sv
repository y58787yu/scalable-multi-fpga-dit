// C206: independent per-port one-chunk credit, full collective identity.
`timescale 1ns/1ps
module c206_tensor_link #(parameter LANE_ID=0)(
 (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF s_axi:s_axis_mm2s:m_axis_s2mm:m_axis_net:s_axis_net, ASSOCIATED_RESET aresetn" *)
 input wire aclk,
 (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *)
 input wire aresetn,
 input wire [31:0] user_tag,input wire [7:0] phy_status,mac_errors,
 output wire rx_polarity,output wire [7:0] debug_status,output wire [7:0] node_id,
 input wire [31:0] router_dropped,router_forwarded,router_delivered,
 input wire [11:0] s_axi_awaddr,input wire [2:0] s_axi_awprot,input wire s_axi_awvalid,output wire s_axi_awready,
 input wire [31:0] s_axi_wdata,input wire [3:0] s_axi_wstrb,input wire s_axi_wvalid,output wire s_axi_wready,
 output reg [1:0] s_axi_bresp=0,output reg s_axi_bvalid=0,input wire s_axi_bready,
 input wire [11:0] s_axi_araddr,input wire [2:0] s_axi_arprot,input wire s_axi_arvalid,output wire s_axi_arready,
 output reg [31:0] s_axi_rdata=0,output reg [1:0] s_axi_rresp=0,output reg s_axi_rvalid=0,input wire s_axi_rready,
 input wire [63:0] s_axis_mm2s_tdata,input wire [7:0] s_axis_mm2s_tkeep,input wire [7:0] s_axis_mm2s_tstrb,input wire s_axis_mm2s_tvalid,
 output wire s_axis_mm2s_tready,input wire s_axis_mm2s_tlast,
 output wire [63:0] m_axis_s2mm_tdata,output wire [7:0] m_axis_s2mm_tkeep,output wire [7:0] m_axis_s2mm_tstrb,output wire m_axis_s2mm_tvalid,
 input wire m_axis_s2mm_tready,output wire m_axis_s2mm_tlast,
 output reg [63:0] m_axis_net_tdata,output wire [7:0] m_axis_net_tkeep,output wire m_axis_net_tvalid,
 input wire m_axis_net_tready,output wire m_axis_net_tlast,output wire m_axis_net_tuser,
 input wire [63:0] s_axis_net_tdata,input wire [7:0] s_axis_net_tkeep,input wire s_axis_net_tvalid,
 output wire s_axis_net_tready,input wire s_axis_net_tlast,s_axis_net_tuser
);
 localparam HEADER_WORDS=10,DATA_WORDS=176;
 wire [31:0] context_csr_data;
 wire [63:0] context_tx_header;
 wire [7:0] context_tx_lane;
 wire context_header_bad,context_ready_matches_rx,context_frame_matches_rx;
 wire context_pending_matches_frame,context_same_namespace,context_frame_older;
 reg credit=0,credit_issued=0;
 // A peer can finish polling the previous chunk first. Preserve its next READY
 // until RX_ARM matches the complete tuple; never grant speculative data credit.
 reg ready_pending=0;
 reg [7:0] ready_source=0,ready_destination=0,frame_source=0,frame_eth_source=0,rx_peer=0;
 reg [31:0] ready_id=0,ready_meta=0;
 wire [31:0] cfg,tx_id_cfg,tx_len_cfg,tx_meta_cfg,tx_count_cfg;
 wire [31:0] rx_id_cfg,rx_len_cfg,rx_meta_cfg,rx_count_cfg;
 wire [7:0] local_node,peer_node;
 reg pop_control=0;
 reg [7:0] control_kinds[0:15];
 reg [31:0] control_ids[0:15],control_metas[0:15];
 reg [3:0] control_wr=0,control_rd=0;
 reg [4:0] control_count=0;
 assign rx_polarity=cfg[1];
 assign node_id=local_node;
 (* ASYNC_REG="TRUE" *)reg [7:0] phy_sync1=0,phy_sync2=0;
 always @(posedge aclk)begin phy_sync1<=phy_status;phy_sync2<=phy_sync1;end
 wire link_up=(phy_sync2[6:0]==7'h53);
 assign debug_status=phy_sync2;
 assign s_axi_arready=!s_axi_rvalid;
 reg start_data=0,start_control=0,arm_rx=0,clear=0;
 wire write_valid,write_ready;
 wire [11:0] write_address;
 wire [31:0] write_value;
 wire [3:0] write_strobe;
 // AW and W remain independent. Only registered payload/strobes reach CSR
 // consumers; a stalled B response cannot overwrite an accepted write.
 // Commands execute one edge after commit, so hold the following commit until
 // that edge has captured its complete configuration snapshot.
 assign write_ready=(!s_axi_bvalid||s_axi_bready)&&
                    !(start_data||start_control||arm_rx||clear||pop_control);
 wire write_fire=write_valid&&write_ready;
 (* keep_hierarchy="yes" *) c209_link_write_capture write_capture(
  .aclk(aclk),.aresetn(aresetn),
  .awaddr(s_axi_awaddr),.awvalid(s_axi_awvalid),.awready(s_axi_awready),
  .wdata(s_axi_wdata),.wstrb(s_axi_wstrb),.wvalid(s_axi_wvalid),.wready(s_axi_wready),
  .command_valid(write_valid),.command_ready(write_ready),
  .command_address(write_address),.command_data(write_value),.command_strobe(write_strobe)
 );
 reg [63:0] ticks=0,tx_begin=0,rx_first=0,rx_end=0,ctrl_tick=0;
 reg [31:0] tx_id=0,tx_meta=0,tx_count=0,tx_length=0,tx_seq=0,tx_words_left=0,tx_frames=0;
 reg [15:0] tx_index=0,frame_data_words=0;
 reg [1:0] tx_state=0;
 reg tx_is_control=0,tx_done=0,tx_error=0;
 wire [15:0] tx_frame_words=tx_is_control?HEADER_WORDS+2:HEADER_WORDS+frame_data_words;
 assign m_axis_net_tvalid=(tx_state==2)?s_axis_mm2s_tvalid:(tx_state!=0);
 assign m_axis_net_tkeep=8'hff;
 assign m_axis_net_tuser=1'b0;
 assign m_axis_net_tlast=(tx_state==2 && tx_index==frame_data_words-1)||(tx_state==3 && tx_index==1);
 assign s_axis_mm2s_tready=tx_state==2&&m_axis_net_tready;
 always @*begin
  m_axis_net_tdata=64'd0;
  if(tx_state==1)case(tx_index)
   0:m_axis_net_tdata=64'h0002ffffffffffff;
   1:m_axis_net_tdata={32'h3530b588,local_node,24'd0};
   2:m_axis_net_tdata={local_node,tx_meta[7:0],tx_frame_words,tx_seq};
   3:m_axis_net_tdata={tx_length,tx_id};
   4:m_axis_net_tdata={peer_node,context_tx_lane,tx_count[15:0],tx_meta};
   5:m_axis_net_tdata=tx_begin;
   6,7,8,9:m_axis_net_tdata=context_tx_header;
  endcase
  else if(tx_state==2)m_axis_net_tdata=s_axis_mm2s_tdata;
 end
 // Payload valid is gated by DMA's valid, never by its ready.
 wire net_valid=m_axis_net_tvalid;
 reg [31:0] rx_id=0,rx_expected_length=0,rx_meta=0,rx_count=0;
 reg rx_armed=0,rx_done=0;
 reg [31:0] rx_bytes=0,rx_frames=0,rx_errors=0,mac_faults=0;
 reg [15:0] rx_index=0,rx_frame_words=0;
 reg [7:0] rx_kind=8'hff;
 reg [31:0] frame_id=0,frame_len=0,frame_seq=0,expected_seq=0,frame_meta=0;
 reg [15:0] frame_count=0;
 reg frame_bad=0;
 reg [31:0] ctrl_kind=0,ctrl_id=0,ctrl_events=0,ctrl_meta=0;
 // Two-beat endpoint-owned elastic store. Reverse READY depends only on
 // registered occupancy, never parser/DMA READY or a same-cycle pop.
 // Every parser consumer below sees this one buffered head and rx_take.
 reg [73:0] rx_queue[0:1];
 reg rx_queue_read=0,rx_queue_write=0;
 reg [1:0] rx_queue_count=0;
 wire [73:0] rx_head=rx_queue[rx_queue_read];
 wire [63:0] rx_head_data=rx_head[63:0];
 wire [7:0] rx_head_keep=rx_head[71:64];
 wire rx_head_last=rx_head[72],rx_head_user=rx_head[73];
 wire rx_head_valid=aresetn&&!clear&&(rx_queue_count!=0);
 assign s_axis_net_tready=aresetn&&!clear&&(rx_queue_count<2);
 wire rx_enqueue=s_axis_net_tvalid&&s_axis_net_tready;
 wire rx_take;
 // CSR0xe0: [1:0] occupancy, [2] partial parser frame, [31:3] reserved0.
 // Reset/CLEAR abort queued work together with parser state. Payload FFs are
 // deliberately not reset; empty entries are never consumed or committed.
 always @(posedge aclk)begin:rx_elastic_store
  if(!aresetn||clear)begin
   rx_queue_read<=0;rx_queue_write<=0;rx_queue_count<=0;
  end else begin
   if(rx_enqueue)begin
    rx_queue[rx_queue_write]<={s_axis_net_tuser,s_axis_net_tlast,s_axis_net_tkeep,s_axis_net_tdata};
    rx_queue_write<=!rx_queue_write;
   end
   if(rx_take)rx_queue_read<=!rx_queue_read;
   case({rx_enqueue,rx_take})
    2'b10:rx_queue_count<=rx_queue_count+1'b1;
    2'b01:rx_queue_count<=rx_queue_count-1'b1;
    default:rx_queue_count<=rx_queue_count;
   endcase
  end
 end
 wire payload=(rx_index>=HEADER_WORDS)&&(rx_kind<2);
 wire good_payload=payload&&rx_armed&&!frame_bad&&rx_errors==0&&rx_head_keep==8'hff&&rx_bytes+8<=rx_expected_length;
 assign m_axis_s2mm_tdata=rx_head_data;
 assign m_axis_s2mm_tkeep=8'hff;
 assign m_axis_s2mm_tstrb=8'hff;
 assign m_axis_s2mm_tvalid=rx_head_valid&&good_payload;
 assign m_axis_s2mm_tlast=rx_head_last&&rx_bytes+8==rx_expected_length;
 wire rx_parser_ready=good_payload?m_axis_s2mm_tready:1'b1;
 assign rx_take=rx_head_valid&&rx_parser_ready;
 wire tx_take=net_valid&&m_axis_net_tready;
 wire control_commit=rx_take&&rx_head_last&&rx_kind>=2&&rx_kind<=5&&!frame_bad&&
                     !rx_head_user&&rx_head_keep==8'hff&&rx_index+1==rx_frame_words;
 wire control_pop=pop_control&&control_count!=0;
 wire nonready_commit=control_commit&&(rx_kind!=2);
 wire ready_matches_rx=ready_pending&&rx_armed&&!credit_issued&&ready_source==rx_peer&&ready_destination==local_node&&ready_id==rx_id&&ready_meta[31:8]==rx_meta[31:8]&&context_ready_matches_rx;
 wire ready_push=ready_matches_rx&&!nonready_commit&&(control_count<16||control_pop);
 wire control_push=(nonready_commit||ready_push)&&(control_count<16||control_pop);
 wire frame_matches_rx=frame_id==rx_id&&frame_meta[31:8]==rx_meta[31:8]&&context_frame_matches_rx;
 // Monotonic collective order: epoch, step, layer, operator, global row chunk.
 // Branch/group/lane are static logical identity and checked at the header.
 // Compare with the last complete ARM snapshot, never with partially updated
 // AXI configuration registers during software's transition to a new operator.
 wire same_namespace=context_same_namespace&&frame_source==rx_peer;
 wire frame_older=same_namespace&&context_frame_older;
 wire ready_commit=control_commit&&rx_kind==2&&!frame_older&&!(frame_matches_rx&&(credit_issued||ready_push));
 wire pending_same_frame=ready_pending&&ready_id==frame_id&&ready_meta==frame_meta&&ready_source==frame_source&&ready_destination==local_node&&context_pending_matches_frame;
 wire ready_conflict=ready_commit&&ready_pending&&!ready_push&&!pending_same_frame;
 // The complete 224-bit identities and their group words remain local. Main
 // link state exchanges word-sized header/CSR data and match flags only.
 (* keep_hierarchy="yes" *) c209_link_context_owner #(.LANE_ID(LANE_ID)) contexts(
  .aclk(aclk),.aresetn(aresetn),.clear(clear),
  .cfg_write(write_fire&&tx_state==0&&!rx_armed),.cfg_address(write_address),
  .cfg_data(write_value),.cfg_strobe(write_strobe),
  .csr_address(s_axi_araddr),.csr_data(context_csr_data),
  .tx_capture((start_data||start_control)&&tx_state==0),.rx_capture(arm_rx),
  .ready_capture(ready_commit&&!ready_conflict),
  .tx_index(tx_index),.tx_header(context_tx_header),.tx_lane(context_tx_lane),
  .frame_take(rx_take),.frame_index(rx_index),.frame_kind(rx_kind),.frame_data(rx_head_data),
  .header_bad(context_header_bad),.ready_matches_rx(context_ready_matches_rx),
  .frame_matches_rx(context_frame_matches_rx),.pending_matches_frame(context_pending_matches_frame),
  .same_namespace(context_same_namespace),.frame_older(context_frame_older)
 );
 // Keep byte preservation on D and share the accepted word write across all
 // bits. This leaves command timing, AW/W capture and B responses unchanged.
 wire [31:0] link_cfg_words[0:8];
 assign cfg=link_cfg_words[0];
 assign tx_id_cfg=link_cfg_words[1];assign tx_len_cfg=link_cfg_words[2];
 assign tx_meta_cfg=link_cfg_words[3];assign tx_count_cfg=link_cfg_words[4];
 assign rx_id_cfg=link_cfg_words[5];assign rx_len_cfg=link_cfg_words[6];
 assign rx_meta_cfg=link_cfg_words[7];assign rx_count_cfg=link_cfg_words[8];
 genvar cw;
 generate for(cw=0;cw<9;cw=cw+1)begin:link_cfg
  localparam integer ADDRESS=(cw==0)?2:cw+3;
  c209_link_cfg_word cfg_word(.aclk(aclk),.aresetn(aresetn),
   .word_write(write_fire&&write_address[11:2]==ADDRESS),.write_data(write_value),
   .write_strobe(write_strobe),
   .reset_value(cw==0&&user_tag!=32'hc2090b00 ? 32'd3 : 32'd0),.value(link_cfg_words[cw]));
 end endgenerate
 c209_link_cfg_word #(.WIDTH(8),.INIT(8'd0)) local_node_word(
  .aclk(aclk),.aresetn(aresetn),.word_write(write_fire&&write_address[11:2]==36),
  .write_data(write_value[7:0]),.write_strobe(write_strobe),
  .reset_value(user_tag==32'hc2090b00 ? 8'd1 : 8'd2),.value(local_node));
 c209_link_cfg_word #(.WIDTH(8),.INIT(8'd1)) peer_node_word(
  .aclk(aclk),.aresetn(aresetn),.word_write(write_fire&&write_address[11:2]==37),
  .write_data(write_value[7:0]),.write_strobe(write_strobe),
  .reset_value(user_tag==32'hc2090b00 ? 8'd2 : 8'd1),.value(peer_node));
 function automatic [31:0] read_register(input [11:0] address);
 begin case(address[11:2])
  0:read_register=32'hc206e001;
  1:read_register={25'd0,credit,link_up,rx_armed,(rx_errors!=0||tx_error),rx_done,tx_done,(tx_state!=0)};
  2:read_register=cfg;
  3:read_register={24'd0,phy_sync2};
  4:read_register=tx_id_cfg;5:read_register=tx_len_cfg;6:read_register=tx_meta_cfg;7:read_register=tx_count_cfg;
  8:read_register=rx_id_cfg;9:read_register=rx_len_cfg;10:read_register=rx_meta_cfg;11:read_register=rx_count_cfg;
  12:read_register=control_count?{24'd0,control_kinds[control_rd]}:0;
  13:read_register=control_count?control_ids[control_rd]:0;
  14:read_register=ctrl_events;
  15:read_register=control_count?control_metas[control_rd]:0;
  16:read_register=rx_bytes;17:read_register=rx_frames;18:read_register=tx_frames;19:read_register=rx_errors;
  20:read_register=mac_faults;21:read_register=rx_first[31:0];22:read_register=rx_first[63:32];
  23:read_register=rx_end[31:0];24:read_register=ticks[31:0];25:read_register=ticks[63:32];
  26:read_register=tx_begin[31:0];27:read_register=tx_begin[63:32];28:read_register=ctrl_tick[31:0];29:read_register=ctrl_tick[63:32];
  30:read_register=rx_end[63:32];31:read_register=user_tag;
  32:read_register={27'd0,control_count};
  36:read_register={24'd0,local_node};37:read_register={24'd0,peer_node};
  39,40,41,42,43,44,45,46:read_register=context_csr_data;
  48:read_register={31'd0,credit};
  49:read_register={31'd0,ready_pending};50:read_register=ready_id;
  51,52:read_register=context_csr_data;
  53:read_register=router_dropped;54:read_register=router_forwarded;55:read_register=router_delivered;
  56:read_register={29'd0,(rx_index!=0),rx_queue_count};
  default:read_register=0;
 endcase end endfunction
 always @(posedge aclk)begin
  if(!aresetn)begin
   s_axi_bvalid<=0;s_axi_rvalid<=0;s_axi_rdata<=0;
   start_data<=0;start_control<=0;arm_rx<=0;clear<=0;pop_control<=0;
  end else begin
   start_data<=0;start_control<=0;arm_rx<=0;clear<=0;pop_control<=0;
   if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
   if(write_fire)begin
    s_axi_bvalid<=1;s_axi_bresp<=0;
    case(write_address[11:2])
     1:begin start_data<=write_value[0];clear<=write_value[1];arm_rx<=write_value[2];start_control<=write_value[4];end
     33:pop_control<=write_value[0];
     39,40,41,42,43,44,45,46:if(tx_state!=0||rx_armed)s_axi_bresp<=2;
    endcase
   end
   if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
   if(s_axi_arvalid&&s_axi_arready)begin s_axi_rvalid<=1;s_axi_rdata<=read_register(s_axi_araddr);s_axi_rresp<=0;end
  end
 end
 always @(posedge aclk)begin
  if(!aresetn||clear)begin
   if(!aresetn)rx_peer<=0;
   ready_pending<=0;credit<=0;credit_issued<=0;ticks<=0;tx_state<=0;tx_done<=0;tx_error<=0;tx_index<=0;tx_frames<=0;tx_begin<=0;
   rx_armed<=0;rx_done<=0;rx_bytes<=0;rx_frames<=0;rx_errors<=0;mac_faults<=0;expected_seq<=0;
   rx_index<=0;rx_kind<=8'hff;frame_bad<=0;ctrl_events<=0;ctrl_kind<=0;ctrl_id<=0;ctrl_tick<=0;
   rx_first<=0;rx_end<=0;
   control_wr<=0;control_rd<=0;control_count<=0;
  end else begin
   ticks<=ticks+1;
   if(ready_push)begin credit<=1;credit_issued<=1;ready_pending<=0;end
   if(ready_commit)begin
    if(!ready_conflict)begin
     ready_pending<=1;ready_id<=frame_id;ready_meta<=frame_meta;
     ready_source<=frame_source;ready_destination<=local_node;
    end
   end
   if(control_push)begin
    control_kinds[control_wr]<=ready_push?8'd2:rx_kind;
    control_ids[control_wr]<=ready_push?ready_id:frame_id;control_metas[control_wr]<=ready_push?ready_meta:frame_meta;
    control_wr<=control_wr+1;
   end
   if(control_pop)control_rd<=control_rd+1;
   case({control_push,control_pop})
    2'b10:control_count<=control_count+1;
    2'b01:control_count<=control_count-1;
    default:control_count<=control_count;
   endcase
   if(nonready_commit&&!control_push)begin rx_errors<=rx_errors|32'h20000;rx_armed<=0;end
   if(|mac_errors)begin mac_faults<=mac_faults+1;if(rx_armed)rx_errors<=rx_errors|32'h10000;end
   if((start_data||start_control)&&tx_state==0)begin
    tx_id<=tx_id_cfg;tx_meta<=tx_meta_cfg;tx_count<=tx_count_cfg;tx_length<=start_control?0:tx_len_cfg;
    tx_is_control<=start_control;tx_words_left<=tx_len_cfg>>3;tx_seq<=0;tx_index<=0;
    frame_data_words<=((tx_len_cfg>>3)>DATA_WORDS)?DATA_WORDS:(tx_len_cfg>>3);
    tx_done<=0;tx_error<=0;tx_frames<=0;tx_begin<=ticks;
    if(start_control && (tx_meta_cfg[7:0]!=2||rx_armed))tx_state<=1;
    else if(!start_control && credit && tx_len_cfg!=0 && tx_len_cfg<=8192 && tx_len_cfg[2:0]==0)begin tx_state<=1;credit<=0;end
    else tx_error<=1;
   end
   if(tx_take)begin
    if(tx_state==1)begin
     if(tx_index==HEADER_WORDS-1)begin tx_index<=0;tx_state<=tx_is_control?3:2;end
     else tx_index<=tx_index+1;
    end else if(tx_state==3)begin
     if(tx_index==1)begin tx_state<=0;tx_done<=1;tx_frames<=tx_frames+1;end
     else tx_index<=tx_index+1;
    end else if(tx_state==2)begin
     tx_words_left<=tx_words_left-1;
     if(s_axis_mm2s_tkeep!=8'hff || s_axis_mm2s_tlast!=(tx_words_left==1))tx_error<=1;
     if(tx_index==frame_data_words-1)begin
      tx_frames<=tx_frames+1;tx_index<=0;tx_seq<=tx_seq+1;
      if(tx_words_left==1)begin tx_state<=0;tx_done<=1;end
      else begin tx_state<=1;frame_data_words<=((tx_words_left-1)>DATA_WORDS)?DATA_WORDS:(tx_words_left-1);end
     end else tx_index<=tx_index+1;
    end
   end
   if(arm_rx)begin
    credit<=0;credit_issued<=0;
    rx_peer<=peer_node;
    rx_id<=rx_id_cfg;rx_expected_length<=rx_len_cfg;rx_meta<=rx_meta_cfg;rx_count<=rx_count_cfg;
    rx_armed<=(rx_len_cfg!=0&&rx_len_cfg<=8192&&rx_len_cfg[2:0]==0);rx_done<=0;rx_bytes<=0;rx_frames<=0;rx_errors<=(rx_len_cfg!=0&&rx_len_cfg<=8192&&rx_len_cfg[2:0]==0)?0:32'h40000;expected_seq<=0;rx_first<=0;rx_end<=0;
   end
   if(rx_take)begin
    if(rx_index==0)begin frame_bad<=rx_head_data!=64'h0002ffffffffffff;rx_kind<=8'hff;end
    if(rx_index==1)begin
     frame_eth_source<=rx_head_data[31:24];
     if(rx_head_data[63:32]!=32'h3530b588)frame_bad<=1;
    end
    if(rx_index==2)begin
     frame_source<=rx_head_data[63:56];
     frame_seq<=rx_head_data[31:0];rx_frame_words<=rx_head_data[47:32];rx_kind<=rx_head_data[55:48];
     if(rx_head_data[63:56]!=frame_eth_source||rx_head_data[63:56]<1||rx_head_data[63:56]>4||rx_head_data[47:32]<HEADER_WORDS+1)frame_bad<=1;
     // READY may arrive before software switches TP/bulk context. Its source
     // is checked against the immutable ARM peer snapshot before use
     // before credit is issued; data and other controls remain strict.
     if(rx_head_data[55:48]!=2&&rx_head_data[63:56]!=peer_node)frame_bad<=1;
    end
    if(rx_index==3)begin
     frame_id<=rx_head_data[31:0];frame_len<=rx_head_data[63:32];
     if(rx_kind<2 && (!rx_armed || rx_head_data[31:0]!=rx_id || rx_head_data[63:32]!=rx_expected_length || frame_seq!=expected_seq))frame_bad<=1;
     if(rx_kind==2 && (rx_head_data[63:32]!=0||rx_frame_words!=HEADER_WORDS+2))frame_bad<=1;
    end
    if(rx_index==4)begin
     frame_meta<=rx_head_data[31:0];frame_count<=rx_head_data[47:32];
     if(rx_head_data[63:56]!=local_node || rx_head_data[55:48]!=LANE_ID)frame_bad<=1;
     if(rx_kind<2 && (rx_head_data[31:0]!=rx_meta || rx_head_data[47:32]!=rx_count[15:0]))frame_bad<=1;
     if(rx_kind>=2&&rx_head_data[7:0]!=rx_kind)frame_bad<=1;
    end
    // Data must match the immutable arm snapshot before any payload is exposed.
    // READY may arrive ahead of software; its full tuple is buffered, then matched
    // at RX_ARM. Other controls require the currently programmed context.
    if(context_header_bad)frame_bad<=1;
    if(good_payload)begin
     if(rx_bytes==0)rx_first<=ticks;
     rx_bytes<=rx_bytes+8;
    end
    if(rx_head_keep!=8'hff || rx_head_user)frame_bad<=1;
    if(rx_head_last)begin
     if(rx_kind<2)begin
      if(frame_bad||rx_head_user||rx_head_keep!=8'hff||rx_index+1!=rx_frame_words||!good_payload)begin rx_errors<=rx_errors|1;rx_armed<=0;end
      else begin
       rx_frames<=rx_frames+1;expected_seq<=expected_seq+1;
       if(rx_bytes+8==rx_expected_length)begin rx_done<=1;rx_armed<=0;rx_end<=ticks;end
      end
     end else if(rx_kind>=2&&rx_kind<=5&&!frame_bad&&rx_index+1==rx_frame_words)begin
      ctrl_kind<=rx_kind;ctrl_id<=frame_id;ctrl_meta<=frame_meta;ctrl_events<=ctrl_events+1;ctrl_tick<=ticks;
     end
     rx_index<=0;frame_bad<=0;
    end else rx_index<=rx_index+1;
   end
   // A malformed competing READY wins over a simultaneous ARM's error clear.
   // It cannot silently overwrite the retained one-ahead tuple.
   if(ready_conflict)begin rx_errors<=rx_errors|32'h80000;rx_armed<=0;end
  end
 end
endmodule
