`timescale 1ns/1ps
// Drop-in replacement for the W generated 5120x32, synchronous one-read /
// one-write u RAM. Keep this generated module name and its original ports:
// no HLS controller, RAM latency, arithmetic, or writeback tag changes.
// The five natural 1024x32 banks each contain both slots of one local head.
module cosmos2b_tile_update_five_to_FixedAttentionDDRSink_s_u_RAM_2P_BRAM_1R1W #(
 parameter DataWidth=32,
 parameter AddressWidth=13,
 parameter AddressRange=5120
)(
 input wire [AddressWidth-1:0] address0,
 input wire ce0,
 output wire [DataWidth-1:0] q0,
 input wire [AddressWidth-1:0] address1,
 input wire ce1,
 input wire [DataWidth-1:0] d1,
 input wire we1,
 input wire reset,
 input wire clk
);
 // Original address = context*512 + half_beat, context=slot*5+head.
 // Remapping is bijective for all5120 legal words; read and late write
 // addresses are decoded independently, so existing HLS tags stay intact.
 wire [3:0] read_context=address0[12:9];
 wire [3:0] write_context=address1[12:9];
 wire read_slot=read_context>=4'd5;
 wire write_slot=write_context>=4'd5;
 wire [3:0] read_head_full=read_slot?read_context-4'd5:read_context;
 wire [3:0] write_head_full=write_slot?write_context-4'd5:write_context;
 wire [2:0] read_head=read_head_full[2:0];
 wire [2:0] write_head=write_head_full[2:0];
 wire read_legal=read_context<4'd10;
 wire write_legal=write_context<4'd10;
 wire [9:0] read_word={read_slot,address0[8:0]};
 wire [9:0] write_word={write_slot,address1[8:0]};
 wire [31:0] bank_q[0:4];
 reg [2:0] read_head_d;
 reg [31:0] selected_q;

 // The primitive's unregistered output is already a one-clock synchronous
 // read. Only its bank tag is registered; registering selected_q here would
 // add a second clock and break the generated HLS schedule.
 always @(posedge clk)if(ce0)read_head_d<=read_head;
 always @*begin
  case(read_head_d)
   3'd0:selected_q=bank_q[0];
   3'd1:selected_q=bank_q[1];
   3'd2:selected_q=bank_q[2];
   3'd3:selected_q=bank_q[3];
   3'd4:selected_q=bank_q[4];
   default:selected_q=32'bx;
  endcase
 end
 assign q0=selected_q;

 genvar head;
 generate for(head=0;head<5;head=head+1)begin:head_banks
  localparam [2:0] HEAD=head;
  wire read_enable=ce0&&read_legal&&(read_head==HEAD);
  wire write_enable=ce1&&we1&&write_legal&&(write_head==HEAD);
  // 36-bit x1024 native shape; four parity bits are unused. One port reads,
  // the other writes. No output register or memory reset is inserted.
  RAMB36E2 #(
   .CLOCK_DOMAINS("COMMON"),.DOA_REG(0),.DOB_REG(0),
   .READ_WIDTH_A(36),.READ_WIDTH_B(0),
   .WRITE_WIDTH_A(0),.WRITE_WIDTH_B(36),
   .WRITE_MODE_A("READ_FIRST"),.WRITE_MODE_B("READ_FIRST"),
   .CASCADE_ORDER_A("NONE"),.CASCADE_ORDER_B("NONE"),
   .ENADDRENA("FALSE"),.ENADDRENB("FALSE"),
   .EN_ECC_PIPE("FALSE"),.EN_ECC_READ("FALSE"),.EN_ECC_WRITE("FALSE"),
   .SIM_COLLISION_CHECK("ALL")
  ) storage(
      .CLKARDCLK(clk),.CLKBWRCLK(clk),
      .ADDRARDADDR({read_word,5'b00000}),.ADDRBWRADDR({write_word,5'b00000}),
      .ENARDEN(read_enable),.ENBWREN(write_enable),
      .WEA(4'b0000),.WEBWE({4'b0000,{4{write_enable}}}),
      .DINADIN(32'b0),.DINBDIN(d1),
      .DINPADINP(4'b0000),.DINPBDINP(4'b0000),
      .DOUTADOUT(bank_q[head]),.DOUTBDOUT(),
      .DOUTPADOUTP(),.DOUTPBDOUTP(),
      .REGCEAREGCE(1'b1),.REGCEB(1'b1),
      .RSTRAMARSTRAM(1'b0),.RSTRAMB(1'b0),
      .RSTREGARSTREG(1'b0),.RSTREGB(1'b0),
      .ADDRENA(1'b1),.ADDRENB(1'b1),
      .SLEEP(1'b0),.ECCPIPECE(1'b0),
      .INJECTSBITERR(1'b0),.INJECTDBITERR(1'b0),
      .CASDINA(32'b0),.CASDINB(32'b0),
      .CASDINPA(4'b0),.CASDINPB(4'b0),
      .CASDIMUXA(1'b0),.CASDIMUXB(1'b0),
      .CASDOMUXA(1'b0),.CASDOMUXB(1'b0),
      .CASDOMUXEN_A(1'b0),.CASDOMUXEN_B(1'b0),
      .CASOREGIMUXA(1'b0),.CASOREGIMUXB(1'b0),
      .CASOREGIMUXEN_A(1'b0),.CASOREGIMUXEN_B(1'b0),
      .CASINDBITERR(1'b0),.CASINSBITERR(1'b0),
      .CASDOUTA(),.CASDOUTB(),.CASDOUTPA(),.CASDOUTPB(),
      .CASOUTDBITERR(),.CASOUTSBITERR(),
      .DBITERR(),.SBITERR(),.ECCPARITY(),.RDADDRECC()
  );
 end endgenerate

 // reset intentionally does not clear memory or output: the original HLS
 // RAM ignores reset. first/last and finite-loop drain remain HLS-owned.
 // synthesis translate_off
 initial begin
  if(DataWidth!=32||AddressWidth!=13||AddressRange!=5120)
   $fatal(1,"ATTENTION_U_STORAGE_UNSUPPORTED_GEOMETRY");
 end
 always @(posedge clk)begin
  if(ce0&&!read_legal)$fatal(1,"ATTENTION_U_STORAGE_READ_RANGE");
  if(ce1&&we1&&!write_legal)$fatal(1,"ATTENTION_U_STORAGE_WRITE_RANGE");
  if(ce0&&ce1&&we1&&(address0==address1))
   $fatal(1,"ATTENTION_U_STORAGE_READ_WRITE_COLLISION");
 end
 // synthesis translate_on
endmodule

