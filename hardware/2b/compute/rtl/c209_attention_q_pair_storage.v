`timescale 1ns/1ps
// The W sequence has two logical 1280x256 Q RAMs. It reads exactly one slot
// in state11 and writes exactly one slot in state5. Those one-hot enables
// allow one physical 1R1W store without arbitration or discarded requests.
// Each of five local heads owns512 words: slot0[0:255],slot1[256:511].
// Four native64-data-bit SDP fragments/head force the72x512 BRAM shape.
module c209_attention_q_pair_storage(
 input wire [10:0] q0_address0,input wire q0_ce0,
 output wire [255:0] q0_q0,
 input wire [10:0] q0_address1,input wire q0_ce1,
 input wire [255:0] q0_d1,input wire q0_we1,
 input wire [10:0] q1_address0,input wire q1_ce0,
 output wire [255:0] q1_q0,
 input wire [10:0] q1_address1,input wire q1_ce1,
 input wire [255:0] q1_d1,input wire q1_we1,
 input wire reset,input wire clk
);
 wire read_enable=q0_ce0||q1_ce0;
 wire read_slot=q1_ce0;
 // One-hot selection, not a priority arbiter. Both enables simultaneously
 // are forbidden by the proved sequence FSM contract and asserted below.
 wire [10:0] read_address=({11{q0_ce0}}&q0_address0)|
                          ({11{q1_ce0}}&q1_address0);
 wire write0=q0_ce1&&q0_we1;
 wire write1=q1_ce1&&q1_we1;
 wire write_enable=write0||write1;
 wire write_slot=write1;
 wire [10:0] write_address=({11{write0}}&q0_address1)|
                           ({11{write1}}&q1_address1);
 wire [255:0] write_data=({256{write0}}&q0_d1)|({256{write1}}&q1_d1);
 wire [2:0] read_head=read_address[10:8];
 wire [2:0] write_head=write_address[10:8];
 wire [8:0] read_word={read_slot,read_address[7:0]};
 wire [8:0] write_word={write_slot,write_address[7:0]};
 wire [255:0] bank_q[0:4];
 reg [2:0] read_head_d;
 reg active_slot;
 reg [255:0] inactive_q;
 reg [255:0] active_q;

 always @*begin
  case(read_head_d)
   3'd0:active_q=bank_q[0];
   3'd1:active_q=bank_q[1];
   3'd2:active_q=bank_q[2];
   3'd3:active_q=bank_q[3];
   3'd4:active_q=bank_q[4];
   default:active_q=256'bx;
  endcase
 end
 // At a slot transition this samples the OLD primitive output before the
 // same clock reads the NEW slot. The new slot sees the primitive directly;
 // the inactive slot sees its saved old value. No extra read clock is added.
 // Repeated reads of one slot do not toggle/update the inactive hold.
 always @(posedge clk)if(read_enable)begin
  if(read_slot!=active_slot)inactive_q<=active_q;
  active_slot<=read_slot;
  read_head_d<=read_head;
 end
 assign q0_q0=active_slot?inactive_q:active_q;
 assign q1_q0=active_slot?active_q:inactive_q;

 genvar head,fragment;
 generate for(head=0;head<5;head=head+1)begin:head_banks
  localparam [2:0] HEAD=head;
  wire read_bank=read_enable&&(read_head==HEAD);
  wire write_bank=write_enable&&(write_head==HEAD);
  for(fragment=0;fragment<4;fragment=fragment+1)begin:word_fragments
   // RAMB36E2 SDP uses A for the72-bit read and B for the72-bit write.
   // Its data bits are {DINBDIN,DINADIN}/{DOUTBDOUT,DOUTADOUT};
   // the eight parity bits are unused. Native DOA_REG=0 is a1-clock read.
   RAMB36E2 #(
    .CLOCK_DOMAINS("COMMON"),.DOA_REG(0),.DOB_REG(0),
    .READ_WIDTH_A(72),.READ_WIDTH_B(0),
    .WRITE_WIDTH_A(0),.WRITE_WIDTH_B(72),
    .WRITE_MODE_A("READ_FIRST"),.WRITE_MODE_B("READ_FIRST"),
    .CASCADE_ORDER_A("NONE"),.CASCADE_ORDER_B("NONE"),
    .ENADDRENA("FALSE"),.ENADDRENB("FALSE"),
    .EN_ECC_PIPE("FALSE"),.EN_ECC_READ("FALSE"),.EN_ECC_WRITE("FALSE"),
    .SIM_COLLISION_CHECK("ALL")
   ) storage(
      .CLKARDCLK(clk),.CLKBWRCLK(clk),
      .ADDRARDADDR({read_word,6'b000000}),.ADDRBWRADDR({write_word,6'b000000}),
      .ENARDEN(read_bank),.ENBWREN(write_bank),
      .WEA(4'b0000),.WEBWE({8{write_bank}}),
      .DINADIN(write_data[fragment*64+:32]),.DINBDIN(write_data[fragment*64+32+:32]),
      .DINPADINP(4'b0000),.DINPBDINP(4'b0000),
      .DOUTADOUT(bank_q[head][fragment*64+:32]),.DOUTBDOUT(bank_q[head][fragment*64+32+:32]),
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
  end
 end endgenerate

 // reset is intentionally unused, matching the old generated RAMs. Every
 // consumed Q word is written by the unchanged loader before any exchange.
 // synthesis translate_off
 always @(posedge clk)begin
  if(q0_ce0&&q1_ce0)$fatal(1,"ATTENTION_Q_PAIR_SIMULTANEOUS_READS");
  if(write0&&write1)$fatal(1,"ATTENTION_Q_PAIR_SIMULTANEOUS_WRITES");
  if(read_enable&&read_head>4)$fatal(1,"ATTENTION_Q_PAIR_READ_RANGE");
  if(write_enable&&write_head>4)$fatal(1,"ATTENTION_Q_PAIR_WRITE_RANGE");
  if(read_enable&&write_enable&&(read_slot==write_slot)&&(read_address==write_address))
   $fatal(1,"ATTENTION_Q_PAIR_READ_WRITE_COLLISION");
 end
 // synthesis translate_on
endmodule

