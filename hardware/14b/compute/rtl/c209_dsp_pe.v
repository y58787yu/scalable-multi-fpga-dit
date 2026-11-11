`timescale 1ns/1ps
// One signed16 x signed16 multiply with DSP-internal M+P accumulation.
module c209_dsp_pe #(parameter C_PIPELINE=0,parameter OPMODE_PIPELINE=0)(
 input wire clk,rst,p_enable,
 input wire [29:0] a,
 input wire [17:0] b,
 input wire [47:0] seed,pcin,
 input wire [8:0] opmode,
 output wire [47:0] p,pcout
);
 DSP48E2 #(
  .AREG(1),.ACASCREG(1),.BREG(1),.BCASCREG(1),.MREG(1),.PREG(1),
  .CREG(C_PIPELINE),.DREG(0),.ADREG(0),.INMODEREG(0),.OPMODEREG(OPMODE_PIPELINE),
  .ALUMODEREG(0),.CARRYINREG(0),.CARRYINSELREG(0),
  .A_INPUT("DIRECT"),.B_INPUT("DIRECT"),.AMULTSEL("A"),.BMULTSEL("B"),
  .PREADDINSEL("A"),.USE_MULT("MULTIPLY"),.USE_SIMD("ONE48"),
  .USE_PATTERN_DETECT("NO_PATDET")
 ) dsp (
  .CLK(clk),.A(a),.B(b),.C(seed),.D(27'd0),
  .ACIN(30'd0),.BCIN(18'd0),.PCIN(pcin),.P(p),.PCOUT(pcout),
  .ALUMODE(4'd0),.INMODE(5'd0),.OPMODE(opmode),
  .CARRYIN(1'b0),.CARRYINSEL(3'd0),.CARRYCASCIN(1'b0),.MULTSIGNIN(1'b0),
  .CEA1(1'b1),.CEA2(1'b1),.CEB1(1'b1),.CEB2(1'b1),.CEM(1'b1),.CEP(p_enable),
  .CEC(1'b1),.CED(1'b1),.CEAD(1'b1),.CEINMODE(1'b1),.CECTRL(1'b1),
  .CEALUMODE(1'b1),.CECARRYIN(1'b1),
  // Only live pipeline registers consume reset. D/AD, INMODE, ALUMODE and
  // external carry registers are bypassed. Internal qmultcarryin is also
  // unobserved: CARRYINSEL is permanently 000, never multiplier-carry 110.
  .RSTA(rst),.RSTB(rst),.RSTC(C_PIPELINE?rst:1'b0),.RSTD(1'b0),.RSTM(rst),.RSTP(rst),
  .RSTINMODE(1'b0),.RSTCTRL(OPMODE_PIPELINE?rst:1'b0),.RSTALUMODE(1'b0),.RSTALLCARRYIN(1'b0)
 );
endmodule
