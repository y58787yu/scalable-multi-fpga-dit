`timescale 1ns/1ps
// Drop-in replacement for the frozen W IP's probability memcore ONLY.
// Keep all five stream-of-blocks slots, commit/release signals, and metadata.
// The original logical address is {word[8:0], slot[2:0]}, slot in 0..4.
// Words 0..255 are P payload; words 256..263 are alpha/denominator headers.
// Dense slot-major payload storage removes the unused slot-5/6/7 holes.
//
// Frozen W port contract (ht4fivewp/hdl/verilog):
// - pipeline_1.v:953 ties consumer writes to zero.
// - softmax_five.v:2385 passes producer q0 ONLY to SOFTMAX_ONLINE_STATE.
// - that process's addresses are 256..263 (lines 913/915), for header RMW.
// - SOFTMAX_PROBABILITY_ROWS_SOFTMAX_PROBABILITY_PACK has no q0 input.
// Payload is therefore one writer / one reader. The producer's observable
// header readback retains synchronous one-cycle read-first behavior.
// Unobserved payload q0 readback is zero; this is not a generic TDP RAM.
module cosmos2b_tile_pipeline_1_probability_RAM_AUTO_1R1W_memcore (
    address0, ce0, d0, we0, q0,
    address1, ce1, d1, we1, q1,
    reset, clk
);
 parameter DataWidth = 256;
 parameter AddressWidth = 12;
 parameter AddressRange = 2112;

 input wire [AddressWidth-1:0] address0, address1;
 input wire ce0, ce1;
 input wire [DataWidth-1:0] d0, d1;
 input wire we0, we1;
 output reg [DataWidth-1:0] q0;
 output wire [DataWidth-1:0] q1;
 input wire reset, clk;

 wire [2:0] slot0 = address0[2:0];
 wire [2:0] slot1 = address1[2:0];
 wire [8:0] word0 = address0[11:3];
 wire [8:0] word1 = address1[11:3];
 wire valid0 = slot0 < 3'd5 && word0 <= 9'd263;
 wire valid1 = slot1 < 3'd5 && word1 <= 9'd263;
 wire header0 = word0[8];
 wire header1 = word1[8];
 wire [10:0] payload_addr0 = {slot0, word0[7:0]};
 wire [10:0] payload_addr1 = {slot1, word1[7:0]};
 wire [5:0] header_addr0 = {slot0, word0[2:0]};
 wire [5:0] header_addr1 = {slot1, word1[2:0]};

 (* ram_style="block" *) reg [DataWidth-1:0] payload [0:1279];
 (* ram_style="distributed" *) reg [DataWidth-1:0] header [0:39];
 reg [DataWidth-1:0] payload_q1, header_q1;
 reg read_header1;

 always @(posedge clk) begin
  if (ce0 && valid0 && !header0 && we0)
   payload[payload_addr0] <= d0;
  if (ce1 && valid1 && !header1)
   payload_q1 <= payload[payload_addr1];
 end

 // Full header words are written after the existing producer's read/modify/
 // write sequence. q0 must return the PRE-write value at every enabled read.
 // The consumer is a distinct owner and can read another slot concurrently.
 always @(posedge clk) begin
  if (ce0 && valid0 && header0 && we0) header[header_addr0] <= d0;
  if (ce0) begin
   if (valid0 && header0) q0 <= header[header_addr0];
   else q0 <= {DataWidth{1'b0}};
  end
  if (ce1 && valid1 && header1) header_q1 <= header[header_addr1];
  if (ce1 && valid1) read_header1 <= header1;
 end
 // One synchronous read edge, as before. CE low holds data and selection.
 assign q1 = read_header1 ? header_q1 : payload_q1;

 // The original memcore did not reset data or output registers. Ownership,
 // initialized only by the untouched wrapper, prevents premature exposure.
 // synthesis translate_off
 initial begin
  if (DataWidth != 256 || AddressWidth != 12 || AddressRange != 2112)
   $fatal(1, "C209_PROBABILITY_STORAGE_FIXED_GEOMETRY");
 end
 always @(posedge clk) if (!reset) begin
  if (ce0 && !valid0) $fatal(1, "C209_PROBABILITY_WRITE_ADDRESS");
  if (ce1 && !valid1) $fatal(1, "C209_PROBABILITY_READ_ADDRESS");
  if (ce1 && we1) $fatal(1, "C209_PROBABILITY_CONSUMER_WRITE");
  if (ce0 && valid0 && we0 && ce1 && valid1 && slot0 == slot1)
   $fatal(1, "C209_PROBABILITY_SLOT_OWNERSHIP_COLLISION");
 end
 // synthesis translate_on
endmodule
