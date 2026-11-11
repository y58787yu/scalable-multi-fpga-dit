`timescale 1ns/1ps
// Four columns (64 DSPs), split into four local four-row accumulator chains.
// B retains two K128 banks. C is four 512x192 banks: twelve RAMB36 per
// four-column bank (48/shard) when packed as intended. Logical private-A
// remains the low16 bits of original C addresses1024..1279; private C0..255.
(* keep_hierarchy="yes" *)
module c209_five_mac_column_bank #(parameter COLUMN_BASE=0)(
 input wire clk,rst,
 input wire b_write_en,input wire [10:0] b_write_addr,b_read_addr,
 input wire [63:0] b_write_data,
 input wire a_write_en,input wire [10:0] a_write_addr,input wire [63:0] a_write_data,
 input wire [10:0] c_read_addr,c_write_addr,
 input wire multiply_issue,c_read_en,c_write_en,
 input wire tile_start,tile_reset,input wire [8:0] tile_k,
 input wire [255:0] a_values,
 output wire [63:0] private_a_values,output wire [191:0] results
);
 wire [63:0] b_values;
 wire [767:0] seed_values,bottom;
 // A narrow tile token is the only phase-control input. Every bank keeps its
 // own sequencer adjacent to its 64 DSPs; no central state decode reaches an
 // unregistered DSP ALU. Ordinary synchronous control may be replicated by
 // synthesis/physical optimization; only the operand relay/CDC boundaries
 // elsewhere require DONT_TOUCH.
 localparam PH_IDLE=5'b00001,PH_CLEAR=5'b00010,PH_SEED=5'b00100,
            PH_MUL=5'b01000,PH_STORE=5'b10000;
 (* MAX_FANOUT=64 *)reg local_rst=1'b1;
 (* fsm_encoding="none" *)reg [4:0] phase=PH_IDLE;
 reg [7:0] phase_count=0;
 reg [7:0] mul_last;
 reg [4:0] issue_pipe=0;
 // Capture immutable geometry at the existing TILE edge. Mirror the K and
 // STORE counters at the same edges as the shard, rather than registering a
 // BRAM address one cycle late. Each four-column bank now drives its own
 // B-read and C-write address trees from nearby FFs. No extra SEED, product,
 // STORE or output cycle is introduced; payload FFs need no reset because
 // the existing tile/valid state qualifies every access.
 (* KEEP="yes" *)reg b_active_bank;
 (* KEEP="yes" *)reg [6:0] read_k;
 (* KEEP="yes" *)reg [3:0] c_row_block,store_row;
 (* KEEP="yes" *)reg [2:0] c_column_block;
 always @(posedge clk)begin
  if(tile_start)begin
   b_active_bank<=b_read_addr[10];read_k<=0;store_row<=0;
   c_row_block<=c_write_addr[10:7];c_column_block<=c_write_addr[2:0];
  end else begin
   if(multiply_issue)read_k<=read_k+1'b1;
   if(c_write_en)store_row<=store_row+1'b1;
  end
 end
 wire [10:0] bank_b_read_addr={b_active_bank,c_column_block,read_k};
 wire [10:0] bank_c_write_addr={c_row_block,~store_row,c_column_block};
 // In private-A mode only the K field varies during MUL. Other C reads are
 // the original synchronous SEED/OUT address, with identical latency.
 wire [10:0] bank_c_read_addr=multiply_issue?{c_read_addr[10:7],read_k}:c_read_addr;
 // A tile token always supplies K before PH_MUL. This immutable per-tile
 // payload needs one local CE, not the reset net used by phase/valid state.
 always @(posedge clk)if(tile_start)
  mul_last<=tile_k[7]?8'd134:tile_k[6]?8'd70:8'd22;
 always @(posedge clk)begin
  local_rst<=rst;
  if(local_rst)begin phase<=PH_IDLE;phase_count<=0;issue_pipe<=0;end
  else begin
   issue_pipe<={issue_pipe[3:0],multiply_issue};
   if(tile_start)begin
    phase<=tile_reset?PH_CLEAR:PH_SEED;
    phase_count<=0;
   end else case(phase)
    PH_CLEAR:begin phase<=PH_MUL;phase_count<=0;end
    // Four reads seed four independent four-row chains. The existing BRAM
    // and CREG latency stays unchanged: last seed is consumed at MUL count0.
    PH_SEED:if(phase_count==4)begin phase<=PH_MUL;phase_count<=0;end
            else phase_count<=phase_count+1'b1;
    // Two equal-depth regional/leaf A stages precede the bank distributor.
    // P consumes each issued product six clocks later; one drain precedes STORE.
    PH_MUL:if(phase_count==mul_last)begin phase<=PH_STORE;phase_count<=0;end
           else phase_count<=phase_count+1'b1;
    PH_STORE:if(phase_count==3)begin phase<=PH_IDLE;phase_count<=0;end
             else phase_count<=phase_count+1'b1;
    default:begin phase<=PH_IDLE;phase_count<=0;end
   endcase
  end
 end
 // These are NEXT-edge operations. OPMODE's internal FF and a local CEP FF
 // sample them together. CLEAR, seed CREG, six-stage MUL and STORE remain
 // aligned; especially, seed count4 prepares the last seed at MUL count0.
 wire next_clear=!local_rst&&tile_start&&tile_reset;
 wire next_seed=!local_rst&&phase==PH_SEED&&phase_count!=0;
 wire next_multiply=!local_rst&&issue_pipe[4];
 wire next_shift=!local_rst&&((phase==PH_MUL&&phase_count==mul_last)||
                              (phase==PH_STORE&&phase_count!=3));
 // Original address={row[7:0],column_block[2:0]}. Partition by the
 // four-row chain ID, row[3:2], NOT row%4: four chain tails must write in
 // parallel. All four banks then use the same nine-bit SEED/STORE address.
 wire [1:0] selected_bank=bank_c_read_addr[6:5];
 wire [8:0] compact_read_addr={bank_c_read_addr[10:7],bank_c_read_addr[4:0]};
 wire [8:0] compact_write_addr={bank_c_write_addr[10:7],bank_c_write_addr[4:0]};
 wire [1:0] a_bank=a_write_addr[6:5];
 wire [8:0] compact_a_addr={a_write_addr[10:7],a_write_addr[4:0]};
 wire [191:0] c_values[0:3];
 reg [1:0] selected_bank_q;
 always @(posedge clk)if(c_read_en)selected_bank_q<=selected_bank;
 // These muxes stay inside this four-column bank. No wide C crossbar is
 // introduced between islands. Only OUT/private-A select a single bank;
 // SEED feeds all four local chain heads directly from their own BRAMs.
 wire [191:0] selected_c=c_values[selected_bank_q];
 assign results=selected_c;
 genvar c,g;
 generate for(c=0;c<4;c=c+1)begin:columns
  (* ram_style="block" *) reg [15:0] b_mem[0:2047];
  reg [15:0] b_q,b_regional_q,b_distribute_q;
  always @(posedge clk)begin
   if(b_write_en)b_mem[b_write_addr]<=b_write_data[c*16+:16];
   if(multiply_issue)b_q<=b_mem[bank_b_read_addr];
   b_regional_q<=b_q;b_distribute_q<=b_regional_q;
  end
  assign b_values[c*16+:16]=b_distribute_q;
  assign private_a_values[c*16+:16]=selected_c[c*48+:16];
 end
 for(g=0;g<4;g=g+1)begin:c_row_chains
  wire private_write=a_write_en&&a_bank==g;
  wire write_enable=private_write||c_write_en;
  wire [8:0] write_address=private_write?compact_a_addr:compact_write_addr;
  wire [191:0] write_data;
  wire [23:0] write_bytes;
  for(c=0;c<4;c=c+1)begin:write_columns
   // Upper32 bits have WE=0 for A, so leave their DSP wires direct.
   // Muxing an unused zero there needlessly adds 512 data gates per bank.
   // Use the shared A-write bit for data selection. The bank-qualified
   // byte WE already suppresses every unselected bank. Repeating the bank
   // decode in each data mux prevents otherwise legal paired-LUT packing.
   assign write_data[c*48+:16]=a_write_en?a_write_data[c*16+:16]:bottom[g*192+c*48+:16];
   assign write_data[c*48+16+:32]=bottom[g*192+c*48+16+:32];
   // Preserve the other C planes on a private-A write. This is a genuine
   // single physical write port with byte enables, never a RAM read/modify.
   assign write_bytes[c*6+:6]=private_write?6'b000011:{6{c_write_en}};
  end
  wire read_enable=c_read_en&&(phase==PH_SEED||selected_bank==g);
  xpm_memory_sdpram #(
   .ADDR_WIDTH_A(9),.ADDR_WIDTH_B(9),.AUTO_SLEEP_TIME(0),
   .BYTE_WRITE_WIDTH_A(8),.CASCADE_HEIGHT(0),.CLOCKING_MODE("common_clock"),
   .ECC_MODE("no_ecc"),.MEMORY_INIT_FILE("none"),.MEMORY_INIT_PARAM("0"),
   .MEMORY_OPTIMIZATION("true"),.MEMORY_PRIMITIVE("block"),.MEMORY_SIZE(98304),
   .MESSAGE_CONTROL(0),.READ_DATA_WIDTH_B(192),.READ_LATENCY_B(1),
   .READ_RESET_VALUE_B("0"),.RST_MODE_B("SYNC"),.SIM_ASSERT_CHK(1),
   .USE_EMBEDDED_CONSTRAINT(0),.USE_MEM_INIT(0),.WAKEUP_TIME("disable_sleep"),
   .WRITE_DATA_WIDTH_A(192),.WRITE_MODE_B("read_first")
  ) storage(
   .clka(clk),.clkb(clk),.ena(write_enable),.wea(write_bytes),
   .addra(write_address),.dina(write_data),.enb(read_enable),
   .addrb(compact_read_addr),.doutb(c_values[g]),.regceb(1'b1),.rstb(1'b0),
   .sleep(1'b0),.injectsbiterra(1'b0),.injectdbiterra(1'b0)
  );
  assign seed_values[g*192+:192]=c_values[g];
 end endgenerate
 c209_dsp_array16x4 #(.PIPELINED_CONTROL(1),.CHAIN_ROWS(4)) array(.clk(clk),.rst(local_rst),
  .clear(1'b0),.seed_valid(1'b0),.multiply_valid(1'b0),.shift(1'b0),
  .next_clear(next_clear),.next_seed(next_seed),.next_multiply(next_multiply),.next_shift(next_shift),
  .a(a_values),.b(b_values),.seed(seed_values),.bottom(bottom));
 // synthesis translate_off
 always @(posedge clk)if(!rst)begin
  if(a_write_en&&c_write_en)$fatal(1,"FIVE_PRIVATE_A_C_WRITE_COLLISION");
  if(a_write_en&&(a_write_addr<1024||a_write_addr>1279))$fatal(1,"FIVE_PRIVATE_A_ADDRESS");
  if(next_seed&&(next_clear||next_multiply||next_shift))$fatal(1,"FIVE_SEED_CREG_MODE_OVERLAP");
  if(tile_start&&phase!=PH_IDLE)$fatal(1,"FIVE_BANK_TILE_OVERLAP");
 end
 // synthesis translate_on
endmodule
