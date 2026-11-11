`timescale 1ns/1ps
// Five physical256 INT16/INT48 islands. The root owns only command admission,
// shared-A L2 storage and group completion. Each island owns its B/C addresses,
// phase counters, read/write enables and ordered384-bit result drain.
(* keep_hierarchy="yes" *)
module c209_five_mac_core(
 input wire clk,rst,start,continue_run,output wire done,idle,
 input wire [105:0] commands_dout,input wire commands_empty_n,output wire commands_read,
 input wire [1279:0] a_dout,b_dout,input wire [4:0] a_empty_n,b_empty_n,
 output wire [4:0] a_read,b_read,
 output wire [1919:0] results_din,input wire [4:0] results_full_n,
 output wire [4:0] results_write,output wire [255:0] perf_snapshot
);
 localparam IDLE=0,WAIT=1,RUN=2,DONE=3,ERROR=4;
 reg [2:0] state;
 reg pending_valid,pending_bank,active_bank,active_commit;
 reg [105:0] pending;
 reg [4:0] active_mask,c_valid;
 reg [8:0] c_m[0:4],c_n[0:4];
 // The two existing B banks retain K and V for the second query group.
 // Valid means a complete private/all-five B load with this geometry; no
 // payload or DDR address tags are added. Fixed HLS ordering owns identity.
 reg [1:0] b_valid;
 reg [8:0] b_n[0:1],b_k[0:1];
 wire pending_stop=pending[98],pending_private=pending[104];
 wire pending_reuse_b=pending[105];
 wire [4:0] pending_mask=pending[103:99];
 wire [8:0] pending_m=pending[8:0],pending_n=pending[40:32],pending_k=pending[72:64];
 wire command_ok=(commands_dout[31:0]==32||commands_dout[31:0]==128||commands_dout[31:0]==256)&&
  (commands_dout[104]?(commands_dout[31:0]==32&&
    (commands_dout[63:32]==16||commands_dout[63:32]==128)):
    (commands_dout[31:0]!=32&&(commands_dout[63:32]==64||commands_dout[63:32]==128)))&&
  (commands_dout[95:64]==16||commands_dout[95:64]==64||commands_dout[95:64]==128)&&
  (commands_dout[103:99]==5'b00001||commands_dout[103:99]==5'b00011||
   commands_dout[103:99]==5'b00111||commands_dout[103:99]==5'b11111)&&
  (!commands_dout[104]||commands_dout[103:99]==5'b11111)&&
  (!commands_dout[105]||(commands_dout[104]&&commands_dout[103:99]==5'b11111));
 assign commands_read=!rst&&commands_empty_n&&!pending_valid&&(state==WAIT||state==RUN);
 wire accept=commands_read;
 wire [4:0] load_a_done,load_b_done,island_done,issue_multiply,in_seed,in_store,output_blocked;
 wire a_complete=&load_a_done,b_complete=&load_b_done;
 reg continuation_ok;
 integer j;
 always @*begin
  continuation_ok=1'b1;
  for(integer t=0;t<5;t=t+1)
   if(pending_mask[t]&&!pending[96]&&(!c_valid[t]||c_m[t]!=pending_m||c_n[t]!=pending_n))continuation_ok=1'b0;
 end
 wire reuse_b_ok=!pending_reuse_b||(pending_private&&pending_mask==5'b11111&&
  b_valid[pending_bank]&&b_n[pending_bank]==pending_n&&b_k[pending_bank]==pending_k);
 wire launch=state==WAIT&&pending_valid&&!pending_stop&&a_complete&&b_complete&&continuation_ok&&reuse_b_ok;
 // Private A uses C_low1024..1279; private results use0..255. A shard may
 // prefetch pending private A during its own OUT/FINISHED phase while other
 // shards finish their result drains. Group launch remains WAIT/all-five.
 wire private_load_allowed=state==WAIT;
 wire private_prefetch_allowed=state==RUN&&pending_valid&&pending_private;
 wire shared_write,shared_read;
 wire [11:0] shared_write_addr,shared_read_addr;
 wire [255:0] shared_a_values;
 genvar lane,branch,s;
 generate for(lane=0;lane<16;lane=lane+1)begin:shared_a_banks
  (* ram_style="block" *)reg [15:0] a_mem[0:4095];
  reg [15:0] a_q;
  always @(posedge clk)begin
   if(shared_write)a_mem[shared_write_addr]<=a_dout[lane*16+:16];
   if(shared_read)a_q<=a_mem[shared_read_addr];
  end
  assign shared_a_values[lane*16+:16]=a_q;
 end endgenerate
 // Two regional relays then five island leaves: equal two-register depth,
 // fanout2 at L2 and fanout3/2 at the relay. B and private A match that depth.
 wire [511:0] a_regional;
 generate for(branch=0;branch<2;branch=branch+1)begin:a_broadcast_regions
  (* DONT_TOUCH="yes" *)reg [255:0] relay;
  always @(posedge clk)relay<=shared_a_values;
  assign a_regional[branch*256+:256]=relay;
 end endgenerate
 generate for(s=0;s<5;s=s+1)begin:shards
  wire local_shared_write,local_shared_read;
  wire [11:0] local_shared_write_addr,local_shared_read_addr;
  c209_five_mac_shard #(.SHARD_ID(s)) tile(
   .clk(clk),.rst(rst),.prepare(accept),.load_command(commands_dout),.load_bank(~active_bank),
   .private_load_allowed(private_load_allowed),.private_prefetch_allowed(private_prefetch_allowed),
   .load_a_done(load_a_done[s]),.load_b_done(load_b_done[s]),
   .a_dout(a_dout[s*256+:256]),.b_dout(b_dout[s*256+:256]),
   .a_empty_n(a_empty_n[s]),.b_empty_n(b_empty_n[s]),.a_read(a_read[s]),.b_read(b_read[s]),
   .run_start(launch),.run_selected(pending_mask[s]),.run_command(pending),.run_bank(pending_bank),
   .regional_a(a_regional[(s<3?0:256)+:256]),
   .shared_write(local_shared_write),.shared_write_addr(local_shared_write_addr),
   .shared_read(local_shared_read),.shared_read_addr(local_shared_read_addr),
   .results_din(results_din[s*384+:384]),.results_full_n(results_full_n[s]),.results_write(results_write[s]),
   .done(island_done[s]),.issue_multiply(issue_multiply[s]),.in_seed(in_seed[s]),
   .in_store(in_store[s]),.output_blocked(output_blocked[s]));
  if(s==0)begin:shared_a_owner
   assign shared_write=local_shared_write;assign shared_write_addr=local_shared_write_addr;
   assign shared_read=local_shared_read;assign shared_read_addr=local_shared_read_addr;
  end
 end endgenerate
 assign done=state==DONE;
 assign idle=state==IDLE;
 always @(posedge clk)begin
  if(rst)begin
   state<=IDLE;pending_valid<=0;pending<=0;pending_bank<=0;active_bank<=1;
   active_commit<=0;active_mask<=0;c_valid<=0;b_valid<=0;
   for(j=0;j<5;j=j+1)begin c_m[j]<=0;c_n[j]<=0;end
   for(j=0;j<2;j=j+1)begin b_n[j]<=0;b_k[j]<=0;end
  end else begin
   if(accept)begin
    pending_valid<=1;pending<=commands_dout;pending_bank<=~active_bank;
    // Fresh data may already overwrite the inactive bank during the current
    // compute. Retire its old tag immediately; mark it valid only at launch.
    if(!commands_dout[98]&&!commands_dout[105])b_valid[~active_bank]<=0;
   end
   if(accept&&!commands_dout[98]&&!command_ok)state<=ERROR;
   else case(state)
    IDLE:if(start)begin state<=WAIT;c_valid<=0;b_valid<=0;end
    WAIT:if(pending_valid)begin
     if(pending_stop)begin
      pending_valid<=0;b_valid<=0;
      if(c_valid!=0)state<=ERROR;else state<=DONE;
     end else if(!reuse_b_ok)state<=ERROR;
     else if(a_complete&&b_complete)begin
      if(!continuation_ok)state<=ERROR;
      else begin
       pending_valid<=0;active_bank<=pending_bank;active_mask<=pending_mask;active_commit<=pending[97];
       if(!pending_reuse_b)begin
        b_valid[pending_bank]<=pending_private&&pending_mask==5'b11111;
        b_n[pending_bank]<=pending_n;b_k[pending_bank]<=pending_k;
       end
       for(j=0;j<5;j=j+1)if(pending_mask[j])begin c_valid[j]<=1;c_m[j]<=pending_m;c_n[j]<=pending_n;end
       state<=RUN;
      end
     end
    end
    RUN:if(&island_done)begin
     if(active_commit)c_valid<=c_valid&~active_mask;
     state<=WAIT;
    end
    DONE:if(continue_run)state<=IDLE;
    ERROR:state<=ERROR;
    default:state<=ERROR;
   endcase
  end
 end
 // FAST_GROUP_V1 occupancy follows island0 (always active in the fixed masks).
 reg [31:0] perf_elapsed,perf_multiply,perf_load_a,perf_load_b,perf_seed,perf_store,perf_output_blocked,perf_other;
 assign perf_snapshot={perf_other,perf_output_blocked,perf_store,perf_seed,perf_load_b,perf_load_a,perf_multiply,perf_elapsed};
 always @(posedge clk)begin
  if(rst||(state==IDLE&&start))begin
   perf_elapsed<=0;perf_multiply<=0;perf_load_a<=0;perf_load_b<=0;perf_seed<=0;perf_store<=0;perf_output_blocked<=0;perf_other<=0;
  end else if(state!=IDLE&&state!=DONE&&perf_elapsed!=32'hffffffff)begin
   perf_elapsed<=perf_elapsed+1;
   if(issue_multiply[0])perf_multiply<=perf_multiply+1;
   else if(state==WAIT&&pending_valid&&!a_complete)perf_load_a<=perf_load_a+1;
   else if(state==WAIT&&pending_valid&&!b_complete)perf_load_b<=perf_load_b+1;
   else if(in_seed[0])perf_seed<=perf_seed+1;
   else if(in_store[0])perf_store<=perf_store+1;
   else if(|output_blocked)perf_output_blocked<=perf_output_blocked+1;
   else perf_other<=perf_other+1;
  end
 end
 // synthesis translate_off
 always @(posedge clk)if(!rst)begin
  if(state==ERROR)$fatal(1,"FIVE_MAC_PROTOCOL_ERROR");
  if(launch&&pending_bank==active_bank)$fatal(1,"FIVE_OPERAND_BANK_REUSE");
  if(launch&&pending_reuse_b&&!reuse_b_ok)$fatal(1,"FIVE_B_REUSE_TAG_MISMATCH");
 end
 // synthesis translate_on
endmodule

// One physically grouped256-DSP island. No sharedK/rb/nb/address/CE bus enters
// its B/C banks. Only group-level prepare/launch and registered A data cross.
(* keep_hierarchy="yes" *)
module c209_five_mac_shard #(parameter SHARD_ID=0)(
 input wire clk,rst,prepare,input wire [105:0] load_command,input wire load_bank,private_load_allowed,private_prefetch_allowed,
 output reg load_a_done,load_b_done,
 input wire [255:0] a_dout,b_dout,input wire a_empty_n,b_empty_n,output wire a_read,b_read,
 input wire run_start,run_selected,input wire [105:0] run_command,input wire run_bank,
 input wire [255:0] regional_a,
 output wire shared_write,shared_read,output wire [11:0] shared_write_addr,shared_read_addr,
 output wire [383:0] results_din,input wire results_full_n,output wire results_write,
 output wire done,issue_multiply,in_seed,in_store,output_blocked
);
 localparam FINISHED=0,TILE=1,CLEAR=2,SEED=3,MUL=4,STORE=5,OUT=6;
 // This is an ordinary synchronous reset root, not a CDC stage. Let Vivado
 // replicate it with the address counters instead of protecting its input net
 // with DONT_TOUCH (which blocked all 35 issue_k replications in placement09).
 (* MAX_FANOUT=64 *)reg local_rst=1'b1;
 reg [2:0] state;
 reg private_a,do_reset,do_commit,active_bank;
 reg load_valid,load_private,load_active_bank;
 reg [8:0] load_m,load_n,load_k;
 reg [6:0] ak,bk;
 reg [2:0] a_panel;
 reg [1:0] b_panel;
 reg a_half,b_half;
 wire need_a=load_command[104]?load_command[99+SHARD_ID]:(SHARD_ID==0);
 wire prepare_valid=!load_command[98];
 wire private_a_load_window=private_load_allowed||
  (private_prefetch_allowed&&private_a&&(state==OUT||state==FINISHED));
 assign a_read=!rst&&!local_rst&&load_valid&&!load_a_done&&a_empty_n&&(!load_private||private_a_load_window);
 assign b_read=!rst&&!local_rst&&load_valid&&!load_b_done&&b_empty_n;
 assign shared_write=(SHARD_ID==0)&&a_read&&!load_private;
 assign shared_write_addr={load_active_bank,a_panel,a_half,ak};
 // Geometry is payload qualified by load_valid. Capture it only at prepare;
 // resetting it would add a reset/CE combination to every island although no
 // RAM port can observe it before this token has initialized all fields.
 always @(posedge clk)if(prepare)begin
  load_private<=load_command[104];load_active_bank<=load_bank;
  load_m<=load_command[8:0];load_n<=load_command[40:32];load_k<=load_command[72:64];
 end
 always @(posedge clk)begin
  local_rst<=rst;
  if(local_rst)begin
   load_valid<=0;
   load_a_done<=1;load_b_done<=1;ak<=0;bk<=0;a_panel<=0;b_panel<=0;a_half<=0;b_half<=0;
  end else if(prepare)begin
   load_valid<=prepare_valid;
   load_a_done<=!prepare_valid||!need_a;
   // Reuse commands still load private A. B RAM and its existing bank data
   // are untouched; the root checks the retained bank's validity/geometry.
   load_b_done<=!prepare_valid||!load_command[99+SHARD_ID]||load_command[105];
   ak<=0;bk<=0;a_panel<=0;b_panel<=0;a_half<=0;b_half<=0;
  end else begin
   if(a_read)begin
    if(!a_half)a_half<=1;
    else begin
     a_half<=0;
     if({2'b00,ak}+1==load_k)begin
      ak<=0;
      if(({1'b0,a_panel}+1)*32>=load_m)load_a_done<=1;
      else a_panel<=a_panel+1;
     end else ak<=ak+1;
    end
   end
   if(b_read)begin
    if(!b_half)b_half<=1;
    else begin
     b_half<=0;
     if({2'b00,bk}+1==load_k)begin
      bk<=0;
      // N16 deliberately consumes a full32-row transposer panel. Only its
      // first16 columns are subsequently addressed by the compute loop.
      if(({1'b0,b_panel}+1)*32>=load_n)load_b_done<=1;
      else b_panel<=b_panel+1;
     end else bk<=bk+1;
    end
   end
  end
 end
 reg [8:0] k;
 (* MAX_FANOUT=64 *)reg [8:0] issue_k;
 reg [3:0] rb;
 reg [3:0] last_row_block;
 reg [2:0] nb;
 reg [2:0] last_column_block;
 reg [7:0] last_row;
 reg [7:0] out_row;
 reg [2:0] out_block;
 reg out_half,out_half_d,out_valid,out_last,out_issued_last;
 reg [5:0] mul_valid;
 reg [4:0] seed_issue;
 reg seed_valid;
 reg [3:0] seed_row_d,store_index;
 // Four independent four-row PC chains restore/retire four rows per clock.
 // The column banks remove row[3:2] from the RAM address and select that
 // bank locally; the externally visible C/private-A address is unchanged.
 wire seed_issue_en=state==SEED&&seed_issue<4;
 wire multiply_issue=state==MUL&&issue_k<k;
 wire out_issue=state==OUT&&!out_issued_last&&(!out_valid||results_full_n);
 wire [7:0] seed_read_row={rb,~seed_issue[3:0]};
 wire [10:0] c_write_addr={rb,~store_index,nb};
 wire [10:0] c_read_addr=(state==MUL&&private_a)?{3'b100,rb[0],issue_k[6:0]}:
                         state==SEED?{seed_read_row,nb}:{out_row,out_block};
 assign shared_read=(SHARD_ID==0)&&multiply_issue&&!private_a;
 assign shared_read_addr={active_bank,rb,issue_k[6:0]};
 wire [255:0] private_a_values;
 (* DONT_TOUCH="yes" *)reg [255:0] private_a_relay,a_distribute_q;
 always @(posedge clk)begin
  private_a_relay<=private_a_values;
  a_distribute_q<=private_a?private_a_relay:regional_a;
 end
 wire [767:0] bank_results;
 genvar g;
 generate for(g=0;g<4;g=g+1)begin:column_banks
  c209_five_mac_column_bank #(.COLUMN_BASE(g*4)) bank(.clk(clk),.rst(local_rst),
   .b_write_en(b_read),.b_write_addr({load_active_bank,b_panel,b_half,bk}),
   .b_read_addr({active_bank,nb,issue_k[6:0]}),.b_write_data(b_dout[g*64+:64]),
   .a_write_en(a_read&&load_private),.a_write_addr({3'b100,a_half,ak}),
   .a_write_data(a_dout[g*64+:64]),.c_read_addr(c_read_addr),.c_write_addr(c_write_addr),
   .multiply_issue(multiply_issue),
   .c_read_en(seed_issue_en||(out_issue&&(out_half==(g>=2)))||(multiply_issue&&private_a)),
   .c_write_en(state==STORE),.tile_start(state==TILE),.tile_reset(do_reset),.tile_k(k),
   .a_values(a_distribute_q),.private_a_values(private_a_values[g*64+:64]),
   .results(bank_results[g*192+:192]));
 end endgenerate
 // Read two192-bit C banks per cycle. The half tag follows the synchronous
 // BRAM read. A blocked half keeps its C outputs/tag unchanged; no768-bit
 // intermediate packet, packet-wide holding CE or off-island mux is needed.
 assign results_din=out_half_d?bank_results[767:384]:bank_results[383:0];
 assign results_write=!rst&&!local_rst&&state==OUT&&out_valid&&results_full_n;
 assign done=state==FINISHED;
 assign issue_multiply=multiply_issue;
 assign in_seed=state==SEED;
 assign in_store=state==STORE;
 assign output_blocked=state==OUT&&out_valid&&!results_full_n;
 // The FINISHED/valid reset gates make these per-command fields unobservable
 // until run_start. Keep them out of the local reset distribution and leave
 // their CE as the one narrow island launch token.
 always @(posedge clk)if(run_start)begin
  private_a<=run_command[104];do_reset<=run_command[96];do_commit<=run_command[97];active_bank<=run_bank;
  k<=run_command[72:64];
  last_row_block<=run_command[8]?4'd15:run_command[7]?4'd7:4'd1;
  last_row<=run_command[8]?8'd255:run_command[7]?8'd127:8'd31;
  last_column_block<=run_command[39]?3'd7:run_command[36]?3'd0:3'd3;
 end
 always @(posedge clk)begin
  if(local_rst)begin
   state<=FINISHED;
   issue_k<=0;rb<=0;nb<=0;
   mul_valid<=0;seed_issue<=0;seed_valid<=0;seed_row_d<=0;store_index<=0;
   out_row<=0;out_block<=0;out_half<=0;out_half_d<=0;out_valid<=0;out_last<=0;out_issued_last<=0;
  end else if(run_start)begin
   state<=run_selected?TILE:FINISHED;rb<=0;nb<=0;
   out_valid<=0;out_issued_last<=0;
  end else case(state)
   TILE:begin
    issue_k<=0;mul_valid<=0;store_index<=0;seed_issue<=0;seed_valid<=0;
    state<=do_reset?CLEAR:SEED;
   end
   CLEAR:state<=MUL;
   SEED:begin
    seed_valid<=seed_issue_en;
    if(seed_issue_en)begin seed_row_d<=seed_issue[3:0];seed_issue<=seed_issue+1;end
    if(seed_valid&&seed_row_d==3)begin seed_valid<=0;state<=MUL;end
   end
   MUL:begin
    mul_valid<={mul_valid[4:0],multiply_issue};
    if(multiply_issue)issue_k<=issue_k+1;
    if(issue_k==k&&mul_valid==0)state<=STORE;
   end
   STORE:begin
    if(store_index!=3)store_index<=store_index+1;
    else if(nb!=last_column_block)begin nb<=nb+1'b1;state<=TILE;end
    else if(rb!=last_row_block)begin rb<=rb+1'b1;nb<=0;state<=TILE;end
    else if(do_commit)begin state<=OUT;out_row<=0;out_block<=0;out_half<=0;out_valid<=0;out_issued_last<=0;end
    else state<=FINISHED;
   end
   OUT:begin
    if(results_write)begin
     out_valid<=0;
     if(out_last)state<=FINISHED;
    end
    if(out_issue)begin
     out_valid<=1;out_half_d<=out_half;
     out_last<=out_half&&out_row==last_row&&out_block==last_column_block;
     if(out_half&&out_row==last_row&&out_block==last_column_block)out_issued_last<=1;
     else if(!out_half)out_half<=1;
     else begin
      out_half<=0;
      if(out_block==last_column_block)begin out_row<=out_row+1'b1;out_block<=0;end
      else out_block<=out_block+1'b1;
     end
    end
   end
   default:state<=FINISHED;
  endcase
 end
 // synthesis translate_off
 always @(posedge clk)if(!rst&&!local_rst)begin
  if(a_read&&load_private&&!private_a_load_window)$fatal(1,"FIVE_PRIVATE_A_ACTIVE_C_OVERWRITE");
  if(a_read&&load_private&&!private_load_allowed&&
     (!private_a||(state!=OUT&&state!=FINISHED)))$fatal(1,"FIVE_PRIVATE_A_PREFETCH_PHASE");
  if(a_read&&load_private&&state==OUT&&c_read_addr>=256)$fatal(1,"FIVE_PRIVATE_A_PREFETCH_ADDRESS");
  if(run_start&&state!=FINISHED)$fatal(1,"FIVE_ISLAND_OVERLAP");
  if(results_write&&out_last&&!out_half_d)$fatal(1,"FIVE_LAST_RESULT_HALF_ALIGNMENT");
 end
 // synthesis translate_on
endmodule
