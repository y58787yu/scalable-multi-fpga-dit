`timescale 1ns/1ps
// Production dual-clock HLS blackbox. No local clock generator, global ready
// lookahead or payload reset. Command and all payload crossings remain XPM.
(* keep_hierarchy="yes" *)
module c209_five_mac_rtl(
 input wire ap_clk,ap_rst,ap_ce,ap_start,ap_continue,
 input wire mac_fast_clk,mac_fast_rst,
 output wire ap_done,ap_idle,ap_ready,
 input wire [9:0] commands_dout,input wire commands_empty_n,output wire commands_read,
 input wire [255:0] a0_dout,input wire a0_empty_n,output wire a0_read,
 input wire [255:0] b0_dout,input wire b0_empty_n,output wire b0_read,
 output wire [383:0] r0_din,input wire r0_full_n,output wire r0_write,
 input wire [255:0] a1_dout,input wire a1_empty_n,output wire a1_read,
 input wire [255:0] b1_dout,input wire b1_empty_n,output wire b1_read,
 output wire [383:0] r1_din,input wire r1_full_n,output wire r1_write,
 input wire [255:0] a2_dout,input wire a2_empty_n,output wire a2_read,
 input wire [255:0] b2_dout,input wire b2_empty_n,output wire b2_read,
 output wire [383:0] r2_din,input wire r2_full_n,output wire r2_write,
 input wire [255:0] a3_dout,input wire a3_empty_n,output wire a3_read,
 input wire [255:0] b3_dout,input wire b3_empty_n,output wire b3_read,
 output wire [383:0] r3_din,input wire r3_full_n,output wire r3_write,
 input wire [255:0] a4_dout,input wire a4_empty_n,output wire a4_read,
 input wire [255:0] b4_dout,input wire b4_empty_n,output wire b4_read,
 output wire [383:0] r4_din,input wire r4_full_n,output wire r4_write,
 output wire [255:0] perf_din,input wire perf_full_n,output wire perf_write
);
 wire reset_request=ap_rst|mac_fast_rst;
 (* ASYNC_REG="TRUE" *)reg [3:0] slow_reset_pipe,fast_reset_pipe;
 always @(posedge ap_clk or posedge reset_request)
  if(reset_request)slow_reset_pipe<=4'b1111;else slow_reset_pipe<={slow_reset_pipe[2:0],1'b0};
 always @(posedge mac_fast_clk or posedge reset_request)
  if(reset_request)fast_reset_pipe<=4'b1111;else fast_reset_pipe<={fast_reset_pipe[2:0],1'b0};
 wire slow_rst=slow_reset_pipe[3],fast_rst=fast_reset_pipe[3];
 localparam IDLE=0,RUN=1,DONE=2;
 reg [1:0] slow_state;
 reg start_toggle,stop_seen;
 // At most command FIFO16 + current/pending groups may be credited.
 // Even19 maximum4096-beat results fit below2^17;20 bits retain margin.
 reg [19:0] a_owed[0:4],b_owed[0:4],r_owed[0:4];
 wire [1279:0] slow_a,slow_b,transposed_a,transposed_b,fast_a,fast_b;
 wire [1919:0] slow_result;
 wire [1919:0] fast_result;
 wire [4:0] slow_a_present,slow_b_present,slow_a_read,slow_b_read,slow_result_ready,slow_result_write;
 wire [4:0] a_space,b_space,fast_a_present,fast_b_present,core_a_read,core_b_read,result_space,result_present,core_result_write;
 wire [4:0] a_cfg_ready,b_cfg_ready,a_input_ready,b_input_ready,a_transposed_valid,b_transposed_valid;
 // M code3 is reserved for private M32 with B retained in its alternating
 // physical bank. Keep the imported HLS FIFO at10 bits; reuse is internal.
 function [105:0] expand_command;
  input [9:0] fixed;
  reg [8:0] mm,nn,kk;
  begin
   case(fixed[1:0])2'd0:mm=32;2'd1:mm=128;2'd2:mm=256;default:mm=32;endcase
   nn=fixed[2]?128:(fixed[9]?16:64);
   case(fixed[4:3])2'd0:kk=16;2'd1:kk=64;2'd2:kk=128;default:kk=0;endcase
   expand_command=0;expand_command[31:0]=mm;expand_command[63:32]=nn;expand_command[95:64]=kk;
   expand_command[96]=fixed[5];expand_command[97]=fixed[6];expand_command[98]=fixed[7];
   expand_command[103:99]=fixed[8]?5'b11111:5'b00001;expand_command[104]=fixed[9];
   expand_command[105]=fixed[9]&&(fixed[1:0]==2'b11);
  end
 endfunction
 wire [105:0] decoded_command=expand_command(commands_dout);
 wire [4:0] a_required=decoded_command[104]?decoded_command[103:99]:5'b00001;
 wire [4:0] b_required=decoded_command[105]?5'b00000:decoded_command[103:99];
 wire transpose_space=decoded_command[98]||
  ((&(a_cfg_ready|~a_required))&&(&(b_cfg_ready|~b_required)));
assign slow_a[0*256+:256]=a0_dout; assign slow_a_present[0]=a0_empty_n; assign a0_read=slow_a_read[0];
 assign slow_b[0*256+:256]=b0_dout; assign slow_b_present[0]=b0_empty_n; assign b0_read=slow_b_read[0];
 assign r0_din=slow_result[0*384+:384]; assign slow_result_ready[0]=r0_full_n; assign r0_write=slow_result_write[0];
 assign slow_a[1*256+:256]=a1_dout; assign slow_a_present[1]=a1_empty_n; assign a1_read=slow_a_read[1];
 assign slow_b[1*256+:256]=b1_dout; assign slow_b_present[1]=b1_empty_n; assign b1_read=slow_b_read[1];
 assign r1_din=slow_result[1*384+:384]; assign slow_result_ready[1]=r1_full_n; assign r1_write=slow_result_write[1];
 assign slow_a[2*256+:256]=a2_dout; assign slow_a_present[2]=a2_empty_n; assign a2_read=slow_a_read[2];
 assign slow_b[2*256+:256]=b2_dout; assign slow_b_present[2]=b2_empty_n; assign b2_read=slow_b_read[2];
 assign r2_din=slow_result[2*384+:384]; assign slow_result_ready[2]=r2_full_n; assign r2_write=slow_result_write[2];
 assign slow_a[3*256+:256]=a3_dout; assign slow_a_present[3]=a3_empty_n; assign a3_read=slow_a_read[3];
 assign slow_b[3*256+:256]=b3_dout; assign slow_b_present[3]=b3_empty_n; assign b3_read=slow_b_read[3];
 assign r3_din=slow_result[3*384+:384]; assign slow_result_ready[3]=r3_full_n; assign r3_write=slow_result_write[3];
 assign slow_a[4*256+:256]=a4_dout; assign slow_a_present[4]=a4_empty_n; assign a4_read=slow_a_read[4];
 assign slow_b[4*256+:256]=b4_dout; assign slow_b_present[4]=b4_empty_n; assign b4_read=slow_b_read[4];
 assign r4_din=slow_result[4*384+:384]; assign slow_result_ready[4]=r4_full_n; assign r4_write=slow_result_write[4];
 wire cmd_space,command_present,core_command_read,core_done,core_idle,perf_space,perf_present;
 wire [9:0] fast_fixed_command;
 wire [105:0] fast_command=expand_command(fast_fixed_command);
 wire [255:0] fast_perf,slow_perf;
 wire command_ok=(commands_dout[4:3]!=2'b11)&&
  (commands_dout[9]?(commands_dout[1:0]==2'b00||commands_dout[1:0]==2'b11):
                    (commands_dout[1:0]==2'b01||commands_dout[1:0]==2'b10))&&
  (commands_dout[1:0]!=2'b11||commands_dout[8]);
 wire [9:0] rounded_m={1'b0,decoded_command[8:0]}+10'd31;
 wire [9:0] rounded_n={1'b0,decoded_command[40:32]}+10'd31;
 wire [9:0] result_n={1'b0,decoded_command[40:32]}+10'd15;
 (* use_dsp="no" *)wire [13:0] new_a=rounded_m[9:5]*{decoded_command[72:64],1'b0};
 (* use_dsp="no" *)wire [13:0] new_b=rounded_n[9:5]*{decoded_command[72:64],1'b0};
 (* use_dsp="no" *)wire [12:0] new_r=decoded_command[8:0]*{result_n[9:4],1'b0};
 assign commands_read=!slow_rst&&ap_ce&&slow_state==RUN&&!stop_seen&&commands_empty_n&&cmd_space&&transpose_space;
 wire budget_add=commands_read&&!decoded_command[98]&&command_ok;
 wire begin_run=ap_ce&&ap_start&&slow_state==IDLE;
 assign ap_idle=slow_state==IDLE;
 assign ap_done=slow_state==DONE;
 assign ap_ready=ap_done&&ap_continue&&ap_ce&&!slow_rst;
 wire all_results_delivered=(r_owed[0]==0)&&(r_owed[1]==0)&&(r_owed[2]==0)&&(r_owed[3]==0)&&(r_owed[4]==0);
 assign perf_write=!slow_rst&&ap_ce&&slow_state==RUN&&stop_seen&&all_results_delivered&&perf_present&&perf_full_n;
 assign perf_din=slow_perf;
 integer i;
 always @(posedge ap_clk)begin
  if(slow_rst)begin
   slow_state<=IDLE;start_toggle<=0;stop_seen<=0;
   for(i=0;i<5;i=i+1)begin a_owed[i]<=0;b_owed[i]<=0;r_owed[i]<=0;end
  end else if(begin_run)begin
   slow_state<=RUN;start_toggle<=!start_toggle;stop_seen<=0;
   for(i=0;i<5;i=i+1)begin a_owed[i]<=0;b_owed[i]<=0;r_owed[i]<=0;end
  end else begin
   for(i=0;i<5;i=i+1)begin
    a_owed[i]<=a_owed[i]+((budget_add&&(decoded_command[104]?decoded_command[99+i]:(i==0)))?new_a:0)-(slow_a_read[i]?1:0);
    b_owed[i]<=b_owed[i]+((budget_add&&b_required[i])?new_b:0)-(slow_b_read[i]?1:0);
    r_owed[i]<=r_owed[i]+((budget_add&&decoded_command[97]&&decoded_command[99+i])?new_r:0)-(slow_result_write[i]?1:0);
   end
   if(commands_read&&decoded_command[98])stop_seen<=1;
   if(perf_write)slow_state<=DONE;
   else if(slow_state==DONE&&ap_continue&&ap_ce)slow_state<=IDLE;
  end
 end
 (* ASYNC_REG="TRUE" *)reg [2:0] start_sync;
 reg start_seen;
 always @(posedge mac_fast_clk)begin
  if(fast_rst)begin start_sync<=0;start_seen<=0;end
  else begin start_sync<={start_sync[1:0],start_toggle};start_seen<=start_sync[2];end
 end
 wire core_start=start_sync[2]^start_seen;
 wire finish_write=core_done&&perf_space&&!fast_rst;
 c209_mac_async_fifo #(.WIDTH(10),.DEPTH(16)) command_cdc(
  .wr_clk(ap_clk),.rd_clk(mac_fast_clk),.rst(slow_rst),.din(commands_dout),.wr_en(commands_read),.full_n(cmd_space),
  .dout(fast_fixed_command),.rd_en(core_command_read),.empty_n(command_present));
 genvar s;
 generate for(s=0;s<5;s=s+1)begin:shard_cdc
  wire a_accept=!slow_rst&&ap_ce&&slow_state==RUN&&a_owed[s]!=0&&slow_a_present[s];
  wire b_accept=!slow_rst&&ap_ce&&slow_state==RUN&&b_owed[s]!=0&&slow_b_present[s];
  wire a_output_ready=!slow_rst&&ap_ce&&slow_state==RUN&&a_space[s];
  wire b_output_ready=!slow_rst&&ap_ce&&slow_state==RUN&&b_space[s];
  assign slow_a_read[s]=a_accept&&a_input_ready[s];
  assign slow_b_read[s]=b_accept&&b_input_ready[s];
  assign slow_result_write[s]=!slow_rst&&ap_ce&&slow_state==RUN&&r_owed[s]!=0&&result_present[s]&&slow_result_ready[s];
  c209_panel_transposer a_transpose(.clk(ap_clk),.rst(slow_rst),
   .cfg_valid(budget_add&&a_required[s]),.cfg_ready(a_cfg_ready[s]),
   .cfg_rows(decoded_command[8:0]),.cfg_k(decoded_command[72:64]),
   .input_data(slow_a[s*256+:256]),.input_valid(a_accept),.input_ready(a_input_ready[s]),
   .output_data(transposed_a[s*256+:256]),.output_valid(a_transposed_valid[s]),.output_ready(a_output_ready));
  c209_panel_transposer b_transpose(.clk(ap_clk),.rst(slow_rst),
   .cfg_valid(budget_add&&b_required[s]),.cfg_ready(b_cfg_ready[s]),
   .cfg_rows(decoded_command[40:32]),.cfg_k(decoded_command[72:64]),
   .input_data(slow_b[s*256+:256]),.input_valid(b_accept),.input_ready(b_input_ready[s]),
   .output_data(transposed_b[s*256+:256]),.output_valid(b_transposed_valid[s]),.output_ready(b_output_ready));
  c209_mac_async_fifo #(.WIDTH(256),.DEPTH(32)) a_cdc(.wr_clk(ap_clk),.rd_clk(mac_fast_clk),.rst(slow_rst),
   .din(transposed_a[s*256+:256]),.wr_en(a_transposed_valid[s]&&a_output_ready),.full_n(a_space[s]),.dout(fast_a[s*256+:256]),.rd_en(core_a_read[s]),.empty_n(fast_a_present[s]));
  c209_mac_async_fifo #(.WIDTH(256),.DEPTH(32)) b_cdc(.wr_clk(ap_clk),.rd_clk(mac_fast_clk),.rst(slow_rst),
   .din(transposed_b[s*256+:256]),.wr_en(b_transposed_valid[s]&&b_output_ready),.full_n(b_space[s]),.dout(fast_b[s*256+:256]),.rd_en(core_b_read[s]),.empty_n(fast_b_present[s]));
  c209_five_result_cdc384 #(.DEPTH(64)) result_cdc(.wr_clk(mac_fast_clk),.rd_clk(ap_clk),.rst(fast_rst),
   .din(fast_result[s*384+:384]),.marker_in(1'b0),.wr_en(core_result_write[s]),.full_n(result_space[s]),
   .dout(slow_result[s*384+:384]),.marker_out(),.empty_n(result_present[s]),.rd_en(slow_result_write[s]));
 end endgenerate
 c209_mac_async_fifo #(.WIDTH(256),.DEPTH(16)) perf_cdc(.wr_clk(mac_fast_clk),.rd_clk(ap_clk),.rst(fast_rst),
  .din(fast_perf),.wr_en(finish_write),.full_n(perf_space),.dout(slow_perf),.rd_en(perf_write),.empty_n(perf_present));
 c209_five_mac_core core(.clk(mac_fast_clk),.rst(fast_rst),.start(core_start),.continue_run(finish_write),.done(core_done),.idle(core_idle),
  .commands_dout(fast_command),.commands_empty_n(command_present),.commands_read(core_command_read),
  .a_dout(fast_a),.b_dout(fast_b),.a_empty_n(fast_a_present),.b_empty_n(fast_b_present),.a_read(core_a_read),.b_read(core_b_read),
  .results_din(fast_result),.results_full_n(result_space),.results_write(core_result_write),.perf_snapshot(fast_perf));
 // synthesis translate_off
 always @(posedge ap_clk)if(!slow_rst)begin
  if(commands_read&&!decoded_command[98]&&!command_ok)$fatal(1,"FIVE_SLOW_INVALID_COMMAND");
  if(perf_write&&(!all_results_delivered||!stop_seen))$fatal(1,"FIVE_EARLY_STOP_PERF");
 end
 // synthesis translate_on
endmodule
