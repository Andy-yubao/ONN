// baseline_cnn_core.v - Complete pure-compute BaselineCNN core (no UART).
//
// End-to-end integer pipeline (Int8Reference scheme A, all stages verified):
//
//     input_q RAM -> stem_conv_serial -> stream MaxPool -> pool1_q RAM
//     -> conv_u8_serial (layer_sel=0) -> stream MaxPool -> pool2_q RAM
//     -> conv_u8_serial (layer_sel=1) -> gap_stream_u8 -> gap_mem
//     -> fc_argmax_serial -> prediction
//
// Controller state machine (frozen):
//     IDLE -> STEM_START -> STEM_WAIT -> CONV2_START -> CONV2_WAIT
//          -> CONV3_START -> CONV3_WAIT -> FC_START -> FC_WAIT -> DONE
//
// Startup rules (frozen):
//   - STEM_START asserts stem_start=1 alone; the stem_pool1_pipeline launches
//     its stem engine and the pool1 maxpool TOGETHER (the pipeline owns that
//     start; stem_done must NOT be used to start the maxpool).
//   - CONV2_START asserts conv_start=1 AND pool2_start=1 in the SAME cycle
//     (pool2 is NOT started by conv2_done).  layer_sel=0 latched by the engine.
//   - CONV3_START asserts conv_start=1 AND gap_start=1 in the SAME cycle
//     (GAP is NOT started by conv3_done).  layer_sel=1 latched by the engine.
//   - FC_START asserts fc_start=1 AFTER all 32 gap values have been written
//     into the FC's gap_mem (i.e. after gap_done).
//   - the controller keeps the FM data mux phase stable for the whole conv run
//     (conv2 phase reads pool1 RAM, conv3 phase reads pool2 RAM), so the
//     synchronous RAM return source never switches mid-run.
//
// Feature-map storage (the ONLY stored FM arrays in the whole core):
//     input_q RAM (784, inside the stem engine), pool1_q RAM (3136, inside the
//     pipeline), pool2_q RAM (1568, here), gap_mem (32x8 registers, inside FC).
//     NO complete stem_q / conv2_q / conv3_q RAM is instantiated: stem_q goes
//     straight into the maxpool, conv2_q straight into pool2, conv3_q straight
//     into the GAP.
//
// Core busy/done protocol (frozen):
//   - `start` is accepted ONLY in IDLE; while busy a second start is ignored.
//   - `busy` stays 1 from start through the DONE stage (never a `busy=0 &&
//     done=0` window before the final done); `done` is a single-cycle pulse.
//   - prediction is latched at FC completion and valid at done; it is held
//     until the next start.
//   - after done, no reset is needed: write 784 input bytes (input_we) in IDLE
//     and assert start again; every stage re-arms from its own start.
//
// The input RAM write port is passed straight through to the stem engine; the
// EXTERNAL host must not assert input_we while busy (it would corrupt the
// feature map being consumed).
//
// Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module baseline_cnn_core #(
    // Relative init paths, resolved from the Questa working directory (repo
    // root).  The Quartus smoke wrapper overrides them with paths that resolve
    // from the project directory.  No absolute paths.
    parameter STEM_WT_MEM_FILE  = "fpga/baseline_cnn/params/weights/stem_weight.mem",
    parameter STEM_BIAS_MEM_FILE= "fpga/baseline_cnn/params/biases/stem_bias.mem",
    parameter CONV2_WT_MEM_FILE = "fpga/baseline_cnn/params/weights/conv2_weight.mem",
    parameter CONV2_BIAS_MEM_FILE = "fpga/baseline_cnn/params/biases/conv2_bias.mem",
    parameter CONV3_WT_MEM_FILE = "fpga/baseline_cnn/params/weights/conv3_weight.mem",
    parameter CONV3_BIAS_MEM_FILE = "fpga/baseline_cnn/params/biases/conv3_bias.mem",
    parameter FC_WT_MEM_FILE    = "fpga/baseline_cnn/params/weights/fc_weight.mem",
    parameter FC_BIAS_MEM_FILE  = "fpga/baseline_cnn/params/biases/fc_bias.mem"
) (
    // ---- clock / reset ----
    input  wire              clk,
    input  wire              rst_n,

    // ---- input_q RAM write port (PC/UART path in the future) ----
    input  wire              input_we,
    input  wire [9:0]        input_waddr,
    input  wire signed [7:0] input_wdata,

    // ---- control ----
    input  wire              start,    // accepted only in IDLE
    output wire              busy,     // 1 from start until the final done
    output reg               done,     // single-cycle pulse at end of inference
    output reg  [3:0]        prediction, // signed argmax; valid at done, held

    // ---- debug stream (simulation / golden-trace interfaces only) ----
    output wire [3:0]        dbg_stage,    // controller stage
    output wire              stem_acc_valid,
    output wire [13:0]       stem_acc_addr,
    output wire signed [31:0] stem_acc_value,
    output wire              stem_q_valid,
    output wire [13:0]       stem_q_addr,
    output wire [7:0]        stem_q_value,
    output wire              pool1_valid,
    output wire [11:0]       pool1_addr,
    output wire [7:0]        pool1_value,
    output wire              conv_layer,   // 0=conv2, 1=conv3 (this run)
    output wire              conv_acc_valid,
    output wire [12:0]       conv_acc_addr,
    output wire signed [31:0] conv_acc_value,
    output wire              conv_q_valid,
    output wire [12:0]       conv_q_addr,
    output wire [7:0]        conv_q_value,
    output wire              pool2_valid,
    output wire [10:0]       pool2_addr,
    output wire [7:0]        pool2_value,
    output wire              gap_valid,
    output wire [4:0]        gap_channel,
    output wire [7:0]        gap_value,
    output wire              fc_valid,
    output wire [3:0]        fc_class,
    output wire signed [31:0] fc_acc
);
    // ================= controller states =================
    localparam ST_IDLE        = 4'd0;
    localparam ST_STEM_START  = 4'd1;
    localparam ST_STEM_WAIT   = 4'd2;
    localparam ST_CONV2_START = 4'd3;
    localparam ST_CONV2_WAIT  = 4'd4;
    localparam ST_CONV3_START = 4'd5;
    localparam ST_CONV3_WAIT  = 4'd6;
    localparam ST_FC_START    = 4'd7;
    localparam ST_FC_WAIT     = 4'd8;
    localparam ST_DONE        = 4'd9;

    reg [3:0] state;

    // ================= controller combinational outputs =================
    wire stem_start   = (state == ST_STEM_START);
    wire conv_start   = (state == ST_CONV2_START) || (state == ST_CONV3_START);
    wire conv_layer_sel = (state == ST_CONV3_START) || (state == ST_CONV3_WAIT);
    wire pool2_start  = (state == ST_CONV2_START);
    wire gap_start    = (state == ST_CONV3_START);
    wire fc_start     = (state == ST_FC_START);
    wire conv2_phase  = (state == ST_CONV2_START) || (state == ST_CONV2_WAIT);
    wire conv3_phase  = (state == ST_CONV3_START) || (state == ST_CONV3_WAIT);

    // ================= sub-block wires =================
    // stem + pool1 pipeline
    wire stem_busy_w, stem_done_w;
    wire stem_acc_valid_w;
    wire [13:0]  stem_acc_addr_w;
    wire signed [31:0] stem_acc_value_w;
    wire stem_q_valid_w;
    wire [13:0]  stem_q_addr_w;
    wire [7:0]   stem_q_value_w;
    wire pool1_valid_w;
    wire [11:0]  pool1_addr_w;
    wire [7:0]   pool1_value_w;
    wire [7:0]   pool1_rdata_w;

    // shared conv engine + pool2 maxpool + pool2 RAM
    wire conv_busy_w, conv_done_w;
    wire conv_acc_valid_w;
    wire [12:0] conv_acc_addr_w;
    wire signed [31:0] conv_acc_value_w;
    wire conv_q_valid_w;
    wire [12:0] conv_q_addr_w;
    wire [7:0]  conv_q_value_w;
    wire [11:0] conv_fm_raddr_w;
    wire pool2_busy_w, pool2_done_w;
    wire pool2_out_valid_w;
    wire [10:0] pool2_out_addr_w;
    wire [7:0]  pool2_out_q_w;
    wire [7:0]  pool2_rdata_w;

    // GAP
    wire gap_busy_w, gap_done_w;
    wire gap_valid_w;
    wire [4:0] gap_channel_w;
    wire [7:0] gap_value_w;

    // FC
    wire fc_busy_w, fc_done_w;
    wire fc_valid_w;
    wire [3:0] fc_class_w;
    wire signed [31:0] fc_acc_w;
    wire [3:0] fc_prediction_w;

    // ================= FM RAM address / data gating =================
    // During CONV2 the engine's fm_raddr reads pool1 RAM; during CONV3 it reads
    // pool2 RAM.  Each RAM's read address is forced to 0 outside its phase so
    // the other RAM is never addressed out of range (pool2 has only 1568 cells
    // while the engine's conv2 address space reaches 3135).  The data mux is
    // phase-stable for the whole run.
    // NOTE: these gated read addresses ARE the raddr of the pool1/pool2 RAMs
    // (connected below); the RAM rdata feeds the conv engine's fm_rdata mux.
    wire [11:0] pool1_raddr = conv2_phase ? conv_fm_raddr_w : 12'd0;
    wire [10:0] pool2_raddr = conv3_phase ? conv_fm_raddr_w[10:0] : 11'd0;
    wire [7:0]  conv_fm_rdata = conv3_phase ? pool2_rdata_w : pool1_rdata_w;

    // ================= GAP input gating (conv3 only) =================
    // conv2's q stream must never enter the GAP; only conv3 phase is enabled.
    wire gap_in_valid = conv_q_valid_w && conv3_phase;
    wire [7:0] gap_in_q = conv_q_value_w;

    // ================= stage 1: stem + pool1 =================
    stem_pool1_pipeline #(
        .WEIGHT_MEM_FILE (STEM_WT_MEM_FILE),
        .BIAS_MEM_FILE   (STEM_BIAS_MEM_FILE)
    ) u_stem_pipe (
        .clk            (clk),
        .rst_n          (rst_n),
        .input_we       (input_we),
        .input_waddr    (input_waddr),
        .input_wdata    (input_wdata),
        .start          (stem_start),
        .busy           (stem_busy_w),
        .done           (stem_done_w),
        .pool_valid     (pool1_valid_w),
        .pool_addr      (pool1_addr_w),
        .pool_value     (pool1_value_w),
        .pool_raddr     (pool1_raddr),
        .pool_rdata     (pool1_rdata_w),
        .stem_busy      (),
        .maxpool_busy   (),
        .stem_done      (),
        .maxpool_done   (),
        .stem_q_valid   (stem_q_valid_w),
        .stem_q_value   (stem_q_value_w),
        .stem_acc_valid (stem_acc_valid_w),
        .stem_acc_addr  (stem_acc_addr_w),
        .stem_acc_value (stem_acc_value_w),
        .stem_q_addr    (stem_q_addr_w)
    );

    // ================= stage 2/3: shared conv2/conv3 engine =================
    conv_u8_serial #(
        .WT2_MEM_FILE   (CONV2_WT_MEM_FILE),
        .BIAS2_MEM_FILE (CONV2_BIAS_MEM_FILE),
        .WT3_MEM_FILE   (CONV3_WT_MEM_FILE),
        .BIAS3_MEM_FILE (CONV3_BIAS_MEM_FILE)
    ) u_conv (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (conv_start),
        .layer_sel  (conv_layer_sel),
        .busy       (conv_busy_w),
        .done       (conv_done_w),
        .fm_raddr   (conv_fm_raddr_w),
        .fm_rdata   (conv_fm_rdata),     // phase-stable pool1/pool2 data mux
        .acc_valid  (conv_acc_valid_w),
        .acc_addr   (conv_acc_addr_w),
        .acc_value  (conv_acc_value_w),
        .q_valid    (conv_q_valid_w),
        .q_addr     (conv_q_addr_w),
        .q_value    (conv_q_value_w)
    );

    // ================= pool2 maxpool (parameterized, consumes conv2 q) =================
    maxpool2x2_stream #(
        .N_CH(32), .IN_H(14), .IN_W(14), .OC_W(5), .XY_W(4), .OUT_ADDR_W(11)
    ) u_pool2 (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (pool2_start),
        .in_valid  (conv_q_valid_w),
        .in_q      (conv_q_value_w),
        .busy      (pool2_busy_w),
        .out_valid (pool2_out_valid_w),
        .out_addr  (pool2_out_addr_w),
        .out_q     (pool2_out_q_w),
        .done      (pool2_done_w)
    );

    // ================= pool2 RAM (1568 x UINT8, the only conv-stage FM store) =================
    sync_ram_u8 #(.DEPTH(1568), .ADDR_W(11)) u_pool2_ram (
        .clk   (clk),
        .we    (pool2_out_valid_w),
        .waddr (pool2_out_addr_w),
        .wdata (pool2_out_q_w),
        .raddr (pool2_raddr),
        .rdata (pool2_rdata_w)
    );

    // ================= stage 4: streaming GAP =================
    gap_stream_u8 u_gap (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (gap_start),
        .in_valid    (gap_in_valid),
        .in_q        (gap_in_q),
        .busy        (gap_busy_w),
        .out_valid   (gap_valid_w),
        .out_channel (gap_channel_w),
        .out_q       (gap_value_w),
        .done        (gap_done_w)
    );

    // ================= stage 5: FC + argmax =================
    fc_argmax_serial #(
        .WT_MEM_FILE  (FC_WT_MEM_FILE),
        .BIAS_MEM_FILE(FC_BIAS_MEM_FILE)
    ) u_fc (
        .clk        (clk),
        .rst_n      (rst_n),
        .gap_we     (gap_valid_w),
        .gap_waddr  (gap_channel_w),
        .gap_wdata  (gap_value_w),
        .start      (fc_start),
        .busy       (fc_busy_w),
        .done       (fc_done_w),
        .fc_valid   (fc_valid_w),
        .fc_class   (fc_class_w),
        .fc_acc     (fc_acc_w),
        .prediction (fc_prediction_w)
    );

    // ================= controller FSM =================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= ST_IDLE;
            done       <= 1'b0;
            prediction <= 4'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) state <= ST_STEM_START;
                end
                ST_STEM_START: begin
                    state <= ST_STEM_WAIT;
                end
                ST_STEM_WAIT: begin
                    if (stem_done_w) state <= ST_CONV2_START;
                end
                ST_CONV2_START: begin
                    state <= ST_CONV2_WAIT;
                end
                ST_CONV2_WAIT: begin
                    if (pool2_done_w) state <= ST_CONV3_START;
                end
                ST_CONV3_START: begin
                    state <= ST_CONV3_WAIT;
                end
                ST_CONV3_WAIT: begin
                    if (gap_done_w) state <= ST_FC_START;
                end
                ST_FC_START: begin
                    state <= ST_FC_WAIT;
                end
                ST_FC_WAIT: begin
                    if (fc_done_w) begin
                        prediction <= fc_prediction_w;   // latch at FC completion
                        state <= ST_DONE;
                    end
                end
                ST_DONE: begin
                    done <= 1'b1;      // single-cycle pulse; prediction already valid
                    state <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

    // ================= core-level outputs =================
    assign busy = (state != ST_IDLE);   // covers ST_DONE: holds until the done pulse

    // debug stream passthrough
    assign dbg_stage      = state;
    assign stem_acc_valid = stem_acc_valid_w;
    assign stem_acc_addr  = stem_acc_addr_w;
    assign stem_acc_value = stem_acc_value_w;
    assign stem_q_valid   = stem_q_valid_w;
    assign stem_q_addr    = stem_q_addr_w;
    assign stem_q_value   = stem_q_value_w;
    assign pool1_valid    = pool1_valid_w;
    assign pool1_addr     = pool1_addr_w;
    assign pool1_value    = pool1_value_w;
    assign conv_layer     = conv_layer_sel;
    assign conv_acc_valid = conv_acc_valid_w;
    assign conv_acc_addr  = conv_acc_addr_w;
    assign conv_acc_value = conv_acc_value_w;
    assign conv_q_valid   = conv_q_valid_w;
    assign conv_q_addr    = conv_q_addr_w;
    assign conv_q_value   = conv_q_value_w;
    assign pool2_valid    = pool2_out_valid_w;
    assign pool2_addr     = pool2_out_addr_w;
    assign pool2_value    = pool2_out_q_w;
    assign gap_valid      = gap_valid_w;
    assign gap_channel    = gap_channel_w;
    assign gap_value      = gap_value_w;
    assign fc_valid       = fc_valid_w;
    assign fc_class       = fc_class_w;
    assign fc_acc         = fc_acc_w;
endmodule
