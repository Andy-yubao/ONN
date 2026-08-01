// stem_pool1_pipeline.v - Phase-2 integration core: stem conv -> maxpool -> pool1 RAM.
//
// Frozen integration topology (docs/rtl_microarchitecture.md §10):
//
//     input_q RAM
//     -> stem_conv_serial  (STORE_OUTPUT_RAM=0: no full stem output RAM)
//     -> q_valid / q_value stream
//     -> maxpool2x2_stream (streaming 2x2 stride-2 max pool)
//     -> pool1_q RAM       (3136 x UINT8, CHW, oc -> pool_y -> pool_x)
//
// Start semantics (frozen): ONE `start` pulse launches stem and maxpool together
// (same sampled edge).  The maxpool must already be in RUN when the first
// stem_q_valid arrives; the current interface guarantees start and the first
// in_valid are NEVER on the same sampling edge, because the stem's first output
// is ~12 cycles after start (PROLOGUE + 9xACC + ADD_BIAS + REQ).  stem_done must
// NOT be used to start the maxpool - it fires only after ALL stem_q have been
// emitted.
//
// busy semantics (frozen): `busy` is a STATUS signal, NOT backpressure.  There
// is no ready/stall; the maxpool cannot pause the stem.  No backpressure is
// needed because the maxpool accepts one input per cycle while the stem emits
// one output every 12 cycles.  busy = stem_busy || maxpool_busy.
//
// done semantics (frozen): completion is `maxpool_done` (last pooled result has
// been written to pool1 RAM).  stem_done fires on the same cycle (the stem's
// S_DONE follows the last S_REQ exactly like the maxpool's S_DONE), so both can
// be observed together, but the pipeline-level `done` is maxpool_done.
//
// Feature-map storage: only pool1_q (16x14x14 = 3136 x UINT8) is stored here.
// The complete 12544x8 stem output RAM of phase 1 is NOT instantiated
// (STORE_OUTPUT_RAM=0) and does NOT appear in the integrated Quartus project.
//
// Verilog-2001, no vendor IP.  The synchronous RAMs keep their one-cycle read
// latency contract (see sync_ram_u8.v).
`timescale 1ns/1ps

module stem_pool1_pipeline #(
    // Relative init paths, resolved from the Questa working directory (repo
    // root).  The Quartus smoke wrapper overrides them with paths that resolve
    // from the project directory.  No absolute paths.
    parameter WEIGHT_MEM_FILE = "fpga/baseline_cnn/params/weights/stem_weight.mem",
    parameter BIAS_MEM_FILE   = "fpga/baseline_cnn/params/biases/stem_bias.mem"
) (
    // ---- clock / reset ----
    input  wire              clk,
    input  wire              rst_n,

    // ---- input RAM write port (PC/UART path in the future) ----
    input  wire              input_we,
    input  wire [9:0]        input_waddr,
    input  wire signed [7:0] input_wdata,

    // ---- control ----
    input  wire              start,   // single-cycle pulse; launches stem + maxpool together
    output wire              busy,    // stem_busy || maxpool_busy (status, not backpressure)
    output wire              done,    // maxpool_done (completion signal)

    // ---- pool1 stream (one pulse per pooled result, CHW 0..3135) ----
    output wire              pool_valid,
    output wire [11:0]       pool_addr,
    output wire [7:0]        pool_value,

    // ---- pool1 RAM read port (readback after done) ----
    input  wire [11:0]       pool_raddr,
    output wire [7:0]        pool_rdata,

    // ---- observability (testbench / smoke debug only; not required for operation) ----
    output wire              stem_busy,
    output wire              maxpool_busy,
    output wire              stem_done,
    output wire              maxpool_done,
    output wire              stem_q_valid,
    output wire [7:0]        stem_q_value
);
    // ================= frozen geometry =================
    localparam POOL1_DEPTH  = 3136;      // 16*14*14
    localparam POOL1_ADDR_W = 12;

    // ================= stem engine (no full output RAM) =================
    wire              stem_busy_w;
    wire              stem_done_w;
    wire              stem_q_valid_w;
    wire       [13:0] stem_q_addr_w;    // CHW 0..12543 (kept for observability)
    wire        [7:0] stem_q_value_w;

    stem_conv_serial #(
        .WEIGHT_MEM_FILE (WEIGHT_MEM_FILE),
        .BIAS_MEM_FILE   (BIAS_MEM_FILE),
        .STORE_OUTPUT_RAM(0)            // integration: no complete stem output RAM
    ) u_stem (
        .clk           (clk),
        .rst_n         (rst_n),
        .input_we      (input_we),
        .input_waddr   (input_waddr),
        .input_wdata   (input_wdata),
        .start         (start),
        .busy          (stem_busy_w),
        .done          (stem_done_w),
        .acc_valid     (),              // debug streams not routed out of the pipeline
        .acc_addr      (),
        .acc_value     (),
        .q_valid       (stem_q_valid_w),
        .q_addr        (stem_q_addr_w),
        .q_value       (stem_q_value_w),
        .output_raddr  (14'd0),         // unused with STORE_OUTPUT_RAM=0
        .output_rdata  ()               // unused with STORE_OUTPUT_RAM=0
    );

    // ================= streaming max pool =================
    wire              mp_busy_w;
    wire              mp_out_valid_w;
    wire       [11:0] mp_out_addr_w;
    wire        [7:0] mp_out_q_w;
    wire              mp_done_w;

    maxpool2x2_stream u_pool (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .in_valid   (stem_q_valid_w),   // stem q stream -> pool input
        .in_q       (stem_q_value_w),
        .busy       (mp_busy_w),
        .out_valid  (mp_out_valid_w),
        .out_addr   (mp_out_addr_w),
        .out_q      (mp_out_q_w),
        .done       (mp_done_w)
    );

    // ================= pool1 RAM (3136 x UINT8, the only stored feature map) =================
    // The maxpool outputs are registered (one-cycle), so pool_valid/pool_addr/
    // pool_value are stable for the whole cycle; the RAM write is sampled at the
    // posedge.  The LAST pool1 value is written at the same rising edge where
    // done goes high (the maxpool advertises its final result during S_DONE and
    // `done <= 1'b1` fires on the very edge the RAM samples we=1).  Once that
    // edge has passed the RAM contents are valid, so a post-done readback is
    // safe and sees all 3136 values.
    sync_ram_u8 #(.DEPTH(POOL1_DEPTH), .ADDR_W(POOL1_ADDR_W)) u_pool1_ram (
        .clk   (clk),
        .we    (mp_out_valid_w),
        .waddr (mp_out_addr_w),
        .wdata (mp_out_q_w),
        .raddr (pool_raddr),
        .rdata (pool_rdata)
    );

    // ================= top-level outputs =================
    assign busy          = stem_busy_w || mp_busy_w;
    assign done          = mp_done_w;            // completion = maxpool done
    assign pool_valid    = mp_out_valid_w;
    assign pool_addr     = mp_out_addr_w;
    assign pool_value    = mp_out_q_w;

    assign stem_busy     = stem_busy_w;
    assign maxpool_busy  = mp_busy_w;
    assign stem_done     = stem_done_w;
    assign maxpool_done  = mp_done_w;
    assign stem_q_valid  = stem_q_valid_w;
    assign stem_q_value  = stem_q_value_w;
endmodule
