// stem_pool1_smoke_top.v - Quartus smoke wrapper for the phase-2 integration.
//
// TOP_LEVEL_ENTITY of the stem_pool1_smoke project.  Compiles the full
// integration core (stem_conv_serial STORE_OUTPUT_RAM=0 + requantize_u8 +
// maxpool2x2_stream + pool1 RAM, via stem_pool1_pipeline) on the frozen target
// EP4CE10F17C8 to prove it fits and is clean (no latches / truncation / signed
// surprises), and to report the REAL memory map:
//   input RAM 1 M9K + stem weight ROM 1 M9K + pool1 RAM N M9K (Fitter) +
//   bias ROM in logic; the complete 12544x8 stem output RAM must NOT appear.
//
// Every port -- including the smoke-only virtual clock `clk` -- is assigned
// VIRTUAL_PIN in the QSF, so NO physical pin is consumed (AC620 pinout is not
// frozen).  All top-level ports are registered one stage so the Fitter never
// sees a register-less virtual pin (Error 171016).
//
// Debug observables (compressed to avoid a virtual-pin explosion):
//   dbg_qcnt = number of stem q_valid pulses (12544 after a run)   [16-bit]
//   dbg_pcnt = number of pool_valid pulses (3136 after a run)      [16-bit]
//   dbg_pxor = xor of every pool_value (cheap observable)          [8-bit]
// The pipeline's stem/maxpool busy & done split is intentionally NOT routed
// out; `done` above is the pipeline completion (= maxpool_done).
//
// `clk` / `rst_n` are TEST-ONLY virtual signals, not board clock/reset.  This
// wrapper is NOT the final board-level top (no UART, no board pins); it exists
// to report the integrated stem+pool1 resource usage on the real device.
`timescale 1ns/1ps

module stem_pool1_smoke_top #(
    // Quartus resolves $readmemh relative paths against the project dir or the
    // RTL dir; "../params/..." is correct for both when the project lives in
    // fpga/baseline_cnn/quartus.
    parameter WEIGHT_MEM_FILE = "../params/weights/stem_weight.mem",
    parameter BIAS_MEM_FILE   = "../params/biases/stem_bias.mem"
) (
    input  wire        clk,          // virtual smoke clock, no physical pin
    input  wire        rst_n,        // virtual reset, no physical pin

    // input RAM write port (registered)
    input  wire        input_we,
    input  wire [9:0]  input_waddr,
    input  wire signed [7:0] input_wdata,

    // control (registered)
    input  wire        start,
    output wire        busy,
    output wire        done,

    // compressed debug observables (registered)
    output wire [15:0] dbg_qcnt,     // stem q_valid pulses (12544 after a run)
    output wire [15:0] dbg_pcnt,     // pool_valid pulses (3136 after a run)
    output wire [7:0]  dbg_pxor,     // xor of pool_value

    // pool1 RAM read port (registered)
    input  wire [11:0] pool_raddr,
    output wire [7:0]  pool_rdata
);
    // ---- registered inputs ----
    reg        rst_n_r;
    reg        input_we_r;
    reg [9:0]  input_waddr_r;
    reg signed [7:0] input_wdata_r;
    reg        start_r;
    reg [11:0] pool_raddr_r;

    // ---- pipeline connections ----
    wire busy_w, done_w;
    wire pool_valid_w, stem_q_valid_w;
    wire [11:0] pool_addr_w;
    wire [7:0]  pool_value_w, pool_rdata_w;
    wire [7:0]  stem_q_value_w;
    wire        stem_busy_w, maxpool_busy_w, stem_done_w, maxpool_done_w;

    stem_pool1_pipeline #(
        .WEIGHT_MEM_FILE(WEIGHT_MEM_FILE),
        .BIAS_MEM_FILE  (BIAS_MEM_FILE)
    ) u_pipeline (
        .clk          (clk),
        .rst_n        (rst_n_r),
        .input_we     (input_we_r),
        .input_waddr  (input_waddr_r),
        .input_wdata  (input_wdata_r),
        .start        (start_r),
        .busy         (busy_w),
        .done         (done_w),
        .pool_valid   (pool_valid_w),
        .pool_addr    (pool_addr_w),
        .pool_value   (pool_value_w),
        .pool_raddr   (pool_raddr_r),
        .pool_rdata   (pool_rdata_w),
        .stem_busy    (stem_busy_w),
        .maxpool_busy (maxpool_busy_w),
        .stem_done    (stem_done_w),
        .maxpool_done (maxpool_done_w),
        .stem_q_valid (stem_q_valid_w),
        .stem_q_value (stem_q_value_w)
    );

    // ---- compressed debug ----
    reg [15:0] qcnt_r;
    reg [15:0] pcnt_r;
    reg [7:0]  pxor_r;
    always @(posedge clk or negedge rst_n_r) begin
        if (!rst_n_r) begin
            qcnt_r <= 16'd0;
            pcnt_r <= 16'd0;
            pxor_r <= 8'd0;
        end else begin
            if (stem_q_valid_w) qcnt_r <= qcnt_r + 16'd1;
            if (pool_valid_w) begin
                pcnt_r <= pcnt_r + 16'd1;
                pxor_r <= pxor_r ^ pool_value_w;
            end
        end
    end

    // ---- register boundary for every top-level input ----
    always @(posedge clk) begin
        rst_n_r       <= rst_n;
        input_we_r    <= input_we;
        input_waddr_r <= input_waddr;
        input_wdata_r <= input_wdata;
        start_r       <= start;
        pool_raddr_r  <= pool_raddr;
    end

    // ---- register boundary for every top-level output ----
    reg busy_r, done_r;
    reg [15:0] qcnt_r2, pcnt_r2;
    reg [7:0]  pxor_r2, pool_rdata_r;
    always @(posedge clk) begin
        busy_r        <= busy_w;
        done_r        <= done_w;
        qcnt_r2       <= qcnt_r;
        pcnt_r2       <= pcnt_r;
        pxor_r2       <= pxor_r;
        pool_rdata_r  <= pool_rdata_w;
    end

    assign busy         = busy_r;
    assign done         = done_r;
    assign dbg_qcnt     = qcnt_r2;
    assign dbg_pcnt     = pcnt_r2;
    assign dbg_pxor     = pxor_r2;
    assign pool_rdata   = pool_rdata_r;
endmodule
