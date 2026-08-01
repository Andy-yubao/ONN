// conv23_smoke_top.v - Quartus smoke wrapper for the shared conv2/conv3 engine.
//
// TOP_LEVEL_ENTITY of the conv23_smoke project.  Compiles conv_u8_serial (both
// conv2/conv3 weight ROMs, both bias ROMs, requantize_u8) + the sync templates
// + an external feature-map RAM (3136 x 8, the conv2 input plane; conv3 reads
// the same RAM but only addresses 0..1567) on the frozen target EP4CE10F17C8 to
// prove it fits and is clean (no latches / truncation / signed surprises, no
// division/modulo inferred).
//
// Every port -- including the smoke-only virtual clock `clk` -- is assigned
// VIRTUAL_PIN in the QSF, so NO physical pin is consumed (AC620 pinout is not
// frozen).  All top-level ports are registered one stage so the Fitter never
// sees a register-less virtual pin (Error 171016).
//
// The engine debug streams are COMPRESSED at this boundary: acc_valid/q_valid
// pass through, plus dbg_qcnt (q_valid pulses mod 256) and dbg_qxor (xor of
// every q_value).  layer_sel is a registered input so the smoke can exercise
// both layers.  The external feature-map RAM is written from the virtual pins
// (fm_we/fm_waddr/fm_wdata) and read by the engine via fm_raddr/fm_rdata.
//
// `clk` / `rst_n` are TEST-ONLY virtual signals, not board clock/reset.  This
// wrapper is NOT the final board-level top.
`timescale 1ns/1ps

module conv23_smoke_top #(
    // Quartus resolves $readmemh relative paths against the project dir; the
    // project lives in fpga/baseline_cnn/quartus so "../params/..." is correct.
    parameter WT2_MEM_FILE   = "../params/weights/conv2_weight.mem",
    parameter BIAS2_MEM_FILE = "../params/biases/conv2_bias.mem",
    parameter WT3_MEM_FILE   = "../params/weights/conv3_weight.mem",
    parameter BIAS3_MEM_FILE = "../params/biases/conv3_bias.mem"
) (
    input  wire        clk,          // virtual smoke clock, no physical pin
    input  wire        rst_n,        // virtual reset, no physical pin

    // control (registered)
    input  wire        start,
    input  wire        layer_sel,    // 0=conv2, 1=conv3
    output wire        busy,
    output wire        done,

    // external feature-map RAM write port (registered)
    input  wire        fm_we,
    input  wire [11:0] fm_waddr,
    input  wire [7:0]  fm_wdata,

    // compressed debug observables (registered)
    output wire        acc_valid,
    output wire        q_valid,
    output wire [7:0]  dbg_qcnt,     // q_valid pulses mod 256
    output wire [7:0]  dbg_qxor      // xor of q_value
);
    // ---- registered inputs ----
    reg        rst_n_r;
    reg        start_r;
    reg        layer_sel_r;
    reg        fm_we_r;
    reg [11:0] fm_waddr_r;
    reg [7:0]  fm_wdata_r;

    // ---- engine / RAM connections ----
    wire busy_w, done_w, acc_valid_w, q_valid_w;
    wire [11:0] fm_raddr_w;
    wire [7:0]  fm_rdata_w;
    wire [7:0]  q_value_w;

    // external feature-map RAM (read port = engine fm_raddr)
    sync_ram_u8 #(.DEPTH(3136), .ADDR_W(12)) u_fm_ram (
        .clk   (clk),
        .we    (fm_we_r),
        .waddr (fm_waddr_r),
        .wdata (fm_wdata_r),
        .raddr (fm_raddr_w),
        .rdata (fm_rdata_w)
    );

    conv_u8_serial #(
        .WT2_MEM_FILE(WT2_MEM_FILE),
        .BIAS2_MEM_FILE(BIAS2_MEM_FILE),
        .WT3_MEM_FILE(WT3_MEM_FILE),
        .BIAS3_MEM_FILE(BIAS3_MEM_FILE)
    ) u_engine (
        .clk        (clk),
        .rst_n      (rst_n_r),
        .start      (start_r),
        .layer_sel  (layer_sel_r),
        .busy       (busy_w),
        .done       (done_w),
        .fm_raddr   (fm_raddr_w),
        .fm_rdata   (fm_rdata_w),
        .acc_valid  (acc_valid_w),
        .acc_addr   (),              // debug compressed away
        .acc_value  (),              // debug compressed away
        .q_valid    (q_valid_w),
        .q_addr     (),              // debug compressed away
        .q_value    (q_value_w)
    );

    // ---- compressed debug ----
    reg [7:0] dbg_qcnt_r;
    reg [7:0] dbg_qxor_r;
    always @(posedge clk or negedge rst_n_r) begin
        if (!rst_n_r) begin
            dbg_qcnt_r <= 8'd0;
            dbg_qxor_r <= 8'd0;
        end else begin
            if (q_valid_w) begin
                dbg_qcnt_r <= dbg_qcnt_r + 8'd1;
                dbg_qxor_r <= dbg_qxor_r ^ q_value_w;
            end
        end
    end

    // ---- register boundary for every top-level port ----
    always @(posedge clk) begin
        rst_n_r     <= rst_n;
        start_r     <= start;
        layer_sel_r <= layer_sel;
        fm_we_r     <= fm_we;
        fm_waddr_r  <= fm_waddr;
        fm_wdata_r  <= fm_wdata;
    end

    reg busy_r, done_r, acc_valid_r, q_valid_r;
    always @(posedge clk) begin
        busy_r      <= busy_w;
        done_r      <= done_w;
        acc_valid_r <= acc_valid_w;
        q_valid_r   <= q_valid_w;
    end

    assign busy      = busy_r;
    assign done      = done_r;
    assign acc_valid = acc_valid_r;
    assign q_valid   = q_valid_r;
    assign dbg_qcnt  = dbg_qcnt_r;
    assign dbg_qxor  = dbg_qxor_r;
endmodule
