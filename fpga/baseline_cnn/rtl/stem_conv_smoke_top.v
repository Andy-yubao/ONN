// stem_conv_smoke_top.v - Quartus smoke wrapper for the stem engine.
//
// TOP_LEVEL_ENTITY of the stem_conv_smoke project.  Compiles the FULL serial
// stem engine (stem_conv_serial + sync_ram_u8 + sync_rom_s8 + sync_rom_s32 +
// requantize_u8) on the frozen target EP4CE10F17C8 to prove it fits and is
// clean (no latches / truncation / signed surprises).
//
// Every port -- including the smoke-only virtual clock `clk` -- is assigned
// VIRTUAL_PIN in the QSF, so NO physical pin is consumed (AC620 pinout is not
// frozen).  All top-level ports are registered one stage so the Fitter never
// sees a register-less virtual pin (Error 171016).
//
// The debug streams of the engine are COMPRESSED at this boundary: instead of
// exposing acc_addr/acc_value/q_addr/q_value as ~70 pins, only acc_valid and
// q_valid are passed through, plus two compact 8-bit indicators:
//   dbg_qcnt = number of q_valid pulses (mod 256) -- proves the run completes
//   dbg_qxor = xor of every q_value -- a cheap observable of the requant stream
// This keeps the virtual-pin count down while still exercising the real engine.
//
// `clk` / `rst_n` are TEST-ONLY virtual signals, not board clock/reset.  This
// wrapper is NOT the final board-level top (no UART, no board pins); it exists
// to report stem-layer resource usage on the real device.
`timescale 1ns/1ps

module stem_conv_smoke_top #(
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
    output wire        acc_valid,
    output wire        q_valid,
    output wire [7:0]  dbg_qcnt,     // q_valid pulses mod 256
    output wire [7:0]  dbg_qxor,     // xor of q_value

    // output feature-map RAM read port (registered)
    input  wire [13:0] output_raddr,
    output wire [7:0]  output_rdata
);
    // ---- registered inputs ----
    reg        rst_n_r;
    reg        input_we_r;
    reg [9:0]  input_waddr_r;
    reg signed [7:0] input_wdata_r;
    reg        start_r;
    reg [13:0] output_raddr_r;

    // ---- engine connections ----
    wire busy_w, done_w, acc_valid_w, q_valid_w;
    wire [7:0] q_value_w;
    wire [7:0] output_rdata_w;

    stem_conv_serial #(
        .WEIGHT_MEM_FILE(WEIGHT_MEM_FILE),
        .BIAS_MEM_FILE  (BIAS_MEM_FILE)
    ) u_engine (
        .clk          (clk),
        .rst_n        (rst_n_r),
        .input_we     (input_we_r),
        .input_waddr  (input_waddr_r),
        .input_wdata  (input_wdata_r),
        .start        (start_r),
        .busy         (busy_w),
        .done         (done_w),
        .acc_valid    (acc_valid_w),
        .acc_addr     (),            // debug compressed away
        .acc_value    (),            // debug compressed away
        .q_valid      (q_valid_w),
        .q_addr       (),            // debug compressed away
        .q_value      (q_value_w),
        .output_raddr (output_raddr_r),
        .output_rdata (output_rdata_w)
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
        rst_n_r       <= rst_n;
        input_we_r    <= input_we;
        input_waddr_r <= input_waddr;
        input_wdata_r <= input_wdata;
        start_r       <= start;
        output_raddr_r <= output_raddr;
    end

    reg busy_r, done_r, acc_valid_r, q_valid_r;
    reg [7:0] output_rdata_r;
    always @(posedge clk) begin
        busy_r        <= busy_w;
        done_r        <= done_w;
        acc_valid_r   <= acc_valid_w;
        q_valid_r     <= q_valid_w;
        output_rdata_r <= output_rdata_w;
    end

    assign busy         = busy_r;
    assign done         = done_r;
    assign acc_valid    = acc_valid_r;
    assign q_valid      = q_valid_r;
    assign dbg_qcnt     = dbg_qcnt_r;
    assign dbg_qxor     = dbg_qxor_r;
    assign output_rdata = output_rdata_r;
endmodule
