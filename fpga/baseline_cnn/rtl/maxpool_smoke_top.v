// maxpool_smoke_top.v - Quartus smoke wrapper for the streaming maxpool.
//
// TOP_LEVEL_ENTITY of the maxpool_smoke project.  Compiles the streaming 2x2
// max-pool primitive (maxpool2x2_stream.v) on the frozen target
// EP4CE10F17C8 to prove it fits and is clean (no latches / truncation /
// signed surprises, no inferred multipliers or dividers).
//
// Every port -- including the smoke-only virtual clock `clk` -- is assigned
// VIRTUAL_PIN in the QSF, so NO physical pin is consumed (AC620 pinout is not
// frozen).  All top-level ports are registered one stage so the Fitter never
// sees a register-less virtual pin (Error 171016).
//
// Debug observables: dbg_incnt / dbg_outcnt (16-bit each) count the accepted
// inputs and produced outputs, so a full run leaves incnt=12544 and
// outcnt=3136 - cheap proof of completion through the virtual pins.
//
// `clk` / `rst_n` are TEST-ONLY virtual signals, not board clock/reset.  This
// wrapper is NOT the final board-level top (no UART, no board pins); it exists
// to report maxpool-layer resource usage on the real device.  The maxpool is
// pure logic - no RAM/ROM, no vendor IP.
`timescale 1ns/1ps

module maxpool_smoke_top (
    input  wire        clk,          // virtual smoke clock, no physical pin
    input  wire        rst_n,        // virtual reset, no physical pin

    // stream control (registered)
    input  wire        start,
    input  wire        in_valid,
    input  wire [7:0]  in_q,

    // results (registered)
    output wire        busy,
    output wire        out_valid,
    output wire [11:0] out_addr,
    output wire [7:0]  out_q,
    output wire        done,

    // debug observables (registered)
    output wire [15:0] dbg_incnt,    // accepted inputs (12544 after a run)
    output wire [15:0] dbg_outcnt    // produced outputs (3136 after a run)
);
    // ---- registered inputs ----
    reg       rst_n_r;
    reg       start_r;
    reg       in_valid_r;
    reg [7:0] in_q_r;

    // ---- engine connections ----
    wire busy_w, out_valid_w, done_w;
    wire [11:0] out_addr_w;
    wire [7:0]  out_q_w;

    maxpool2x2_stream u_engine (
        .clk       (clk),
        .rst_n     (rst_n_r),
        .start     (start_r),
        .in_valid  (in_valid_r),
        .in_q      (in_q_r),
        .busy      (busy_w),
        .out_valid (out_valid_w),
        .out_addr  (out_addr_w),
        .out_q     (out_q_w),
        .done      (done_w)
    );

    // ---- debug counters ----
    reg [15:0] incnt_r;
    reg [15:0] outcnt_r;
    always @(posedge clk or negedge rst_n_r) begin
        if (!rst_n_r) begin
            incnt_r  <= 16'd0;
            outcnt_r <= 16'd0;
        end else begin
            if (busy_w && in_valid_r) incnt_r  <= incnt_r + 16'd1;
            if (out_valid_w)          outcnt_r <= outcnt_r + 16'd1;
        end
    end

    // ---- register boundary for every top-level input ----
    always @(posedge clk) begin
        rst_n_r    <= rst_n;
        start_r    <= start;
        in_valid_r <= in_valid;
        in_q_r     <= in_q;
    end

    // ---- register boundary for every top-level output ----
    reg busy_r, out_valid_r, done_r;
    reg [11:0] out_addr_r;
    reg [7:0]  out_q_r;
    reg [15:0] incnt_r2, outcnt_r2;
    always @(posedge clk) begin
        busy_r      <= busy_w;
        out_valid_r <= out_valid_w;
        out_addr_r  <= out_addr_w;
        out_q_r     <= out_q_w;
        done_r      <= done_w;
        incnt_r2    <= incnt_r;
        outcnt_r2   <= outcnt_r;
    end

    assign busy         = busy_r;
    assign out_valid    = out_valid_r;
    assign out_addr     = out_addr_r;
    assign out_q        = out_q_r;
    assign done         = done_r;
    assign dbg_incnt    = incnt_r2;
    assign dbg_outcnt   = outcnt_r2;
endmodule
