// baseline_cnn_smoke_top.v - Registered smoke wrapper (TOP_LEVEL_ENTITY).
//
// Quartus's Fitter rejects VIRTUAL_PIN on combinational (register-less)
// ports (Error 171016), so the pure-combinational arithmetic core is wrapped
// in one register stage.  Every port -- including the smoke-only virtual
// clock `clk` -- is assigned VIRTUAL_PIN in the QSF, so NO physical pin is
// consumed anywhere while the AC620 board pinout is still unfrozen.
//
// `clk` is a TEST-ONLY virtual clock, not a board clock: it has no physical
// pin and no timing intent.  This wrapper exists solely to prove that
// requantize_u8 / gap_div49 synthesise and fit on EP4CE10F17C8.  When the
// real CNN top is built, this file (and its VIRTUAL_PIN assignments in
// baseline_cnn_smoke.qsf) will be deleted or replaced; arithmetic_smoke_top
// remains the pure combinational core and is reused unchanged.
`timescale 1ns/1ps

module baseline_cnn_smoke_top (
    input  wire        clk,        // virtual smoke clock, no physical pin
    input  wire signed [31:0] acc,
    input  wire signed [31:0] multiplier,
    input  wire         [5:0]  shift,
    input  wire        [13:0] gap_sum,
    output reg   [7:0]  requant_q,
    output reg   [7:0]  gap_q
);
    reg  signed [31:0] acc_r;
    reg  signed [31:0] multiplier_r;
    reg          [5:0] shift_r;
    reg         [13:0] gap_sum_r;
    wire         [7:0] requant_q_w;
    wire         [7:0] gap_q_w;

    arithmetic_smoke_top u_core (
        .acc        (acc_r),
        .multiplier (multiplier_r),
        .shift      (shift_r),
        .gap_sum    (gap_sum_r),
        .requant_q  (requant_q_w),
        .gap_q      (gap_q_w)
    );

    always @(posedge clk) begin
        acc_r        <= acc;
        multiplier_r <= multiplier;
        shift_r      <= shift;
        gap_sum_r    <= gap_sum;
        requant_q    <= requant_q_w;
        gap_q        <= gap_q_w;
    end
endmodule
