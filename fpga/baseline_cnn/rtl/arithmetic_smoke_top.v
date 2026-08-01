// arithmetic_smoke_top.v - Smoke-test top for the two arithmetic modules.
//
// NOT the final board-level top.  Its only purpose is to prove that
// requantize_u8 and gap_div49 synthesise and fit on the real device
// (EP4CE10F17C8, Cyclone IV E) through the complete Quartus compile flow.
//
// Every port is assigned VIRTUAL_PIN in the Quartus QSF so no physical pin
// is consumed while the AC620 board pinout is still unfrozen.  There is
// intentionally no clock and no UART at this stage.
//
// When the real CNN top is built, this file (and its VIRTUAL_PIN
// assignments in baseline_cnn_smoke.qsf) will be deleted or replaced.
`timescale 1ns/1ps

module arithmetic_smoke_top (
    input  signed [31:0] acc,
    input  signed [31:0] multiplier,
    input         [5:0]  shift,
    input         [13:0] gap_sum,
    output        [7:0]  requant_q,
    output        [7:0]  gap_q
);
    requantize_u8 u_requant (
        .acc        (acc),
        .multiplier (multiplier),
        .shift      (shift),
        .q          (requant_q)
    );

    gap_div49 u_gap (
        .sum (gap_sum),
        .q   (gap_q)
    );
endmodule
