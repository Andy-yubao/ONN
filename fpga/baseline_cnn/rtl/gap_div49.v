// gap_div49.v - Integer GAP: q = saturate_uint8(round_half_away_from_zero(sum/49)).
//
// Frozen semantics (Int8Reference, model/onn_model/int8_reference.py):
//   sum is a non-negative 14-bit integer in [0, 12495] (49 * 255, i.e. the
//   per-channel sum of UINT8 conv3_q over a 7x7 patch).  For a non-negative
//   numerator round-half-away-from-zero is round-half-up, i.e.
//       q = floor((sum + 24) / 49)                (24 = 49 >> 1)
//   saturated to UINT8 [0, 255].
//
// Exact constant division by 49, NOT a multiply-by-reciprocal approximation
// and NOT a divider IP.  The magic pair  q = (n * 2675) >> 17  (M = 0xA73,
// S = 17) was verified exhaustively over n in [0, 12519]; it equals floor
// division by 49 for every input in range.  n = sum + 24 in [24, 12519] fits
// 14 bits and (n * 2675) fits 25 bits, so the multiply cannot overflow.
//
// Pure combinational, no latches.  Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module gap_div49 (
    input      [13:0] sum,
    output     [7:0]  q
);
    // n = sum + 24, in [24, 12519] -> 14-bit unsigned
    wire [13:0] n = sum + 14'd24;

    // exact constant division: floor(n / 49) = (n * 2675) >> 17.
    // {11'd0, n} widens the multiplicand so the product is computed in 25 bits.
    wire [24:0] shifted = ({11'd0, n} * 25'd2675) >> 17;   // max 255
    wire [7:0]  quot    = shifted[7:0];

    // UINT8 saturation guard (sum <= 12495 can only reach 255, kept anyway).
    assign q = (quot > 8'd255) ? 8'd255 : quot;
endmodule
