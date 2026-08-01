// requantize_u8.v - Fixed-point requantisation to UINT8 (scheme A, frozen).
//
// Frozen semantics (Int8Reference, model/onn_model/int8_reference.py):
//     product = signed(acc) * signed(multiplier)        (signed INT64)
//     rounded = round_half_away_from_zero(product / 2^shift)
//     q       = saturate(rounded, 0, 255)               (UINT8)
//
// Rounding is computed on the magnitude (add 2^(shift-1), right shift,
// restore sign) so negative products never rely on Verilog's arithmetic
// shift (>>>), which rounds toward -inf.  shift = 0 is an identity (no
// rounding term, no shift), matching the reference round_shift().
//
// Product magnitude bound: |acc| <= 2^31-1, |multiplier| < 2^31  =>  the
// worst-case |mag + half| (shift = 63) is < 2^63 and still fits the 64-bit
// magnitude bus, so no overflow is possible over the full [0,63] shift range.
//
// Pure combinational, every output assigned in all paths (no latches).
// Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module requantize_u8 (
    input  signed [31:0] acc,
    input  signed [31:0] multiplier,
    input         [5:0]  shift,
    output        [7:0]  q
);
    // ---- sign-extend to 64-bit, then signed multiply (64x64 -> 64) ----
    wire signed [63:0] acc64   = $signed(acc);
    wire signed [63:0] mult64  = $signed(multiplier);
    wire signed [63:0] product = acc64 * mult64;

    // ---- magnitude of product (two's complement negation in 64-bit
    //      unsigned space; |product| < 2^63 always, so no negation overflow) ----
    wire [63:0] mag = product[63] ? (~product[63:0] + 64'd1) : product[63:0];

    // ---- round-half-away-from-zero: add 2^(shift-1), then logical right shift ----
    wire [63:0] half      = (shift == 6'd0) ? 64'd0 : (64'd1 << (shift - 6'd1));
    wire [63:0] round_mag = mag + half;
    wire [63:0] shifted   = round_mag >> shift;

    // ---- restore sign, then saturate to UINT8 [0,255] ----
    wire signed [63:0] signed_shifted = $signed(shifted);
    wire signed [63:0] value = product[63] ? -signed_shifted : signed_shifted;

    assign q = (value > 64'sd255) ? 8'd255 :
               (value < 64'sd0)   ? 8'd0   :
               value[7:0];
endmodule
