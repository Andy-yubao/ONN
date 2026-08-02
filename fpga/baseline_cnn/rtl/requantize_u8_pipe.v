// requantize_u8_pipe.v - Three-stage pipelined scheme-A requantisation.
//
// Pipeline boundaries (one token may be accepted on every rising edge):
//   1. SAT32            : S64 post-bias accumulator -> saturated S32,
//                         latch the token's S32 multiplier
//   2. MUL              : explicit signed S32 x S32 -> S64 product
//   3. ROUND_SHIFT_SAT  : magnitude-based round-half-away-from-zero,
//                         fixed logical shift, restore sign, UINT8 saturation
//
// The original combinational requantize_u8.v remains unchanged as the
// bit-exact oracle.  SHIFT is a compile-time parameter, never a run-time
// input.  No vendor IP.
`timescale 1ns/1ps

module requantize_u8_pipe #(
    parameter [5:0] SHIFT = 6'd38
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               in_valid,
    input  wire signed [63:0] in_acc64,
    input  wire signed [31:0] in_multiplier,

    output wire               out_valid,
    output wire signed [31:0] out_acc32,
    output wire        [7:0]  out_q
);
    localparam signed [63:0] INT32_MAX =  64'sd2147483647;
    localparam signed [63:0] INT32_MIN = -64'sd2147483648;

    // Stage 1: SAT32 and token-local multiplier capture.
    wire signed [31:0] sat32_w = (in_acc64 > INT32_MAX) ? 32'sh7FFFFFFF :
                                 (in_acc64 < INT32_MIN) ? 32'sh80000000 :
                                 in_acc64[31:0];
    reg                sat_valid_r;
    reg signed [31:0]  sat_acc32_r;
    reg signed [31:0]  sat_multiplier_r;

    // Stage 2: the only requant multiplier in this module.  Both operands are
    // explicitly 32-bit signed; the result register is signed S64.
    reg                mul_valid_r;
    reg signed [63:0]  mul_product_r;
    reg signed [31:0]  mul_acc32_r;

    // Stage 3: registered q and the matching acc32 token.
    reg                round_valid_r;
    reg signed [31:0]  round_acc32_r;
    reg        [7:0]   round_q_r;

    // Magnitude-based rounding, exactly matching requantize_u8.v.  The
    // generate keeps SHIFT=0 free of an invalid (SHIFT-1) expression and makes
    // the shift a compile-time constant in both branches.
    wire [63:0] product_mag = mul_product_r[63] ?
                              (~mul_product_r[63:0] + 64'd1) :
                              mul_product_r[63:0];
    wire [63:0] shifted_mag;
    generate
        if (SHIFT == 6'd0) begin : gen_shift_zero
            assign shifted_mag = product_mag;
        end else begin : gen_shift_nonzero
            localparam [63:0] ROUND_HALF = 64'd1 << (SHIFT - 6'd1);
            wire [63:0] rounded_mag = product_mag + ROUND_HALF;
            assign shifted_mag = rounded_mag >> SHIFT;
        end
    endgenerate

    wire signed [63:0] shifted_signed = $signed(shifted_mag);
    wire signed [63:0] rounded_value = mul_product_r[63] ?
                                       -shifted_signed : shifted_signed;
    wire [7:0] round_q_w = (rounded_value > 64'sd255) ? 8'd255 :
                           (rounded_value < 64'sd0)   ? 8'd0   :
                           rounded_value[7:0];

    assign out_valid = round_valid_r;
    assign out_acc32 = round_acc32_r;
    assign out_q     = round_q_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sat_valid_r      <= 1'b0;
            mul_valid_r      <= 1'b0;
            round_valid_r    <= 1'b0;
            sat_acc32_r      <= 32'sd0;
            sat_multiplier_r <= 32'sd0;
            mul_product_r    <= 64'sd0;
            mul_acc32_r      <= 32'sd0;
            round_acc32_r    <= 32'sd0;
            round_q_r        <= 8'd0;
        end else begin
            sat_valid_r   <= in_valid;
            mul_valid_r   <= sat_valid_r;
            round_valid_r <= mul_valid_r;

            if (in_valid) begin
                sat_acc32_r      <= sat32_w;
                sat_multiplier_r <= in_multiplier;
            end
            if (sat_valid_r) begin
                mul_product_r <= $signed(sat_acc32_r) * $signed(sat_multiplier_r);
                mul_acc32_r   <= sat_acc32_r;
            end
            if (mul_valid_r) begin
                round_acc32_r <= mul_acc32_r;
                round_q_r     <= round_q_w;
            end
        end
    end
endmodule
