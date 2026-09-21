`timescale 1ns/1ps

// Signed two's-complement clamp. IN_BITS must be no smaller than OUT_BITS.
module onn_signed_saturate #(
    parameter integer IN_BITS  = 19,
    parameter integer OUT_BITS = 18
) (
    input  logic signed [IN_BITS-1:0]  value_in,
    output logic signed [OUT_BITS-1:0] value_out
);
    localparam logic signed [IN_BITS-1:0] MAX_VALUE =
        (64'sd1 <<< (OUT_BITS - 1)) - 1;
    localparam logic signed [IN_BITS-1:0] MIN_VALUE =
        -(64'sd1 <<< (OUT_BITS - 1));

    initial begin
        if (IN_BITS < OUT_BITS) begin
            $error("onn_signed_saturate requires IN_BITS >= OUT_BITS");
        end
    end

    always_comb begin
        if ($signed(value_in) > $signed(MAX_VALUE)) begin
            value_out = MAX_VALUE[OUT_BITS-1:0];
        end else if ($signed(value_in) < $signed(MIN_VALUE)) begin
            value_out = MIN_VALUE[OUT_BITS-1:0];
        end else begin
            value_out = value_in[OUT_BITS-1:0];
        end
    end
endmodule
