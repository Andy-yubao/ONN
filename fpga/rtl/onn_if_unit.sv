`timescale 1ns/1ps

// Bit-accurate integer IF update used by the frozen deployment model.
module onn_if_unit #(
    parameter integer CURRENT_IN_BITS = 32,
    parameter integer CURRENT_BITS    = 16,
    parameter integer MEM_BITS        = 18,
    parameter integer THRESHOLD       = 939
) (
    input  logic signed [CURRENT_IN_BITS-1:0] current_in,
    input  logic signed [MEM_BITS-1:0]        membrane_in,
    output logic signed [CURRENT_BITS-1:0]    current_saturated,
    output logic signed [MEM_BITS-1:0]        integrated,
    output logic                              spike,
    output logic signed [MEM_BITS-1:0]        membrane_next
);
    logic signed [MEM_BITS:0] sum_extended;
    logic signed [MEM_BITS:0] reset_extended;
    localparam logic signed [MEM_BITS-1:0] THRESHOLD_SIGNED = THRESHOLD;

    initial begin
        if (MEM_BITS < CURRENT_BITS) begin
            $error("onn_if_unit requires MEM_BITS >= CURRENT_BITS");
        end
    end

    onn_signed_saturate #(
        .IN_BITS(CURRENT_IN_BITS),
        .OUT_BITS(CURRENT_BITS)
    ) clamp_current (
        .value_in(current_in),
        .value_out(current_saturated)
    );

    always_comb begin
        sum_extended =
            $signed({membrane_in[MEM_BITS-1], membrane_in}) +
            $signed({{(MEM_BITS + 1 - CURRENT_BITS){current_saturated[CURRENT_BITS-1]}},
                     current_saturated});
    end

    onn_signed_saturate #(
        .IN_BITS(MEM_BITS + 1),
        .OUT_BITS(MEM_BITS)
    ) clamp_integrated (
        .value_in(sum_extended),
        .value_out(integrated)
    );

    always_comb begin
        spike = $signed(integrated) > $signed(THRESHOLD_SIGNED);
        reset_extended = $signed({integrated[MEM_BITS-1], integrated});
        if (spike) begin
            reset_extended = reset_extended - THRESHOLD;
        end
    end

    onn_signed_saturate #(
        .IN_BITS(MEM_BITS + 1),
        .OUT_BITS(MEM_BITS)
    ) clamp_post_reset (
        .value_in(reset_extended),
        .value_out(membrane_next)
    );
endmodule
