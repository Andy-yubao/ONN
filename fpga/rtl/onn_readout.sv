`timescale 1ns/1ps

// Serial 512->10 readout with four-step weighted accumulation and argmax.
module onn_readout #(
    parameter READOUT_WEIGHT_FILE = "model/snn/export/readout_weight.mem"
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic [511:0]            spike_in,
    input  logic                    step_valid,
    input  logic                    frame_start,
    input  logic [1:0]              step_index,
    output logic                    ready,
    output logic                    result_valid,
    output logic [3:0]              result_class,
    output logic signed [16:0]      current_out,
    output logic signed [20:0]      score_out,
    output logic                    step_done,
    output logic                    frame_done,
    output logic [3:0]              class_out
);
    import onn_snn_params_pkg::*;

    typedef enum logic [1:0] {STATE_IDLE, STATE_CLEAR, STATE_MAC} state_t;
    state_t state;
    logic [511:0] spike_latched;
    logic [1:0] step_latched;
    logic [3:0] clear_index;
    logic [3:0] class_index;
    logic [8:0] input_index;
    logic signed [READOUT_CURRENT_BITS-1:0] raw_accumulator;
    logic signed [7:0] readout_weights [0:READOUT_WEIGHT_COUNT-1];
    logic signed [WEIGHTED_LOGITS_BITS-1:0] scores [0:9];
    logic signed [WEIGHTED_LOGITS_BITS-1:0] best_score;
    logic [3:0] best_class;

    integer weight_address;
    integer temporal_coefficient;
    logic signed [READOUT_CURRENT_BITS-1:0] contribution;
    logic signed [READOUT_CURRENT_BITS-1:0] raw_sum_comb;
    logic signed [31:0] weighted_extended;
    logic signed [WEIGHTED_LOGITS_BITS-1:0] weighted_comb;
    logic class_is_better;

    initial begin
        if (READOUT_CURRENT_BITS != 17 || WEIGHTED_LOGITS_BITS != 21) begin
            $error("onn_readout port widths disagree with frozen params.svh");
        end
        $readmemh(READOUT_WEIGHT_FILE, readout_weights);
    end

    always_comb begin
        case (step_latched)
            2'd0: temporal_coefficient = TEMPORAL_COEFF_0;
            2'd1: temporal_coefficient = TEMPORAL_COEFF_1;
            2'd2: temporal_coefficient = TEMPORAL_COEFF_2;
            default: temporal_coefficient = TEMPORAL_COEFF_3;
        endcase
        weight_address = class_index * 512 + input_index;
        contribution = '0;
        if (spike_latched[input_index]) begin
            contribution = {{(READOUT_CURRENT_BITS-8){readout_weights[weight_address][7]}},
                            readout_weights[weight_address]};
        end
        if (input_index == 0) begin
            raw_sum_comb = contribution;
        end else begin
            raw_sum_comb = raw_accumulator + contribution;
        end
        weighted_extended =
            $signed(scores[class_index]) + $signed(raw_sum_comb) * temporal_coefficient;
    end

    onn_signed_saturate #(
        .IN_BITS(32),
        .OUT_BITS(WEIGHTED_LOGITS_BITS)
    ) clamp_weighted_score (
        .value_in(weighted_extended),
        .value_out(weighted_comb)
    );

    always_comb begin
        ready = (state == STATE_IDLE);
        class_is_better = $signed(weighted_comb) > $signed(best_score);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            spike_latched <= '0;
            step_latched <= '0;
            clear_index <= '0;
            class_index <= '0;
            input_index <= '0;
            raw_accumulator <= '0;
            best_score <= '0;
            best_class <= '0;
            result_valid <= 1'b0;
            result_class <= '0;
            current_out <= '0;
            score_out <= '0;
            step_done <= 1'b0;
            frame_done <= 1'b0;
            class_out <= '0;
        end else begin
            result_valid <= 1'b0;
            step_done <= 1'b0;
            frame_done <= 1'b0;
            case (state)
                STATE_IDLE: begin
                    if (step_valid) begin
                        spike_latched <= spike_in;
                        step_latched <= step_index;
                        class_index <= '0;
                        input_index <= '0;
                        raw_accumulator <= '0;
                        best_score <= '0;
                        best_class <= '0;
                        if (frame_start) begin
                            clear_index <= '0;
                            state <= STATE_CLEAR;
                        end else begin
                            state <= STATE_MAC;
                        end
                    end
                end
                STATE_CLEAR: begin
                    scores[clear_index] <= '0;
                    if (clear_index == 9) begin
                        class_index <= '0;
                        input_index <= '0;
                        raw_accumulator <= '0;
                        state <= STATE_MAC;
                    end else begin
                        clear_index <= clear_index + 1'b1;
                    end
                end
                STATE_MAC: begin
                    raw_accumulator <= raw_sum_comb;
                    if (input_index == 511) begin
                        scores[class_index] <= weighted_comb;
                        result_valid <= 1'b1;
                        result_class <= class_index;
                        current_out <= raw_sum_comb;
                        score_out <= weighted_comb;
                        input_index <= '0;
                        raw_accumulator <= '0;
                        if (class_index == 0) begin
                            best_score <= weighted_comb;
                            best_class <= '0;
                        end else if (class_is_better) begin
                            best_score <= weighted_comb;
                            best_class <= class_index;
                        end
                        if (class_index == 9) begin
                            step_done <= 1'b1;
                            if (step_latched == 3) begin
                                frame_done <= 1'b1;
                                if (class_is_better) begin
                                    class_out <= 9;
                                end else begin
                                    class_out <= best_class;
                                end
                            end
                            state <= STATE_IDLE;
                        end else begin
                            class_index <= class_index + 1'b1;
                        end
                    end else begin
                        input_index <= input_index + 1'b1;
                    end
                end
                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
