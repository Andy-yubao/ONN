`timescale 1ns/1ps

// Spike-driven 512->10 readout. Each L2 spike causes exactly ten weight adds.
module onn_sparse_readout #(
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
    output logic [3:0]              class_out,
    output logic [15:0]             synaptic_additions,
    output logic [31:0]             cycles_used
);
    import onn_snn_params_pkg::*;

    typedef enum logic [2:0] {
        STATE_IDLE, STATE_CLEAR, STATE_FIND_EVENT, STATE_SCATTER, STATE_UPDATE
    } state_t;
    state_t state;
    logic [511:0] spike_latched;
    logic [511:0] event_mask;
    logic [4:0] event_group;
    logic [31:0] event_word;
    logic event_found;
    logic [4:0] event_bit_comb;
    logic [8:0] event_index_comb;
    integer priority_index;
    logic [8:0] active_event_index;
    logic [3:0] class_index;
    logic [3:0] clear_index;
    logic [1:0] step_latched;
    logic signed [7:0] readout_weights [0:READOUT_WEIGHT_COUNT-1];
    logic signed [READOUT_CURRENT_BITS-1:0] current_accumulator [0:9];
    logic signed [WEIGHTED_LOGITS_BITS-1:0] scores [0:9];
    logic signed [WEIGHTED_LOGITS_BITS-1:0] best_score;
    logic [3:0] best_class;

    integer weight_address;
    integer temporal_coefficient;
    logic signed [READOUT_CURRENT_BITS-1:0] weight_extended;
    logic signed [31:0] weighted_extended;
    logic signed [WEIGHTED_LOGITS_BITS-1:0] weighted_comb;
    logic class_is_better;

    initial $readmemh(READOUT_WEIGHT_FILE, readout_weights);

    always_comb begin
        if (event_group < 16) event_word = event_mask[event_group * 32 +: 32];
        else event_word = '0;
        event_found = 1'b0;
        event_bit_comb = '0;
        for (priority_index = 0; priority_index < 32; priority_index = priority_index + 1) begin
            if (!event_found && event_word[priority_index]) begin
                event_found = 1'b1;
                event_bit_comb = priority_index;
            end
        end
        event_index_comb = event_group * 32 + event_bit_comb;
    end

    always_comb begin
        case (step_latched)
            2'd0: temporal_coefficient = TEMPORAL_COEFF_0;
            2'd1: temporal_coefficient = TEMPORAL_COEFF_1;
            2'd2: temporal_coefficient = TEMPORAL_COEFF_2;
            default: temporal_coefficient = TEMPORAL_COEFF_3;
        endcase
        weight_address = class_index * 512 + active_event_index;
        weight_extended = {{(READOUT_CURRENT_BITS-8){readout_weights[weight_address][7]}},
                           readout_weights[weight_address]};
        weighted_extended = $signed(scores[class_index]) +
                            $signed(current_accumulator[class_index]) * temporal_coefficient;
        class_is_better = $signed(weighted_comb) > $signed(best_score);
    end

    onn_signed_saturate #(
        .IN_BITS(32), .OUT_BITS(WEIGHTED_LOGITS_BITS)
    ) clamp_weighted_score (
        .value_in(weighted_extended), .value_out(weighted_comb)
    );

    always_comb ready = (state == STATE_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            spike_latched <= '0;
            event_mask <= '0;
            event_group <= '0;
            active_event_index <= '0;
            class_index <= '0;
            clear_index <= '0;
            step_latched <= '0;
            best_score <= '0;
            best_class <= '0;
            result_valid <= 1'b0;
            result_class <= '0;
            current_out <= '0;
            score_out <= '0;
            step_done <= 1'b0;
            frame_done <= 1'b0;
            class_out <= '0;
            synaptic_additions <= '0;
            cycles_used <= '0;
        end else begin
            result_valid <= 1'b0;
            step_done <= 1'b0;
            frame_done <= 1'b0;
            if (state != STATE_IDLE) cycles_used <= cycles_used + 1'b1;
            case (state)
                STATE_IDLE: begin
                    if (step_valid) begin
                        spike_latched <= spike_in;
                        step_latched <= step_index;
                        synaptic_additions <= '0;
                        cycles_used <= '0;
                        if (frame_start) begin
                            clear_index <= '0;
                            state <= STATE_CLEAR;
                        end else begin
                            event_mask <= spike_in;
                            event_group <= '0;
                            state <= STATE_FIND_EVENT;
                        end
                    end
                end
                STATE_CLEAR: begin
                    scores[clear_index] <= '0;
                    current_accumulator[clear_index] <= '0;
                    if (clear_index == 9) begin
                        event_mask <= spike_latched;
                        event_group <= '0;
                        state <= STATE_FIND_EVENT;
                    end else begin
                        clear_index <= clear_index + 1'b1;
                    end
                end
                STATE_FIND_EVENT: begin
                    if (event_group == 16) begin
                        class_index <= '0;
                        best_score <= '0;
                        best_class <= '0;
                        state <= STATE_UPDATE;
                    end else if (event_found) begin
                        event_mask[event_index_comb] <= 1'b0;
                        active_event_index <= event_index_comb;
                        class_index <= '0;
                        state <= STATE_SCATTER;
                    end else begin
                        event_group <= event_group + 1'b1;
                    end
                end
                STATE_SCATTER: begin
                    current_accumulator[class_index] <=
                        current_accumulator[class_index] + weight_extended;
                    synaptic_additions <= synaptic_additions + 1'b1;
                    if (class_index == 9) begin
                        class_index <= '0;
                        state <= STATE_FIND_EVENT;
                    end else begin
                        class_index <= class_index + 1'b1;
                    end
                end
                STATE_UPDATE: begin
                    scores[class_index] <= weighted_comb;
                    current_accumulator[class_index] <= '0;
                    result_valid <= 1'b1;
                    result_class <= class_index;
                    current_out <= current_accumulator[class_index];
                    score_out <= weighted_comb;
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
                            if (class_is_better) class_out <= 9;
                            else class_out <= best_class;
                        end
                        state <= STATE_IDLE;
                    end else begin
                        class_index <= class_index + 1'b1;
                    end
                end
                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
