`timescale 1ns/1ps

// Spike-driven Conv1 scatter accumulation followed by an exact IF1 barrier.
// Zero input spikes consume no synaptic-add cycles.
module onn_sparse_conv1_if1 #(
    parameter CONV1_WEIGHT_FILE = "model/snn/export/conv1_weight.mem"
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic [63:0]             spike_in,
    input  logic                    step_valid,
    input  logic                    frame_start,
    output logic                    ready,
    output logic                    result_valid,
    output logic [9:0]              result_index,
    output logic signed [15:0]      current_out,
    output logic signed [17:0]      integrated_out,
    output logic signed [17:0]      membrane_out,
    output logic                    spike_out,
    output logic                    step_done,
    output logic [15:0]             synaptic_additions,
    output logic [31:0]             cycles_used
);
    import onn_snn_params_pkg::*;
    localparam integer OUTPUT_COUNT = 16 * 8 * 8;

    typedef enum logic [2:0] {
        STATE_IDLE, STATE_CLEAR, STATE_FIND_EVENT, STATE_SCATTER, STATE_IF_SCAN
    } state_t;
    state_t state;
    logic [63:0] spike_latched;
    logic [63:0] event_mask;
    logic event_found;
    logic [5:0] event_index_comb;
    integer priority_index;
    logic [2:0] event_y;
    logic [2:0] event_x;
    logic [3:0] output_channel;
    logic [3:0] kernel_index;
    logic [9:0] clear_index;
    logic [9:0] scan_index;
    logic signed [7:0] conv1_weights [0:CONV1_WEIGHT_COUNT-1];
    logic signed [CONV1_RAW_ACC_BITS-1:0] current_accumulator [0:OUTPUT_COUNT-1];
    logic signed [LIF1_MEM_BITS-1:0] membrane [0:OUTPUT_COUNT-1];

    integer kernel_y;
    integer kernel_x;
    integer output_y;
    integer output_x;
    integer output_address;
    integer weight_address;
    logic scatter_is_valid;
    logic signed [CONV1_RAW_ACC_BITS-1:0] weight_extended;
    logic signed [31:0] current_wide;
    logic signed [CONV1_CURRENT_BITS-1:0] current_comb;
    logic signed [LIF1_MEM_BITS-1:0] integrated_comb;
    logic signed [LIF1_MEM_BITS-1:0] membrane_next_comb;
    logic spike_comb;

    initial $readmemh(CONV1_WEIGHT_FILE, conv1_weights);

    always_comb begin
        event_found = 1'b0;
        event_index_comb = '0;
        for (priority_index = 0; priority_index < 64; priority_index = priority_index + 1) begin
            if (!event_found && event_mask[priority_index]) begin
                event_found = 1'b1;
                event_index_comb = priority_index;
            end
        end
    end

    always_comb begin
        kernel_y = kernel_index / 3;
        kernel_x = kernel_index % 3;
        output_y = event_y - kernel_y + 1;
        output_x = event_x - kernel_x + 1;
        output_address = output_channel * 64 + output_y * 8 + output_x;
        weight_address = output_channel * 9 + kernel_index;
        scatter_is_valid = (output_y >= 0) && (output_y < 8) &&
                           (output_x >= 0) && (output_x < 8);
        weight_extended = {{(CONV1_RAW_ACC_BITS-8){conv1_weights[weight_address][7]}},
                           conv1_weights[weight_address]};
        current_wide =
            $signed({{(32-CONV1_RAW_ACC_BITS){current_accumulator[scan_index][CONV1_RAW_ACC_BITS-1]}},
                     current_accumulator[scan_index]}) <<< SNN_STATE_GUARD_BITS;
    end

    onn_if_unit #(
        .CURRENT_IN_BITS(32), .CURRENT_BITS(CONV1_CURRENT_BITS),
        .MEM_BITS(LIF1_MEM_BITS), .THRESHOLD(LIF1_THRESHOLD)
    ) if1_update (
        .current_in(current_wide), .membrane_in(membrane[scan_index]),
        .current_saturated(current_comb), .integrated(integrated_comb),
        .spike(spike_comb), .membrane_next(membrane_next_comb)
    );

    always_comb ready = (state == STATE_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            spike_latched <= '0;
            event_mask <= '0;
            event_y <= '0;
            event_x <= '0;
            output_channel <= '0;
            kernel_index <= '0;
            clear_index <= '0;
            scan_index <= '0;
            result_valid <= 1'b0;
            result_index <= '0;
            current_out <= '0;
            integrated_out <= '0;
            membrane_out <= '0;
            spike_out <= 1'b0;
            step_done <= 1'b0;
            synaptic_additions <= '0;
            cycles_used <= '0;
        end else begin
            result_valid <= 1'b0;
            step_done <= 1'b0;
            if (state != STATE_IDLE) cycles_used <= cycles_used + 1'b1;
            case (state)
                STATE_IDLE: begin
                    if (step_valid) begin
                        spike_latched <= spike_in;
                        synaptic_additions <= '0;
                        cycles_used <= '0;
                        if (frame_start) begin
                            clear_index <= '0;
                            state <= STATE_CLEAR;
                        end else begin
                            event_mask <= spike_in;
                            state <= STATE_FIND_EVENT;
                        end
                    end
                end
                STATE_CLEAR: begin
                    membrane[clear_index] <= '0;
                    current_accumulator[clear_index] <= '0;
                    if (clear_index == OUTPUT_COUNT - 1) begin
                        event_mask <= spike_latched;
                        state <= STATE_FIND_EVENT;
                    end else begin
                        clear_index <= clear_index + 1'b1;
                    end
                end
                STATE_FIND_EVENT: begin
                    if (event_found) begin
                        event_mask[event_index_comb] <= 1'b0;
                        event_y <= event_index_comb / 8;
                        event_x <= event_index_comb % 8;
                        output_channel <= '0;
                        kernel_index <= '0;
                        state <= STATE_SCATTER;
                    end else begin
                        scan_index <= '0;
                        state <= STATE_IF_SCAN;
                    end
                end
                STATE_SCATTER: begin
                    if (scatter_is_valid) begin
                        current_accumulator[output_address] <=
                            current_accumulator[output_address] + weight_extended;
                        synaptic_additions <= synaptic_additions + 1'b1;
                    end
                    if (kernel_index == 8) begin
                        kernel_index <= '0;
                        if (output_channel == 15) begin
                            output_channel <= '0;
                            state <= STATE_FIND_EVENT;
                        end else begin
                            output_channel <= output_channel + 1'b1;
                        end
                    end else begin
                        kernel_index <= kernel_index + 1'b1;
                    end
                end
                STATE_IF_SCAN: begin
                    membrane[scan_index] <= membrane_next_comb;
                    current_accumulator[scan_index] <= '0;
                    result_valid <= 1'b1;
                    result_index <= scan_index;
                    current_out <= current_comb;
                    integrated_out <= integrated_comb;
                    membrane_out <= membrane_next_comb;
                    spike_out <= spike_comb;
                    if (scan_index == OUTPUT_COUNT - 1) begin
                        step_done <= 1'b1;
                        state <= STATE_IDLE;
                    end else begin
                        scan_index <= scan_index + 1'b1;
                    end
                end
                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
