`timescale 1ns/1ps

// Phase-1 serial Conv1 + IF1 engine.
//
// Input bit (y*8+x) is pixel [y,x]. Results stream in contiguous PyTorch
// [channel,y,x] order, result_index = channel*64 + y*8 + x.
module onn_conv1_if1 #(
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
    output logic                    step_done
);
    import onn_snn_params_pkg::*;

    localparam integer OUTPUT_COUNT = 16 * 8 * 8;

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_CLEAR,
        STATE_MAC
    } state_t;

    state_t state;
    logic [63:0] spike_latched;
    logic [9:0] clear_index;
    logic [9:0] output_index;
    logic [3:0] kernel_index;
    logic signed [CONV1_RAW_ACC_BITS-1:0] raw_accumulator;
    logic signed [7:0] conv1_weights [0:CONV1_WEIGHT_COUNT-1];
    logic signed [LIF1_MEM_BITS-1:0] membrane [0:OUTPUT_COUNT-1];

    integer output_channel;
    integer output_pixel;
    integer output_y;
    integer output_x;
    integer kernel_y;
    integer kernel_x;
    integer input_y;
    integer input_x;
    integer weight_address;
    logic input_is_valid;
    logic signed [CONV1_RAW_ACC_BITS-1:0] contribution;
    logic signed [CONV1_RAW_ACC_BITS-1:0] raw_sum_comb;
    logic signed [31:0] raw_sum_extended;
    logic signed [31:0] current_wide;
    logic signed [CONV1_CURRENT_BITS-1:0] current_comb;
    logic signed [LIF1_MEM_BITS-1:0] integrated_comb;
    logic signed [LIF1_MEM_BITS-1:0] membrane_next_comb;
    logic spike_comb;

    initial begin
        if (CONV1_CURRENT_BITS != 16 || LIF1_MEM_BITS != 18 ||
            CONV1_RAW_ACC_BITS != 12 || LIF1_THRESHOLD != 939) begin
            $error("onn_conv1_if1 port widths disagree with frozen params.svh");
        end
        $readmemh(CONV1_WEIGHT_FILE, conv1_weights);
    end

    always_comb begin
        output_channel = output_index / 64;
        output_pixel = output_index % 64;
        output_y = output_pixel / 8;
        output_x = output_pixel % 8;
        kernel_y = kernel_index / 3;
        kernel_x = kernel_index % 3;
        input_y = output_y + kernel_y - 1;
        input_x = output_x + kernel_x - 1;
        weight_address = output_channel * 9 + kernel_index;
        input_is_valid = (input_y >= 0) && (input_y < 8) &&
                         (input_x >= 0) && (input_x < 8);
        contribution = '0;
        if (input_is_valid && spike_latched[input_y * 8 + input_x]) begin
            contribution = {{(CONV1_RAW_ACC_BITS-8){conv1_weights[weight_address][7]}},
                            conv1_weights[weight_address]};
        end
        if (kernel_index == 0) begin
            raw_sum_comb = contribution;
        end else begin
            raw_sum_comb = raw_accumulator + contribution;
        end
        raw_sum_extended =
            {{(32-CONV1_RAW_ACC_BITS){raw_sum_comb[CONV1_RAW_ACC_BITS-1]}}, raw_sum_comb};
        current_wide = raw_sum_extended <<< SNN_STATE_GUARD_BITS;
    end

    onn_if_unit #(
        .CURRENT_IN_BITS(32),
        .CURRENT_BITS(CONV1_CURRENT_BITS),
        .MEM_BITS(LIF1_MEM_BITS),
        .THRESHOLD(LIF1_THRESHOLD)
    ) if1_update (
        .current_in(current_wide),
        .membrane_in(membrane[output_index]),
        .current_saturated(current_comb),
        .integrated(integrated_comb),
        .spike(spike_comb),
        .membrane_next(membrane_next_comb)
    );

    always_comb begin
        ready = (state == STATE_IDLE);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            spike_latched <= '0;
            clear_index <= '0;
            output_index <= '0;
            kernel_index <= '0;
            raw_accumulator <= '0;
            result_valid <= 1'b0;
            result_index <= '0;
            current_out <= '0;
            integrated_out <= '0;
            membrane_out <= '0;
            spike_out <= 1'b0;
            step_done <= 1'b0;
        end else begin
            result_valid <= 1'b0;
            step_done <= 1'b0;
            case (state)
                STATE_IDLE: begin
                    if (step_valid) begin
                        spike_latched <= spike_in;
                        output_index <= '0;
                        kernel_index <= '0;
                        raw_accumulator <= '0;
                        if (frame_start) begin
                            clear_index <= '0;
                            state <= STATE_CLEAR;
                        end else begin
                            state <= STATE_MAC;
                        end
                    end
                end
                STATE_CLEAR: begin
                    membrane[clear_index] <= '0;
                    if (clear_index == OUTPUT_COUNT - 1) begin
                        output_index <= '0;
                        kernel_index <= '0;
                        raw_accumulator <= '0;
                        state <= STATE_MAC;
                    end else begin
                        clear_index <= clear_index + 1'b1;
                    end
                end
                STATE_MAC: begin
                    raw_accumulator <= raw_sum_comb;
                    if (kernel_index == 8) begin
                        membrane[output_index] <= membrane_next_comb;
                        result_valid <= 1'b1;
                        result_index <= output_index;
                        current_out <= current_comb;
                        integrated_out <= integrated_comb;
                        membrane_out <= membrane_next_comb;
                        spike_out <= spike_comb;
                        kernel_index <= '0;
                        raw_accumulator <= '0;
                        if (output_index == OUTPUT_COUNT - 1) begin
                            step_done <= 1'b1;
                            state <= STATE_IDLE;
                        end else begin
                            output_index <= output_index + 1'b1;
                        end
                    end else begin
                        kernel_index <= kernel_index + 1'b1;
                    end
                end
                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
