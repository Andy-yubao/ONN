`timescale 1ns/1ps

// Phase-2 serial Conv2 + IF2 engine.
// Input bit (channel*64+y*8+x) uses [16,8,8] contiguous order.
// Results stream as [32,4,4], index = channel*16+y*4+x.
module onn_conv2_if2 #(
    parameter CONV2_WEIGHT_FILE = "model/snn/export/conv2_weight.mem"
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic [1023:0]           spike_in,
    input  logic                    step_valid,
    input  logic                    frame_start,
    output logic                    ready,
    output logic                    result_valid,
    output logic [8:0]              result_index,
    output logic signed [19:0]      current_out,
    output logic signed [21:0]      integrated_out,
    output logic signed [21:0]      membrane_out,
    output logic                    spike_out,
    output logic                    step_done
);
    import onn_snn_params_pkg::*;

    localparam integer OUTPUT_COUNT = 32 * 4 * 4;
    localparam integer MAC_COUNT = 16 * 3 * 3;

    typedef enum logic [1:0] {STATE_IDLE, STATE_CLEAR, STATE_MAC} state_t;
    state_t state;
    logic [1023:0] spike_latched;
    logic [8:0] clear_index;
    logic [8:0] output_index;
    logic [7:0] mac_index;
    logic signed [CONV2_RAW_ACC_BITS-1:0] raw_accumulator;
    logic signed [7:0] conv2_weights [0:CONV2_WEIGHT_COUNT-1];
    logic signed [LIF2_MEM_BITS-1:0] membrane [0:OUTPUT_COUNT-1];

    integer output_channel;
    integer output_pixel;
    integer output_y;
    integer output_x;
    integer input_channel;
    integer kernel_index;
    integer kernel_y;
    integer kernel_x;
    integer input_y;
    integer input_x;
    integer input_address;
    integer weight_address;
    logic input_is_valid;
    logic signed [CONV2_RAW_ACC_BITS-1:0] contribution;
    logic signed [CONV2_RAW_ACC_BITS-1:0] raw_sum_comb;
    logic signed [31:0] raw_sum_extended;
    logic signed [31:0] current_wide;
    logic signed [CONV2_CURRENT_BITS-1:0] current_comb;
    logic signed [LIF2_MEM_BITS-1:0] integrated_comb;
    logic signed [LIF2_MEM_BITS-1:0] membrane_next_comb;
    logic spike_comb;

    initial begin
        if (CONV2_CURRENT_BITS != 20 || LIF2_MEM_BITS != 22 ||
            CONV2_RAW_ACC_BITS != 16 || LIF2_THRESHOLD != 1715) begin
            $error("onn_conv2_if2 port widths disagree with frozen params.svh");
        end
        $readmemh(CONV2_WEIGHT_FILE, conv2_weights);
    end

    always_comb begin
        output_channel = output_index / 16;
        output_pixel = output_index % 16;
        output_y = output_pixel / 4;
        output_x = output_pixel % 4;
        input_channel = mac_index / 9;
        kernel_index = mac_index % 9;
        kernel_y = kernel_index / 3;
        kernel_x = kernel_index % 3;
        input_y = output_y * 2 + kernel_y - 1;
        input_x = output_x * 2 + kernel_x - 1;
        input_address = input_channel * 64 + input_y * 8 + input_x;
        weight_address = output_channel * MAC_COUNT + input_channel * 9 + kernel_index;
        input_is_valid = (input_y >= 0) && (input_y < 8) &&
                         (input_x >= 0) && (input_x < 8);
        contribution = '0;
        if (input_is_valid && spike_latched[input_address]) begin
            contribution = {{(CONV2_RAW_ACC_BITS-8){conv2_weights[weight_address][7]}},
                            conv2_weights[weight_address]};
        end
        if (mac_index == 0) begin
            raw_sum_comb = contribution;
        end else begin
            raw_sum_comb = raw_accumulator + contribution;
        end
        raw_sum_extended =
            {{(32-CONV2_RAW_ACC_BITS){raw_sum_comb[CONV2_RAW_ACC_BITS-1]}}, raw_sum_comb};
        current_wide = raw_sum_extended <<< SNN_STATE_GUARD_BITS;
    end

    onn_if_unit #(
        .CURRENT_IN_BITS(32),
        .CURRENT_BITS(CONV2_CURRENT_BITS),
        .MEM_BITS(LIF2_MEM_BITS),
        .THRESHOLD(LIF2_THRESHOLD)
    ) if2_update (
        .current_in(current_wide),
        .membrane_in(membrane[output_index]),
        .current_saturated(current_comb),
        .integrated(integrated_comb),
        .spike(spike_comb),
        .membrane_next(membrane_next_comb)
    );

    always_comb ready = (state == STATE_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            spike_latched <= '0;
            clear_index <= '0;
            output_index <= '0;
            mac_index <= '0;
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
                        mac_index <= '0;
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
                        mac_index <= '0;
                        raw_accumulator <= '0;
                        state <= STATE_MAC;
                    end else begin
                        clear_index <= clear_index + 1'b1;
                    end
                end
                STATE_MAC: begin
                    raw_accumulator <= raw_sum_comb;
                    if (mac_index == MAC_COUNT - 1) begin
                        membrane[output_index] <= membrane_next_comb;
                        result_valid <= 1'b1;
                        result_index <= output_index;
                        current_out <= current_comb;
                        integrated_out <= integrated_comb;
                        membrane_out <= membrane_next_comb;
                        spike_out <= spike_comb;
                        mac_index <= '0;
                        raw_accumulator <= '0;
                        if (output_index == OUTPUT_COUNT - 1) begin
                            step_done <= 1'b1;
                            state <= STATE_IDLE;
                        end else begin
                            output_index <= output_index + 1'b1;
                        end
                    end else begin
                        mac_index <= mac_index + 1'b1;
                    end
                end
                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
