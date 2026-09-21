`timescale 1ns/1ps

// Complete frozen T=4 SNN inference core assembled from the serial engines.
module onn_snn_core_dense #(
    parameter CONV1_WEIGHT_FILE = "model/snn/export/conv1_weight.mem",
    parameter CONV2_WEIGHT_FILE = "model/snn/export/conv2_weight.mem",
    parameter READOUT_WEIGHT_FILE = "model/snn/export/readout_weight.mem"
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic [63:0]             spike_in,
    input  logic                    step_valid,
    input  logic                    frame_start,
    output logic                    ready,
    output logic                    step_done,
    output logic                    frame_done,
    output logic [3:0]              class_out,

    output logic                    l1_valid,
    output logic [9:0]              l1_index,
    output logic signed [15:0]      l1_current,
    output logic signed [17:0]      l1_integrated,
    output logic signed [17:0]      l1_membrane,
    output logic                    l1_spike,

    output logic                    l2_valid,
    output logic [8:0]              l2_index,
    output logic signed [19:0]      l2_current,
    output logic signed [21:0]      l2_integrated,
    output logic signed [21:0]      l2_membrane,
    output logic                    l2_spike,

    output logic                    readout_valid,
    output logic [3:0]              readout_class,
    output logic signed [16:0]      readout_current,
    output logic signed [20:0]      weighted_score
);
    typedef enum logic [1:0] {
        CORE_IDLE,
        CORE_RUN_L1,
        CORE_RUN_L2,
        CORE_RUN_READOUT
    } core_state_t;

    core_state_t state;
    logic [1:0] timestep_counter;
    logic [1:0] active_step;
    logic active_frame_start;
    logic l1_start;
    logic l2_start;
    logic readout_start;
    logic l1_ready;
    logic l2_ready;
    logic readout_ready;
    logic l1_done;
    logic l2_done;
    logic readout_done;
    logic readout_frame_done;
    logic [63:0] input_spike_latched;
    logic [1023:0] l1_spike_buffer;
    logic [511:0] l2_spike_buffer;

    onn_conv1_if1 #(
        .CONV1_WEIGHT_FILE(CONV1_WEIGHT_FILE)
    ) layer1 (
        .clk(clk), .rst_n(rst_n), .spike_in(input_spike_latched),
        .step_valid(l1_start), .frame_start(active_frame_start),
        .ready(l1_ready), .result_valid(l1_valid), .result_index(l1_index),
        .current_out(l1_current), .integrated_out(l1_integrated),
        .membrane_out(l1_membrane), .spike_out(l1_spike), .step_done(l1_done)
    );

    onn_conv2_if2 #(
        .CONV2_WEIGHT_FILE(CONV2_WEIGHT_FILE)
    ) layer2 (
        .clk(clk), .rst_n(rst_n), .spike_in(l1_spike_buffer),
        .step_valid(l2_start), .frame_start(active_frame_start),
        .ready(l2_ready), .result_valid(l2_valid), .result_index(l2_index),
        .current_out(l2_current), .integrated_out(l2_integrated),
        .membrane_out(l2_membrane), .spike_out(l2_spike), .step_done(l2_done)
    );

    onn_readout #(
        .READOUT_WEIGHT_FILE(READOUT_WEIGHT_FILE)
    ) output_layer (
        .clk(clk), .rst_n(rst_n), .spike_in(l2_spike_buffer),
        .step_valid(readout_start), .frame_start(active_frame_start),
        .step_index(active_step), .ready(readout_ready),
        .result_valid(readout_valid), .result_class(readout_class),
        .current_out(readout_current), .score_out(weighted_score),
        .step_done(readout_done), .frame_done(readout_frame_done),
        .class_out(class_out)
    );

    always_comb begin
        ready = (state == CORE_IDLE) && l1_ready && l2_ready && readout_ready;
        step_done = readout_done;
        frame_done = readout_frame_done;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= CORE_IDLE;
            timestep_counter <= '0;
            active_step <= '0;
            active_frame_start <= 1'b0;
            l1_start <= 1'b0;
            l2_start <= 1'b0;
            readout_start <= 1'b0;
            input_spike_latched <= '0;
            l1_spike_buffer <= '0;
            l2_spike_buffer <= '0;
        end else begin
            l1_start <= 1'b0;
            l2_start <= 1'b0;
            readout_start <= 1'b0;
            if (l1_valid) begin
                l1_spike_buffer[l1_index] <= l1_spike;
            end
            if (l2_valid) begin
                l2_spike_buffer[l2_index] <= l2_spike;
            end
            case (state)
                CORE_IDLE: begin
                    if (step_valid && ready) begin
                        input_spike_latched <= spike_in;
                        active_frame_start <= frame_start;
                        if (frame_start) begin
                            active_step <= '0;
                            timestep_counter <= '0;
                        end else begin
                            active_step <= timestep_counter;
                        end
                        l1_start <= 1'b1;
                        state <= CORE_RUN_L1;
                    end
                end
                CORE_RUN_L1: begin
                    if (l1_done) begin
                        l2_start <= 1'b1;
                        state <= CORE_RUN_L2;
                    end
                end
                CORE_RUN_L2: begin
                    if (l2_done) begin
                        readout_start <= 1'b1;
                        state <= CORE_RUN_READOUT;
                    end
                end
                CORE_RUN_READOUT: begin
                    if (readout_done) begin
                        if (active_step == 3) begin
                            timestep_counter <= '0;
                        end else begin
                            timestep_counter <= active_step + 1'b1;
                        end
                        state <= CORE_IDLE;
                    end
                end
                default: state <= CORE_IDLE;
            endcase
        end
    end
endmodule
