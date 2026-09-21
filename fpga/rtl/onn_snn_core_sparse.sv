`timescale 1ns/1ps

// Deployment top: spike-driven synaptic work with exact per-timestep IF barriers.
module onn_snn_core #(
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
    output logic signed [20:0]      weighted_score,

    output logic [31:0]             step_synaptic_additions,
    output logic [31:0]             frame_synaptic_additions,
    output logic [31:0]             step_compute_cycles,
    output logic [31:0]             frame_compute_cycles
);
    typedef enum logic [1:0] {
        CORE_IDLE, CORE_RUN_L1, CORE_RUN_L2, CORE_RUN_READOUT
    } core_state_t;
    core_state_t state;
    logic [1:0] timestep_counter;
    logic [1:0] active_step;
    logic active_frame_start;
    logic [63:0] input_spike_latched;
    logic l1_start, l2_start, readout_start;
    logic l1_ready, l2_ready, readout_ready;
    logic l1_done, l2_done, readout_done, readout_frame_done;
    logic [1023:0] l1_spike_buffer;
    logic [511:0] l2_spike_buffer;
    logic [15:0] l1_additions;
    logic [16:0] l2_additions;
    logic [15:0] readout_additions;
    logic [31:0] l1_cycles, l2_cycles, readout_cycles;
    logic [31:0] completed_step_additions;
    logic [31:0] completed_step_cycles;

    onn_sparse_conv1_if1 #(
        .CONV1_WEIGHT_FILE(CONV1_WEIGHT_FILE)
    ) layer1 (
        .clk(clk), .rst_n(rst_n), .spike_in(input_spike_latched),
        .step_valid(l1_start), .frame_start(active_frame_start),
        .ready(l1_ready), .result_valid(l1_valid), .result_index(l1_index),
        .current_out(l1_current), .integrated_out(l1_integrated),
        .membrane_out(l1_membrane), .spike_out(l1_spike), .step_done(l1_done),
        .synaptic_additions(l1_additions), .cycles_used(l1_cycles)
    );

    onn_sparse_conv2_if2 #(
        .CONV2_WEIGHT_FILE(CONV2_WEIGHT_FILE)
    ) layer2 (
        .clk(clk), .rst_n(rst_n), .spike_in(l1_spike_buffer),
        .step_valid(l2_start), .frame_start(active_frame_start),
        .ready(l2_ready), .result_valid(l2_valid), .result_index(l2_index),
        .current_out(l2_current), .integrated_out(l2_integrated),
        .membrane_out(l2_membrane), .spike_out(l2_spike), .step_done(l2_done),
        .synaptic_additions(l2_additions), .cycles_used(l2_cycles)
    );

    onn_sparse_readout #(
        .READOUT_WEIGHT_FILE(READOUT_WEIGHT_FILE)
    ) output_layer (
        .clk(clk), .rst_n(rst_n), .spike_in(l2_spike_buffer),
        .step_valid(readout_start), .frame_start(active_frame_start),
        .step_index(active_step), .ready(readout_ready),
        .result_valid(readout_valid), .result_class(readout_class),
        .current_out(readout_current), .score_out(weighted_score),
        .step_done(readout_done), .frame_done(readout_frame_done),
        .class_out(class_out), .synaptic_additions(readout_additions),
        .cycles_used(readout_cycles)
    );

    always_comb begin
        ready = (state == CORE_IDLE) && l1_ready && l2_ready && readout_ready;
        step_done = readout_done;
        frame_done = readout_frame_done;
        completed_step_additions = l1_additions + l2_additions + readout_additions;
        completed_step_cycles = l1_cycles + l2_cycles + readout_cycles;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= CORE_IDLE;
            timestep_counter <= '0;
            active_step <= '0;
            active_frame_start <= 1'b0;
            input_spike_latched <= '0;
            l1_start <= 1'b0;
            l2_start <= 1'b0;
            readout_start <= 1'b0;
            l1_spike_buffer <= '0;
            l2_spike_buffer <= '0;
            step_synaptic_additions <= '0;
            frame_synaptic_additions <= '0;
            step_compute_cycles <= '0;
            frame_compute_cycles <= '0;
        end else begin
            l1_start <= 1'b0;
            l2_start <= 1'b0;
            readout_start <= 1'b0;
            if (l1_valid) l1_spike_buffer[l1_index] <= l1_spike;
            if (l2_valid) l2_spike_buffer[l2_index] <= l2_spike;
            case (state)
                CORE_IDLE: begin
                    if (step_valid && ready) begin
                        input_spike_latched <= spike_in;
                        active_frame_start <= frame_start;
                        l1_spike_buffer <= '0;
                        l2_spike_buffer <= '0;
                        if (frame_start) begin
                            active_step <= '0;
                            timestep_counter <= '0;
                            frame_synaptic_additions <= '0;
                            frame_compute_cycles <= '0;
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
                        step_synaptic_additions <= completed_step_additions;
                        step_compute_cycles <= completed_step_cycles;
                        frame_synaptic_additions <=
                            frame_synaptic_additions + completed_step_additions;
                        frame_compute_cycles <= frame_compute_cycles + completed_step_cycles;
                        if (active_step == 3) timestep_counter <= '0;
                        else timestep_counter <= active_step + 1'b1;
                        state <= CORE_IDLE;
                    end
                end
                default: state <= CORE_IDLE;
            endcase
        end
    end
endmodule
