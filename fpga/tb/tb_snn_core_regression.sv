`timescale 1ns/1ps

// Compact end-to-end differential regression; detailed streams remain in tb_snn_core.
module tb_snn_core_regression;
    logic clk = 0, rst_n = 0;
    logic [63:0] spike_in = 0;
    logic step_valid = 0, frame_start = 0;
    wire ready, step_done, frame_done;
    wire [3:0] class_out;
    wire l1_valid, l1_spike, l2_valid, l2_spike, readout_valid;
    wire [9:0] l1_index;
    wire [8:0] l2_index;
    wire [3:0] readout_class;
    wire signed [20:0] weighted_score;
    wire [31:0] step_synaptic_additions, frame_synaptic_additions;
    wire [31:0] step_compute_cycles, frame_compute_cycles;
    reg [1023:0] observed_l1, expected_l1;
    reg [511:0] observed_l2, expected_l2;
    integer observed_scores [0:9], expected_scores [0:9];
    integer input_file, expected_file, step_count, step_id, expected_step;
    integer frame_flag, expected_class, expected_step_add, expected_step_cycles;
    integer expected_frame_add, expected_frame_cycles;
    integer count, timeout, score_count, k;
    reg [63:0] input_word;

    always #5 clk = ~clk;

    onn_snn_core #(
        .CONV1_WEIGHT_FILE("../../model/snn/export/conv1_weight.mem"),
        .CONV2_WEIGHT_FILE("../../model/snn/export/conv2_weight.mem"),
        .READOUT_WEIGHT_FILE("../../model/snn/export/readout_weight.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .spike_in(spike_in), .step_valid(step_valid),
        .frame_start(frame_start), .ready(ready), .step_done(step_done),
        .frame_done(frame_done), .class_out(class_out),
        .l1_valid(l1_valid), .l1_index(l1_index), .l1_spike(l1_spike),
        .l2_valid(l2_valid), .l2_index(l2_index), .l2_spike(l2_spike),
        .readout_valid(readout_valid), .readout_class(readout_class),
        .weighted_score(weighted_score),
        .step_synaptic_additions(step_synaptic_additions),
        .frame_synaptic_additions(frame_synaptic_additions),
        .step_compute_cycles(step_compute_cycles),
        .frame_compute_cycles(frame_compute_cycles)
    );

    initial begin
        input_file = $fopen("../vectors/regression_inputs.txt", "r");
        expected_file = $fopen("../vectors/regression_expected.txt", "r");
        if (!input_file || !expected_file) $fatal(1, "cannot open compact regression vectors");
        count = $fscanf(input_file, "%d\n", step_count);
        repeat (3) @(negedge clk);
        rst_n = 1;
        for (step_id = 0; step_id < step_count; step_id = step_id + 1) begin
            count = $fscanf(input_file, "%d %d %h\n", expected_step, frame_flag, input_word);
            if (count != 3 || expected_step != step_id) $fatal(1, "bad regression input %0d", step_id);
            count = $fscanf(expected_file, "%d %h %h %d %d %d %d %d\n",
                expected_step, expected_l1, expected_l2, expected_step_add,
                expected_step_cycles, expected_frame_add, expected_frame_cycles,
                expected_class);
            if (count != 8 || expected_step != step_id) $fatal(1, "bad expected header %0d", step_id);
            for (k = 0; k < 10; k = k + 1) begin
                count = $fscanf(expected_file, "%d", expected_scores[k]);
                if (count != 1) $fatal(1, "bad expected score %0d/%0d", step_id, k);
            end
            observed_l1 = '0;
            observed_l2 = '0;
            score_count = 0;
            while (!ready) @(negedge clk);
            spike_in = input_word;
            frame_start = frame_flag;
            step_valid = 1;
            @(negedge clk);
            step_valid = 0;
            frame_start = 0;
            timeout = 0;
            while (!step_done) begin
                if (l1_valid) observed_l1[l1_index] = l1_spike;
                if (l2_valid) observed_l2[l2_index] = l2_spike;
                if (readout_valid) begin
                    observed_scores[readout_class] = $signed(weighted_score);
                    score_count = score_count + 1;
                end
                @(negedge clk);
                timeout = timeout + 1;
                if (timeout > 120000) $fatal(1, "regression timeout step %0d", step_id);
            end
            if (l1_valid) observed_l1[l1_index] = l1_spike;
            if (l2_valid) observed_l2[l2_index] = l2_spike;
            if (readout_valid) begin
                observed_scores[readout_class] = $signed(weighted_score);
                score_count = score_count + 1;
            end
            if (observed_l1 !== expected_l1 || observed_l2 !== expected_l2)
                $fatal(1, "hidden spike mismatch step %0d", step_id);
            if (score_count != 10) $fatal(1, "score count mismatch step %0d", step_id);
            for (k = 0; k < 10; k = k + 1)
                if (observed_scores[k] !== expected_scores[k])
                    $fatal(1, "score mismatch step %0d class %0d got %0d expected %0d",
                        step_id, k, observed_scores[k], expected_scores[k]);
            if (frame_done != (step_id % 4 == 3))
                $fatal(1, "frame_done mismatch step %0d", step_id);
            if (frame_done && class_out != expected_class)
                $fatal(1, "class mismatch step %0d", step_id);
            @(negedge clk);
            if (step_synaptic_additions != expected_step_add ||
                step_compute_cycles != expected_step_cycles ||
                frame_synaptic_additions != expected_frame_add ||
                frame_compute_cycles != expected_frame_cycles)
                $fatal(1, "counter mismatch step %0d", step_id);
        end
        $display("PASS tb_snn_core_regression: %0d steps, %0d frames", step_count, step_count / 4);
        $finish;
    end
endmodule
