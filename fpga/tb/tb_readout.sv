`timescale 1ns/1ps

module tb_readout;
    logic clk = 0;
    logic rst_n = 0;
    logic [511:0] spike_in;
    logic step_valid = 0, frame_start = 0;
    logic [1:0] step_index;
    logic ready, result_valid, step_done, frame_done;
    logic [3:0] result_class, class_out;
    logic signed [16:0] current_out;
    logic signed [20:0] score_out;
    integer input_file, expected_file, scan_count, case_count, case_loop;
    integer case_id, frame_integer, step_integer, class_loop, timeout_cycles;
    integer expected_case, expected_class, expected_current, expected_score;
    integer expected_best_score, expected_best_class, failures = 0;
    reg [511:0] spike_word;

    always #5 clk = ~clk;

    onn_readout #(
        .READOUT_WEIGHT_FILE("../../model/snn/export/readout_weight.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .spike_in(spike_in),
        .step_valid(step_valid), .frame_start(frame_start), .step_index(step_index),
        .ready(ready), .result_valid(result_valid), .result_class(result_class),
        .current_out(current_out), .score_out(score_out), .step_done(step_done),
        .frame_done(frame_done), .class_out(class_out)
    );

    initial begin
        input_file = $fopen("../vectors/readout_inputs.txt", "r");
        expected_file = $fopen("../vectors/readout_expected.txt", "r");
        if (!input_file || !expected_file) $fatal(1, "cannot open readout vectors");
        scan_count = $fscanf(input_file, "%d\n", case_count);
        repeat (3) @(negedge clk); rst_n = 1;
        for (case_loop = 0; case_loop < case_count; case_loop = case_loop + 1) begin
            scan_count = $fscanf(input_file, "%d %d %d %h\n",
                case_id, frame_integer, step_integer, spike_word);
            if (scan_count != 4 || case_id != case_loop) $fatal(1, "bad readout input");
            while (!ready) @(negedge clk);
            spike_in = spike_word; frame_start = frame_integer;
            step_index = step_integer; step_valid = 1;
            @(negedge clk); step_valid = 0; frame_start = 0;
            for (class_loop = 0; class_loop < 10; class_loop = class_loop + 1) begin
                timeout_cycles = 0;
                while (!result_valid) begin
                    @(negedge clk); timeout_cycles = timeout_cycles + 1;
                    if (timeout_cycles > 600) $fatal(1, "readout timeout");
                end
                scan_count = $fscanf(expected_file, "%d %d %d %d\n",
                    expected_case, expected_class, expected_current, expected_score);
                if (class_loop == 0 || expected_score > expected_best_score) begin
                    expected_best_score = expected_score;
                    expected_best_class = class_loop;
                end
                if (scan_count != 4 || expected_case != case_loop ||
                    result_class != expected_class || $signed(current_out) != expected_current ||
                    $signed(score_out) != expected_score) begin
                    failures = failures + 1;
                    if (failures <= 20) $display("readout mismatch case=%0d class=%0d", case_loop, class_loop);
                end
                if (class_loop == 9 && step_integer == 3 &&
                    (!frame_done || class_out != expected_best_class)) begin
                    failures = failures + 1;
                    $display("argmax mismatch got=%0d expected=%0d", class_out, expected_best_class);
                end
                @(negedge clk);
            end
        end
        if (failures) $fatal(1, "readout failed: %0d mismatches", failures);
        $display("PASS tb_readout: %0d cases x 10 classes matched", case_count);
        $finish;
    end
endmodule
