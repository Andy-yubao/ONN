`timescale 1ns/1ps

module tb_conv2_if2;
    logic clk = 0;
    logic rst_n = 0;
    logic [1023:0] spike_in;
    logic step_valid = 0;
    logic frame_start = 0;
    logic ready, result_valid, spike_out, step_done;
    logic [8:0] result_index;
    logic signed [19:0] current_out;
    logic signed [21:0] integrated_out, membrane_out;
    integer input_file, expected_file, scan_count, case_count, case_loop;
    integer case_id, frame_integer, element_loop, timeout_cycles, failures = 0;
    integer expected_case, expected_index, expected_current;
    integer expected_integrated, expected_membrane, expected_spike;
    reg [1023:0] spike_word;

    always #5 clk = ~clk;

    onn_conv2_if2 #(
        .CONV2_WEIGHT_FILE("../../model/snn/export/conv2_weight.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .spike_in(spike_in),
        .step_valid(step_valid), .frame_start(frame_start), .ready(ready),
        .result_valid(result_valid), .result_index(result_index),
        .current_out(current_out), .integrated_out(integrated_out),
        .membrane_out(membrane_out), .spike_out(spike_out), .step_done(step_done)
    );

    initial begin
        input_file = $fopen("../vectors/conv2_inputs.txt", "r");
        expected_file = $fopen("../vectors/conv2_expected.txt", "r");
        if (!input_file || !expected_file) $fatal(1, "cannot open Conv2 vectors");
        scan_count = $fscanf(input_file, "%d\n", case_count);
        repeat (3) @(negedge clk); rst_n = 1;
        for (case_loop = 0; case_loop < case_count; case_loop = case_loop + 1) begin
            scan_count = $fscanf(input_file, "%d %d %h\n", case_id, frame_integer, spike_word);
            if (scan_count != 3 || case_id != case_loop) $fatal(1, "bad Conv2 input");
            while (!ready) @(negedge clk);
            spike_in = spike_word; frame_start = frame_integer; step_valid = 1;
            @(negedge clk); step_valid = 0; frame_start = 0;
            for (element_loop = 0; element_loop < 512; element_loop = element_loop + 1) begin
                timeout_cycles = 0;
                while (!result_valid) begin
                    @(negedge clk); timeout_cycles = timeout_cycles + 1;
                    if (timeout_cycles > 800) $fatal(1, "Conv2 timeout");
                end
                scan_count = $fscanf(expected_file, "%d %d %d %d %d %d\n",
                    expected_case, expected_index, expected_current,
                    expected_integrated, expected_membrane, expected_spike);
                if (scan_count != 6) $fatal(1, "bad Conv2 expected vector");
                if (expected_case != case_loop || expected_index != element_loop ||
                    result_index != expected_index || $signed(current_out) != expected_current ||
                    $signed(integrated_out) != expected_integrated ||
                    $signed(membrane_out) != expected_membrane || spike_out != expected_spike) begin
                    failures = failures + 1;
                    if (failures <= 20) $display("Conv2 mismatch case=%0d index=%0d", case_loop, element_loop);
                end
                @(negedge clk);
            end
        end
        if (failures) $fatal(1, "Conv2+IF2 failed: %0d mismatches", failures);
        $display("PASS tb_conv2_if2: %0d cases x 512 elements matched", case_count);
        $finish;
    end
endmodule
