`timescale 1ns/1ps

module tb_conv1_if1;
    logic clk;
    logic rst_n;
    logic [63:0] spike_in;
    logic step_valid;
    logic frame_start;
    logic ready;
    logic result_valid;
    logic [9:0] result_index;
    logic signed [15:0] current_out;
    logic signed [17:0] integrated_out;
    logic signed [17:0] membrane_out;
    logic spike_out;
    logic step_done;

    integer input_file;
    integer expected_file;
    integer scan_count;
    integer case_count;
    integer case_loop;
    integer case_id;
    integer frame_start_integer;
    reg [63:0] spike_word;
    integer element_loop;
    integer expected_case;
    integer expected_index;
    integer expected_current;
    integer expected_integrated;
    integer expected_membrane;
    integer expected_spike;
    integer failures;
    integer timeout_cycles;

    always #5 clk = ~clk;

    onn_conv1_if1 #(
        .CONV1_WEIGHT_FILE("../../model/snn/export/conv1_weight.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .spike_in(spike_in),
        .step_valid(step_valid),
        .frame_start(frame_start),
        .ready(ready),
        .result_valid(result_valid),
        .result_index(result_index),
        .current_out(current_out),
        .integrated_out(integrated_out),
        .membrane_out(membrane_out),
        .spike_out(spike_out),
        .step_done(step_done)
    );

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        spike_in = '0;
        step_valid = 1'b0;
        frame_start = 1'b0;
        failures = 0;
        input_file = $fopen("../vectors/conv1_inputs.txt", "r");
        expected_file = $fopen("../vectors/conv1_expected.txt", "r");
        if ((input_file == 0) || (expected_file == 0)) begin
            $fatal(1, "cannot open Conv1 vector files");
        end
        scan_count = $fscanf(input_file, "%d\n", case_count);
        if (scan_count != 1) begin
            $fatal(1, "invalid Conv1 case count");
        end

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        for (case_loop = 0; case_loop < case_count; case_loop = case_loop + 1) begin
            scan_count = $fscanf(
                input_file, "%d %d %h\n", case_id, frame_start_integer, spike_word
            );
            if (scan_count != 3 || case_id != case_loop) begin
                $fatal(1, "invalid Conv1 input case %0d", case_loop);
            end
            while (!ready) @(negedge clk);
            spike_in = spike_word;
            frame_start = frame_start_integer;
            step_valid = 1'b1;
            @(negedge clk);
            step_valid = 1'b0;
            frame_start = 1'b0;

            for (element_loop = 0; element_loop < 1024; element_loop = element_loop + 1) begin
                timeout_cycles = 0;
                while (!result_valid) begin
                    @(negedge clk);
                    timeout_cycles = timeout_cycles + 1;
                    if (timeout_cycles > 1100) begin
                        $fatal(1, "timeout waiting for case %0d element %0d", case_loop, element_loop);
                    end
                end
                scan_count = $fscanf(
                    expected_file,
                    "%d %d %d %d %d %d\n",
                    expected_case,
                    expected_index,
                    expected_current,
                    expected_integrated,
                    expected_membrane,
                    expected_spike
                );
                if (scan_count != 6) begin
                    $fatal(1, "invalid expected result for case %0d element %0d", case_loop, element_loop);
                end
                if ((expected_case != case_loop) ||
                    (expected_index != element_loop) ||
                    (result_index != expected_index) ||
                    ($signed(current_out) != expected_current) ||
                    ($signed(integrated_out) != expected_integrated) ||
                    ($signed(membrane_out) != expected_membrane) ||
                    (spike_out != expected_spike)) begin
                    failures = failures + 1;
                    if (failures <= 20) begin
                        $display(
                            "Conv1 mismatch case=%0d element=%0d got=(idx=%0d,c=%0d,i=%0d,m=%0d,s=%0d) expected=(c=%0d,i=%0d,m=%0d,s=%0d)",
                            case_loop, element_loop, result_index,
                            $signed(current_out), $signed(integrated_out),
                            $signed(membrane_out), spike_out,
                            expected_current, expected_integrated,
                            expected_membrane, expected_spike
                        );
                    end
                end
                @(negedge clk);
            end
        end
        $fclose(input_file);
        $fclose(expected_file);
        if (failures != 0) begin
            $fatal(1, "Conv1+IF1 bit-accurate comparison failed: %0d mismatches", failures);
        end
        $display("PASS tb_conv1_if1: %0d cases x 1024 elements matched", case_count);
        $finish;
    end
endmodule
