`timescale 1ns/1ps

module tb_snn_core;
    logic clk = 0;
    logic rst_n = 0;
    logic [63:0] spike_in;
    logic step_valid = 0, frame_start = 0;
    logic ready, step_done, frame_done;
    logic [3:0] class_out;
    logic l1_valid, l1_spike;
    logic [9:0] l1_index;
    logic signed [15:0] l1_current;
    logic signed [17:0] l1_integrated, l1_membrane;
    logic l2_valid, l2_spike;
    logic [8:0] l2_index;
    logic signed [19:0] l2_current;
    logic signed [21:0] l2_integrated, l2_membrane;
    logic readout_valid;
    logic [3:0] readout_class;
    logic signed [16:0] readout_current;
    logic signed [20:0] weighted_score;
    logic [31:0] step_synaptic_additions, frame_synaptic_additions;
    logic [31:0] step_compute_cycles, frame_compute_cycles;

    integer input_file, l1_file, l2_file, readout_file, frame_file, sparse_file;
    integer scan_count, step_count, step_loop, input_step, frame_integer;
    integer expected_step, expected_index, expected_current, expected_integrated;
    integer expected_membrane, expected_spike, expected_score;
    integer l1_count, l2_count, readout_count, timeout_cycles, done_seen;
    integer frame_count, frame_seen = 0, expected_last_step, expected_class;
    integer expected_l1_additions, expected_l2_additions, expected_readout_additions;
    integer expected_step_additions, expected_l1_cycles, expected_l2_cycles;
    integer expected_readout_cycles, expected_step_cycles;
    integer expected_frame_additions, expected_frame_cycles;
    integer failures = 0;
    reg [63:0] spike_word;

    always #5 clk = ~clk;

    onn_snn_core #(
        .CONV1_WEIGHT_FILE("../../model/snn/export/conv1_weight.mem"),
        .CONV2_WEIGHT_FILE("../../model/snn/export/conv2_weight.mem"),
        .READOUT_WEIGHT_FILE("../../model/snn/export/readout_weight.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .spike_in(spike_in),
        .step_valid(step_valid), .frame_start(frame_start), .ready(ready),
        .step_done(step_done), .frame_done(frame_done), .class_out(class_out),
        .l1_valid(l1_valid), .l1_index(l1_index), .l1_current(l1_current),
        .l1_integrated(l1_integrated), .l1_membrane(l1_membrane), .l1_spike(l1_spike),
        .l2_valid(l2_valid), .l2_index(l2_index), .l2_current(l2_current),
        .l2_integrated(l2_integrated), .l2_membrane(l2_membrane), .l2_spike(l2_spike),
        .readout_valid(readout_valid), .readout_class(readout_class),
        .readout_current(readout_current), .weighted_score(weighted_score),
        .step_synaptic_additions(step_synaptic_additions),
        .frame_synaptic_additions(frame_synaptic_additions),
        .step_compute_cycles(step_compute_cycles),
        .frame_compute_cycles(frame_compute_cycles)
    );

    initial begin
        input_file = $fopen("../vectors/core_inputs.txt", "r");
        l1_file = $fopen("../vectors/core_l1_expected.txt", "r");
        l2_file = $fopen("../vectors/core_l2_expected.txt", "r");
        readout_file = $fopen("../vectors/core_readout_expected.txt", "r");
        frame_file = $fopen("../vectors/core_frames.txt", "r");
        sparse_file = $fopen("../vectors/core_sparse_expected.txt", "r");
        if (!input_file || !l1_file || !l2_file || !readout_file ||
            !frame_file || !sparse_file)
            $fatal(1, "cannot open core vectors");
        scan_count = $fscanf(input_file, "%d\n", step_count);
        scan_count = $fscanf(frame_file, "%d\n", frame_count);
        repeat (3) @(negedge clk); rst_n = 1;

        for (step_loop = 0; step_loop < step_count; step_loop = step_loop + 1) begin
            scan_count = $fscanf(input_file, "%d %d %h\n",
                input_step, frame_integer, spike_word);
            if (scan_count != 3 || input_step != step_loop) $fatal(1, "bad core input");
            while (!ready) @(negedge clk);
            spike_in = spike_word; frame_start = frame_integer; step_valid = 1;
            @(negedge clk); step_valid = 0; frame_start = 0;
            l1_count = 0; l2_count = 0; readout_count = 0;
            timeout_cycles = 0; done_seen = 0;
            while (!done_seen) begin
                // Deliberately disturb the external request while busy. The core
                // must ignore it because ready is low and retain the accepted input.
                if (step_loop == 0 && timeout_cycles == 5) begin
                    if (ready) begin
                        failures = failures + 1;
                        $display("core unexpectedly ready while processing");
                    end
                    step_valid = 1;
                    frame_start = 1;
                    spike_in = 64'hDEADBEEF01234567;
                end else if (step_loop == 0 && timeout_cycles == 6) begin
                    step_valid = 0;
                    frame_start = 0;
                    spike_in = spike_word;
                end
                if (l1_valid) begin
                    scan_count = $fscanf(l1_file, "%d %d %d %d %d %d\n",
                        expected_step, expected_index, expected_current,
                        expected_integrated, expected_membrane, expected_spike);
                    if (scan_count != 6 || expected_step != step_loop ||
                        l1_index != expected_index || $signed(l1_current) != expected_current ||
                        $signed(l1_integrated) != expected_integrated ||
                        $signed(l1_membrane) != expected_membrane || l1_spike != expected_spike) begin
                        failures = failures + 1;
                        if (failures <= 20) $display("core L1 mismatch step=%0d index=%0d", step_loop, l1_count);
                    end
                    l1_count = l1_count + 1;
                end
                if (l2_valid) begin
                    scan_count = $fscanf(l2_file, "%d %d %d %d %d %d\n",
                        expected_step, expected_index, expected_current,
                        expected_integrated, expected_membrane, expected_spike);
                    if (scan_count != 6 || expected_step != step_loop ||
                        l2_index != expected_index || $signed(l2_current) != expected_current ||
                        $signed(l2_integrated) != expected_integrated ||
                        $signed(l2_membrane) != expected_membrane || l2_spike != expected_spike) begin
                        failures = failures + 1;
                        if (failures <= 20) $display("core L2 mismatch step=%0d index=%0d", step_loop, l2_count);
                    end
                    l2_count = l2_count + 1;
                end
                if (readout_valid) begin
                    scan_count = $fscanf(readout_file, "%d %d %d %d\n",
                        expected_step, expected_index, expected_current, expected_score);
                    if (scan_count != 4 || expected_step != step_loop ||
                        readout_class != expected_index ||
                        $signed(readout_current) != expected_current ||
                        $signed(weighted_score) != expected_score) begin
                        failures = failures + 1;
                        if (failures <= 20) $display("core readout mismatch step=%0d class=%0d", step_loop, readout_count);
                    end
                    readout_count = readout_count + 1;
                end
                if (step_done) begin
                    done_seen = 1;
                    if ((step_loop % 4) == 3) begin
                        scan_count = $fscanf(frame_file, "%d %d\n", expected_last_step, expected_class);
                        if (scan_count != 2 || expected_last_step != step_loop ||
                            !frame_done || class_out != expected_class) begin
                            failures = failures + 1;
                            $display("core frame mismatch step=%0d got=%0d expected=%0d",
                                     step_loop, class_out, expected_class);
                        end
                        frame_seen = frame_seen + 1;
                    end else if (frame_done) begin
                        failures = failures + 1;
                        $display("unexpected frame_done at step=%0d", step_loop);
                    end
                end else begin
                    @(negedge clk);
                    timeout_cycles = timeout_cycles + 1;
                    if (timeout_cycles > 100000) $fatal(1, "core timeout at step %0d", step_loop);
                end
            end
            if (l1_count != 1024 || l2_count != 512 || readout_count != 10) begin
                $fatal(1, "wrong stream counts at step %0d: %0d/%0d/%0d",
                       step_loop, l1_count, l2_count, readout_count);
            end
            @(negedge clk);
            scan_count = $fscanf(sparse_file, "%d %d %d %d %d %d %d %d %d %d %d\n",
                expected_step, expected_l1_additions, expected_l2_additions,
                expected_readout_additions, expected_step_additions,
                expected_l1_cycles, expected_l2_cycles, expected_readout_cycles,
                expected_step_cycles, expected_frame_additions, expected_frame_cycles);
            if (scan_count != 11 || expected_step != step_loop ||
                step_synaptic_additions != expected_step_additions ||
                frame_synaptic_additions != expected_frame_additions ||
                step_compute_cycles != expected_step_cycles ||
                frame_compute_cycles != expected_frame_cycles) begin
                failures = failures + 1;
                $display(
                    "sparse counter mismatch step=%0d got add/cycle=%0d/%0d frame=%0d/%0d expected=%0d/%0d frame=%0d/%0d",
                    step_loop, step_synaptic_additions, step_compute_cycles,
                    frame_synaptic_additions, frame_compute_cycles,
                    expected_step_additions, expected_step_cycles,
                    expected_frame_additions, expected_frame_cycles
                );
            end
            if (step_synaptic_additions >= 88064) begin
                failures = failures + 1;
                $display("sparse step did not reduce work at step=%0d", step_loop);
            end
        end
        if (frame_seen != frame_count) $fatal(1, "wrong frame count");
        if (failures) $fatal(1, "full core failed: %0d mismatches", failures);
        $display("PASS tb_snn_core: %0d timesteps, %0d frames matched end-to-end",
                 step_count, frame_count);
        $finish;
    end
endmodule
