`timescale 1ns/1ps

module tb_if_unit;
    logic signed [31:0] current_in;
    logic signed [17:0] membrane_in;
    logic signed [15:0] current_saturated;
    logic signed [17:0] integrated;
    logic spike;
    logic signed [17:0] membrane_next;

    integer vector_file;
    integer scan_count;
    integer vector_count;
    integer vector_index;
    integer input_current;
    integer input_membrane;
    integer expected_current;
    integer expected_integrated;
    integer expected_membrane;
    integer expected_spike;
    integer failures;

    onn_if_unit #(
        .CURRENT_IN_BITS(32),
        .CURRENT_BITS(16),
        .MEM_BITS(18),
        .THRESHOLD(939)
    ) dut (
        .current_in(current_in),
        .membrane_in(membrane_in),
        .current_saturated(current_saturated),
        .integrated(integrated),
        .spike(spike),
        .membrane_next(membrane_next)
    );

    initial begin
        failures = 0;
        vector_file = $fopen("../vectors/if_vectors.txt", "r");
        if (vector_file == 0) begin
            $fatal(1, "cannot open ../vectors/if_vectors.txt");
        end
        scan_count = $fscanf(vector_file, "%d\n", vector_count);
        if (scan_count != 1) begin
            $fatal(1, "invalid IF vector count");
        end
        for (vector_index = 0; vector_index < vector_count; vector_index = vector_index + 1) begin
            scan_count = $fscanf(
                vector_file,
                "%d %d %d %d %d %d\n",
                input_current,
                input_membrane,
                expected_current,
                expected_integrated,
                expected_membrane,
                expected_spike
            );
            if (scan_count != 6) begin
                $fatal(1, "invalid IF vector at index %0d", vector_index);
            end
            current_in = input_current;
            membrane_in = input_membrane;
            #1;
            if (($signed(current_saturated) != expected_current) ||
                ($signed(integrated) != expected_integrated) ||
                ($signed(membrane_next) != expected_membrane) ||
                (spike != expected_spike)) begin
                failures = failures + 1;
                $display(
                    "IF mismatch %0d: in=(%0d,%0d) got=(%0d,%0d,%0d,%0d) expected=(%0d,%0d,%0d,%0d)",
                    vector_index, input_current, input_membrane,
                    $signed(current_saturated), $signed(integrated),
                    $signed(membrane_next), spike,
                    expected_current, expected_integrated,
                    expected_membrane, expected_spike
                );
            end
        end
        $fclose(vector_file);
        if (failures != 0) begin
            $fatal(1, "IF bit-accurate comparison failed: %0d mismatches", failures);
        end
        $display("PASS tb_if_unit: %0d vectors matched", vector_count);
        $finish;
    end
endmodule
