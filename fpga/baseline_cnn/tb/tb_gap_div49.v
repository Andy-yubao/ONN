// tb_gap_div49.v - GAP verification: exhaustive + golden trace.
//
//   1) Exhaustive: sum = 0..12495 against the exact integer reference
//      expected = (sum + 24) / 49, saturated to [0,255]  (12496 items).
//   2) Golden trace: per-channel sums of conv3_q (32 channels x 7x7) driven
//      through gap_div49 and compared bit-exact with gap_q.mem (32 items).
//
// Any mismatch terminates with $fatal (non-zero vsim exit code); a clean run
// prints GAP_ALL_PASS and $finish (exit 0).
`timescale 1ns/1ps

module tb_gap_div49;
    reg  [13:0] sum;
    wire [7:0]  q;

    gap_div49 dut (.*);

    // Golden memories.
    reg [7:0] conv3_q [0:1567];
    reg [7:0] gap_q   [0:31];

    integer i, c, s, rawsum, expected, mismatches, total;

    initial begin
        // Paths are relative to the repo root (run_questa.ps1 cd's there).
        // $readmemh is a task (no return value), so loading is guarded with
        // sentinel checks against known frozen values: a missing file leaves
        // the memory all zeros, the sentinel fails, and the run aborts.
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_q.mem", conv3_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/gap_q.mem",   gap_q);

        if (conv3_q[4] !== 8'h09) $fatal(1, "GAP: conv3_q.mem not loaded (conv3_q[4]=%h)", conv3_q[4]);
        if (gap_q[0]   !== 8'h04) $fatal(1, "GAP: gap_q.mem not loaded (gap_q[0]=%h)", gap_q[0]);

        mismatches = 0;
        total = 0;

        // ---- exhaustive 0..12495 ----
        for (s = 0; s <= 12495; s = s + 1) begin
            sum      = s[13:0];
            expected = (s + 24) / 49;
            if (expected > 255) expected = 255;
            total = total + 1;
            #1;
            if (q !== expected[7:0]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10)
                    $display("EXH MISMATCH sum=%0d q=%0d expect=%0d", s, q, expected);
            end
            #1;
        end

        // ---- golden trace: 32 channels ----
        for (c = 0; c < 32; c = c + 1) begin
            rawsum = 0;
            for (i = 0; i < 49; i = i + 1)
                rawsum = rawsum + conv3_q[c*49 + i];
            sum = rawsum[13:0];
            total = total + 1;
            #1;
            if (q !== gap_q[c]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10)
                    $display("TRACE MISMATCH ch=%0d sum=%0d q=%0d expect=%0d",
                             c, rawsum, q, gap_q[c]);
            end
            #1;
        end

        $display("GAP: total=%0d mismatches=%0d", total, mismatches);
        if (mismatches == 0) begin
            $display("GAP_ALL_PASS");
            $finish;
        end else begin
            $fatal(1, "GAP: %0d mismatches out of %0d", mismatches, total);
        end
    end
endmodule
