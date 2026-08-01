// tb_requantize_u8.v - Golden-vector verification of requantize_u8.
//
// Compares the module against the frozen Int8Reference golden trace
// (fpga/baseline_cnn/sim/vectors/golden_trace):
//     conv1_acc -> stem_q    (12544)
//     conv2_acc -> conv2_q   ( 6272)
//     conv3_acc -> conv3_q   ( 1568)
// total 20384 comparisons; every one must be bit-exact.
//
// $readmemh paths are relative to the repo root (run_questa.ps1 cd's there).
// Any mismatch terminates with $fatal (non-zero vsim exit code); a clean
// run prints REQUANT_ALL_PASS and $finish (exit 0).
`timescale 1ns/1ps

module tb_requantize_u8;
    reg  signed [31:0] acc;
    reg  signed [31:0] multiplier;
    reg         [5:0]  shift;
    wire        [7:0]  q;

    requantize_u8 dut (.*);

    // Layer constants (frozen, see params/baseline_cnn_params.vh).
    localparam STEM_MULT   = 32'sd2056884242;
    localparam STEM_SHIFT  = 6'd38;
    localparam CONV2_MULT  = 32'sd1097020857;
    localparam CONV2_SHIFT = 6'd38;
    localparam CONV3_MULT  = 32'sd1298974956;
    localparam CONV3_SHIFT = 6'd38;

    // Golden memories.
    reg [31:0] stem_acc  [0:12543];
    reg [7:0]  stem_q    [0:12543];
    reg [31:0] conv2_acc [0:6271];
    reg [7:0]  conv2_q   [0:6271];
    reg [31:0] conv3_acc [0:1567];
    reg [7:0]  conv3_q   [0:1567];

    integer i, mismatches, total;

    initial begin
        // Paths are relative to the repo root (run_questa.ps1 cd's there).
        // $readmemh is a task (no return value), so loading is guarded with
        // sentinel checks against known frozen values: if a file was not
        // loaded the memory is all zeros, the sentinel fails, and the run
        // aborts instead of silently passing.
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv1_acc.mem", stem_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/stem_q.mem",    stem_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv2_acc.mem", conv2_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv2_q.mem",   conv2_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_acc.mem", conv3_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_q.mem",   conv3_q);

        if (stem_acc[0]  !== 32'h000012DC) $fatal(1, "REQUANT: conv1_acc.mem not loaded (stem_acc[0]=%h)", stem_acc[0]);
        if (stem_q[0]    !== 8'h24)        $fatal(1, "REQUANT: stem_q.mem not loaded (stem_q[0]=%h)", stem_q[0]);
        if (conv2_acc[0] !== 32'h00000F8A) $fatal(1, "REQUANT: conv2_acc.mem not loaded (conv2_acc[0]=%h)", conv2_acc[0]);
        if (conv2_q[0]   !== 8'h10)        $fatal(1, "REQUANT: conv2_q.mem not loaded (conv2_q[0]=%h)", conv2_q[0]);
        if (conv3_acc[0] !== 32'hFFFFF7E6) $fatal(1, "REQUANT: conv3_acc.mem not loaded (conv3_acc[0]=%h)", conv3_acc[0]);
        if (conv3_q[4]   !== 8'h09)        $fatal(1, "REQUANT: conv3_q.mem not loaded (conv3_q[4]=%h)", conv3_q[4]);

        mismatches = 0;
        total = 0;

        // ---- stem: conv1_acc -> stem_q (12544) ----
        multiplier = STEM_MULT;
        shift      = STEM_SHIFT;
        for (i = 0; i < 12544; i = i + 1) begin
            acc   = $signed(stem_acc[i]);
            total = total + 1;
            #1;
            if (q !== stem_q[i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10)
                    $display("STEM  MISMATCH i=%0d acc=%0d q=%0d expect=%0d",
                             i, $signed(acc), q, stem_q[i]);
            end
            #1;
        end

        // ---- conv2: conv2_acc -> conv2_q (6272) ----
        multiplier = CONV2_MULT;
        shift      = CONV2_SHIFT;
        for (i = 0; i < 6272; i = i + 1) begin
            acc   = $signed(conv2_acc[i]);
            total = total + 1;
            #1;
            if (q !== conv2_q[i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10)
                    $display("CONV2 MISMATCH i=%0d acc=%0d q=%0d expect=%0d",
                             i, $signed(acc), q, conv2_q[i]);
            end
            #1;
        end

        // ---- conv3: conv3_acc -> conv3_q (1568) ----
        multiplier = CONV3_MULT;
        shift      = CONV3_SHIFT;
        for (i = 0; i < 1568; i = i + 1) begin
            acc   = $signed(conv3_acc[i]);
            total = total + 1;
            #1;
            if (q !== conv3_q[i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10)
                    $display("CONV3 MISMATCH i=%0d acc=%0d q=%0d expect=%0d",
                             i, $signed(acc), q, conv3_q[i]);
            end
            #1;
        end

        $display("REQUANT: total=%0d mismatches=%0d", total, mismatches);
        if (mismatches == 0) begin
            $display("REQUANT_ALL_PASS");
            $finish;
        end else begin
            $fatal(1, "REQUANT: %0d mismatches out of %0d", mismatches, total);
        end
    end
endmodule
