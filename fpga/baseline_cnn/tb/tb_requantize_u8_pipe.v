// tb_requantize_u8_pipe.v - Fixed-latency and bit-exact verification of the
// three-stage pipelined requant unit.
`timescale 1ns/1ps

module tb_requantize_u8_pipe;
    localparam PIPE_STAGES = 3;
    localparam GOLDEN_TOKENS = 20384;
    localparam signed [31:0] STEM_MULT  = 32'sd2056884242;
    localparam signed [31:0] CONV2_MULT = 32'sd1097020857;
    localparam signed [31:0] CONV3_MULT = 32'sd1298974956;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst_n = 1'b0;
    reg in_valid = 1'b0;
    reg signed [63:0] in_acc64 = 64'sd0;
    reg signed [31:0] in_multiplier = 32'sd0;
    wire out_valid;
    wire signed [31:0] out_acc32;
    wire [7:0] out_q;

    requantize_u8_pipe #(.SHIFT(6'd38)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
        .in_acc64(in_acc64), .in_multiplier(in_multiplier),
        .out_valid(out_valid), .out_acc32(out_acc32), .out_q(out_q)
    );

    // Frozen combinational oracle, fed with the same saturated acc32 token.
    wire signed [31:0] oracle_acc = (in_acc64 > 64'sd2147483647) ? 32'sh7FFFFFFF :
                                      (in_acc64 < -64'sd2147483648) ? 32'sh80000000 :
                                      in_acc64[31:0];
    wire [7:0] oracle_q;
    requantize_u8 oracle (
        .acc(oracle_acc), .multiplier(in_multiplier), .shift(6'd38), .q(oracle_q)
    );

    // Separate SHIFT=0 instance covers the identity-shift edge contract.
    reg zero_in_valid = 1'b0;
    reg signed [63:0] zero_in_acc64 = 64'sd0;
    reg signed [31:0] zero_in_multiplier = 32'sd0;
    wire zero_out_valid;
    wire signed [31:0] zero_out_acc32;
    wire [7:0] zero_out_q;
    requantize_u8_pipe #(.SHIFT(6'd0)) dut_shift0 (
        .clk(clk), .rst_n(rst_n), .in_valid(zero_in_valid),
        .in_acc64(zero_in_acc64), .in_multiplier(zero_in_multiplier),
        .out_valid(zero_out_valid), .out_acc32(zero_out_acc32), .out_q(zero_out_q)
    );
    wire signed [31:0] zero_oracle_acc =
        (zero_in_acc64 > 64'sd2147483647) ? 32'sh7FFFFFFF :
        (zero_in_acc64 < -64'sd2147483648) ? 32'sh80000000 :
        zero_in_acc64[31:0];
    wire [7:0] zero_oracle_q;
    requantize_u8 zero_oracle (
        .acc(zero_oracle_acc), .multiplier(zero_in_multiplier), .shift(6'd0),
        .q(zero_oracle_q)
    );

    reg [31:0] stem_acc [0:12543];
    reg [7:0]  stem_q   [0:12543];
    reg [31:0] conv2_acc[0:6271];
    reg [7:0]  conv2_q  [0:6271];
    reg [31:0] conv3_acc[0:1567];
    reg [7:0]  conv3_q  [0:1567];

    reg signed [31:0] exp_acc [0:32767];
    reg [7:0] exp_q [0:32767];
    integer exp_cycle [0:32767];
    integer wr_ptr, rd_ptr;
    reg [PIPE_STAGES-1:0] valid_history;

    reg signed [31:0] zero_exp_acc [0:31];
    reg [7:0] zero_exp_q [0:31];
    integer zero_exp_cycle [0:31];
    integer zero_wr_ptr, zero_rd_ptr;
    reg [PIPE_STAGES-1:0] zero_valid_history;

    integer cyc, checked, zero_checked, latency_bad, align_bad, xz_bad;
    integer golden_oracle_bad, reset_leak_bad;
    integer i;
    reg [31:0] lfsr;

    // Scoreboards explicitly model the three registered boundaries.  A token
    // accepted on the first boundary must be visible after the third boundary:
    // (current_cycle - accept_cycle + 1) == PIPE_STAGES.
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_history = {PIPE_STAGES{1'b0}};
            zero_valid_history = {PIPE_STAGES{1'b0}};
            wr_ptr = 0; rd_ptr = 0;
            zero_wr_ptr = 0; zero_rd_ptr = 0;
            #1;
            if (out_valid || zero_out_valid) reset_leak_bad = reset_leak_bad + 1;
        end else begin
            cyc = cyc + 1;
            valid_history = {valid_history[PIPE_STAGES-2:0], in_valid};
            zero_valid_history = {zero_valid_history[PIPE_STAGES-2:0], zero_in_valid};

            if (in_valid) begin
                exp_acc[wr_ptr] = oracle_acc;
                exp_q[wr_ptr] = oracle_q;
                exp_cycle[wr_ptr] = cyc;
                wr_ptr = wr_ptr + 1;
            end
            if (zero_in_valid) begin
                zero_exp_acc[zero_wr_ptr] = zero_oracle_acc;
                zero_exp_q[zero_wr_ptr] = zero_oracle_q;
                zero_exp_cycle[zero_wr_ptr] = cyc;
                zero_wr_ptr = zero_wr_ptr + 1;
            end

            #1;
            if (out_valid !== valid_history[PIPE_STAGES-1]) latency_bad = latency_bad + 1;
            if (zero_out_valid !== zero_valid_history[PIPE_STAGES-1]) latency_bad = latency_bad + 1;
            if (out_valid) begin
                if ((cyc - exp_cycle[rd_ptr] + 1) != PIPE_STAGES) latency_bad = latency_bad + 1;
                if (out_acc32 !== exp_acc[rd_ptr] || out_q !== exp_q[rd_ptr]) align_bad = align_bad + 1;
                if (^out_acc32 === 1'bx || ^out_q === 1'bx) xz_bad = xz_bad + 1;
                rd_ptr = rd_ptr + 1;
                checked = checked + 1;
            end
            if (zero_out_valid) begin
                if ((cyc - zero_exp_cycle[zero_rd_ptr] + 1) != PIPE_STAGES) latency_bad = latency_bad + 1;
                if (zero_out_acc32 !== zero_exp_acc[zero_rd_ptr] || zero_out_q !== zero_exp_q[zero_rd_ptr])
                    align_bad = align_bad + 1;
                if (^zero_out_acc32 === 1'bx || ^zero_out_q === 1'bx) xz_bad = xz_bad + 1;
                zero_rd_ptr = zero_rd_ptr + 1;
                zero_checked = zero_checked + 1;
            end
        end
    end

    task drive_main;
        input signed [63:0] a;
        input signed [31:0] m;
        input [7:0] golden;
        input check_golden;
        begin
            @(negedge clk);
            in_acc64 = a;
            in_multiplier = m;
            in_valid = 1'b1;
            #1;
            if (check_golden && oracle_q !== golden) golden_oracle_bad = golden_oracle_bad + 1;
        end
    endtask

    task drive_gap;
        begin
            @(negedge clk);
            in_valid = 1'b0;
            in_acc64 = 64'shx;
            in_multiplier = 32'shx;
        end
    endtask

    task drive_zero;
        input signed [63:0] a;
        input signed [31:0] m;
        begin
            @(negedge clk);
            zero_in_acc64 = a;
            zero_in_multiplier = m;
            zero_in_valid = 1'b1;
        end
    endtask

    initial begin : watchdog
        repeat (200000) @(posedge clk);
        $fatal(1, "REQUANT_PIPE: TIMEOUT");
    end

    initial begin
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv1_acc.mem", stem_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/stem_q.mem", stem_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv2_acc.mem", conv2_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv2_q.mem", conv2_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_acc.mem", conv3_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_q.mem", conv3_q);
        if (stem_acc[0] !== 32'h000012DC || stem_q[0] !== 8'h24 ||
            conv2_acc[0] !== 32'h00000F8A || conv2_q[0] !== 8'h10 ||
            conv3_acc[0] !== 32'hFFFFF7E6 || conv3_q[4] !== 8'h09)
            $fatal(1, "REQUANT_PIPE: golden vectors not loaded");

        cyc = 0; checked = 0; zero_checked = 0;
        latency_bad = 0; align_bad = 0; xz_bad = 0;
        golden_oracle_bad = 0; reset_leak_bad = 0;
        wr_ptr = 0; rd_ptr = 0; zero_wr_ptr = 0; zero_rd_ptr = 0;
        valid_history = 0; zero_valid_history = 0; lfsr = 32'h1ACE_B00C;

        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        // Single token, then a fully drained pipeline.
        drive_main(64'sd4828, STEM_MULT, 8'h24, 1'b1);
        drive_gap();
        repeat (3) @(posedge clk);

        // Continuous tokens, multiplier changes every token.
        drive_main(64'sd1,  32'sd1, 8'd0, 1'b0);
        drive_main(-64'sd1, 32'sd1, 8'd0, 1'b0);
        drive_main(-64'sd5, -32'sd1073741824, 8'd0, 1'b0);
        drive_main(64'sd5,  -32'sd1073741824, 8'd0, 1'b0);
        drive_main(64'sd2147483648, 32'sd32768, 8'd0, 1'b0);   // INT32_MAX clamp
        drive_main(-64'sd2147483649, 32'sd32768, 8'd0, 1'b0); // INT32_MIN clamp
        // Negative half boundary: just below half and exactly half.
        drive_main(-64'sd128, 32'sd1073741823, 8'd0, 1'b0);
        drive_main(-64'sd128, 32'sd1073741824, 8'd0, 1'b0);
        drive_gap();
        repeat (3) @(posedge clk);

        // Mid-pipeline reset must discard every outstanding valid token.
        drive_main(64'sd1234, CONV2_MULT, 8'd0, 1'b0);
        drive_main(64'sd5678, CONV3_MULT, 8'd0, 1'b0);
        @(negedge clk); rst_n = 1'b0; in_valid = 1'b0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (3) @(posedge clk);

        // SHIFT=0 covers pre-saturation signed values -1/0/255/256 and both
        // multiplier signs.  The oracle checks the exact UINT8 neighborhood.
        drive_zero(-64'sd1,  32'sd1);
        drive_zero(64'sd0,   32'sd1);
        drive_zero(64'sd255, 32'sd1);
        drive_zero(64'sd256, 32'sd1);
        drive_zero(-64'sd5, -32'sd1);
        drive_zero(64'sd5,  -32'sd1);
        @(negedge clk); zero_in_valid = 1'b0;
        repeat (4) @(posedge clk);

        // All 20,384 frozen tokens with deterministic pseudo-random gaps.
        for (i = 0; i < 12544; i = i + 1) begin
            lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            if (!lfsr[0]) drive_gap();
            drive_main($signed(stem_acc[i]), STEM_MULT, stem_q[i], 1'b1);
        end
        for (i = 0; i < 6272; i = i + 1) begin
            lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            if (!lfsr[0]) drive_gap();
            drive_main($signed(conv2_acc[i]), CONV2_MULT, conv2_q[i], 1'b1);
        end
        for (i = 0; i < 1568; i = i + 1) begin
            lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            if (!lfsr[0]) drive_gap();
            drive_main($signed(conv3_acc[i]), CONV3_MULT, conv3_q[i], 1'b1);
        end
        drive_gap();
        repeat (5) @(posedge clk);

        $display("REQUANT_PIPE: checked=%0d zero_checked=%0d latency_bad=%0d align_bad=%0d xz_bad=%0d golden_oracle_bad=%0d reset_leak_bad=%0d",
                 checked, zero_checked, latency_bad, align_bad, xz_bad,
                 golden_oracle_bad, reset_leak_bad);
        if (checked != GOLDEN_TOKENS + 9 || zero_checked != 6 || latency_bad != 0 ||
            align_bad != 0 || xz_bad != 0 || golden_oracle_bad != 0 ||
            reset_leak_bad != 0 || rd_ptr != wr_ptr || zero_rd_ptr != zero_wr_ptr)
            $fatal(1, "REQUANT_PIPE: FAILED");

        $display("REQUANT_PIPE_ALL_PASS");
        $finish;
    end
endmodule
