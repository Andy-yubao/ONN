// tb_stem_conv_padding.v - Boundary/padding专项 verification of stem_conv_serial.
//
// Loads a CANARY input (input[addr] = addr&0xFF) so any illegal or wrong-address
// read changes the accumulator and is caught.  It then verifies, by internal
// counters over the DUT's own ACC state, that every out-of-bounds tap:
//   * never produces an illegal RAM address (in_raddr <= 783), and
//   * contributes 0 to the accumulator (valid-tap counts below).
//
// Targets (oc=0):  corners 4 taps, edges 6 taps, centre 9 taps.
//   (0,0,0)=idx0,  (0,0,27)=idx27,  (0,27,0)=idx756,  (0,27,27)=idx783
//   edge (0,0,14)=idx14, edge (0,14,0)=idx392, centre (0,14,14)=idx406
//
// Additionally the corner (0,0,0) accumulator is checked against a hand-built
// expected value from the 4 valid taps (proving out-of-bounds taps really
// contribute nothing), and done/busy timing is re-verified.
//
// Any failure prints diagnostics and $fatal (non-zero vsim exit).
`timescale 1ns/1ps

module tb_stem_conv_padding;
    reg clk = 1'b0;
    always #5 clk = ~clk;             // 10 ns period

    reg              rst_n = 1'b0;
    reg              input_we = 1'b0;
    reg [9:0]        input_waddr;
    reg signed [7:0] input_wdata;
    reg              start = 1'b0;
    wire             busy;
    wire             done;
    wire             acc_valid;
    wire [13:0]      acc_addr;
    wire signed [31:0] acc_value;
    wire             q_valid;
    wire [13:0]      q_addr;
    wire [7:0]       q_value;
    reg [13:0]       output_raddr = 14'd0;
    wire [7:0]       output_rdata;

    stem_conv_serial dut (.*);

    // canary + weights/bias for the corner functional check
    reg [7:0]  canary [0:783];
    reg [7:0]  wt     [0:143];
    reg [31:0] bias   [0:15];

    // ---- targets (oc=0) ----
    integer k, cur;
    integer target[0:6];
    integer valid_cnt[0:6];
    integer tap_seen[0:6];
    localparam integer T0 = 0;        // corner (0,0,0)    expect 4
    localparam integer T1 = 27;       // corner (0,0,27)   expect 4
    localparam integer T2 = 756;      // corner (0,27,0)   expect 4
    localparam integer T3 = 783;      // corner (0,27,27)  expect 4
    localparam integer T4 = 14;       // edge   (0,0,14)   expect 6
    localparam integer T5 = 392;      // edge   (0,14,0)   expect 6
    localparam integer T6 = 406;      // centre (0,14,14)  expect 9

    integer i, cyc, busy_cycles, done_count, illegal_addr;
    integer corner_acc_seen, corner_acc_ok;
    integer corner_expected;

    // monitor the DUT internals every clock edge
    always @(posedge clk) begin
        #1;
        cyc = cyc + 1;
        if (busy) busy_cycles = busy_cycles + 1;
        if (done) done_count = done_count + 1;

        // out-of-bounds taps must never read an illegal input address
        if (busy && (dut.in_raddr > 10'd783)) begin
            illegal_addr = illegal_addr + 1;
            if (illegal_addr <= 10)
                $display("PAD: ILLEGAL in_raddr=%0d (oc=%0d y=%0d x=%0d tap=%0d)",
                         dut.in_raddr, dut.oc, dut.y, dut.x, dut.tap);
        end

        // count valid taps per target while the DUT is in S_ACC
        if (dut.state == 3'd2) begin       // S_ACC
            cur = (dut.oc * 5'd28 + dut.y) * 5'd28 + dut.x;
            for (k = 0; k < 7; k = k + 1) begin
                if (cur == target[k]) begin
                    tap_seen[k] = tap_seen[k] + 1;
                    if (dut.tap_valid_r) valid_cnt[k] = valid_cnt[k] + 1;
                end
            end
        end

        // functional corner check: capture acc_value when acc_addr == 0
        if (acc_valid && acc_addr == 14'd0) begin
            corner_acc_seen = corner_acc_seen + 1;
            // expected = input[0]*w4 + input[1]*w5 + input[28]*w7 + input[29]*w8 + bias[0]
            // (taps (1,1),(1,2),(2,1),(2,2) of oc=0; the 5 out-of-bounds taps contribute 0)
            if (acc_value !== corner_expected)
                $display("PAD: corner(0,0,0) acc=%0d expected=%0d", $signed(acc_value), corner_expected);
            else
                corner_acc_ok = corner_acc_ok + 1;
        end
    end

    // watchdog: fires $fatal if the FSM hangs (done never asserted).  The
    // normal flow calls $finish, which terminates this thread first.
    initial begin : pad_watchdog
        repeat (500000) @(posedge clk);
        $fatal(1, "PAD: TIMEOUT - state machine stuck");
    end

    initial begin
        $readmemh("fpga/baseline_cnn/params/weights/stem_weight.mem", wt);
        $readmemh("fpga/baseline_cnn/params/biases/stem_bias.mem",   bias);
        // sentinels (frozen values)
        if (wt[0]   !== 8'hB6) $fatal(1, "PAD: stem_weight.mem not loaded (wt[0]=%h)", wt[0]);
        if (bias[0] !== 32'h00000132) $fatal(1, "PAD: stem_bias.mem not loaded (bias[0]=%h)", bias[0]);

        for (i = 0; i < 784; i = i + 1) canary[i] = i[7:0];   // addr&0xFF
        for (i = 0; i < 7; i = i + 1) begin
            target[i] = 0; valid_cnt[i] = 0; tap_seen[i] = 0;
        end
        target[0] = T0; target[1] = T1; target[2] = T2; target[3] = T3;
        target[4] = T4; target[5] = T5; target[6] = T6;

        // corner expected value (signed 8-bit inputs and weights)
        corner_expected = $signed(canary[0])  * $signed(wt[4]) +
                          $signed(canary[1])  * $signed(wt[5]) +
                          $signed(canary[28]) * $signed(wt[7]) +
                          $signed(canary[29]) * $signed(wt[8]) +
                          $signed(bias[0]);

        cyc = 0; busy_cycles = 0; done_count = 0; illegal_addr = 0;
        corner_acc_seen = 0; corner_acc_ok = 0;

        // reset
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // load canary input
        for (i = 0; i < 784; i = i + 1) begin
            input_we   = 1'b1;
            input_waddr = i[9:0];
            input_wdata = $signed(canary[i]);
            @(posedge clk);
        end
        input_we = 1'b0;
        @(posedge clk);

        // start
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        while (!done) @(posedge clk);
        #1;

        $display("PAD: illegal_addr_reads=%0d corner_acc_seen=%0d corner_acc_ok=%0d done=%0d busy_cycles=%0d",
                 illegal_addr, corner_acc_seen, corner_acc_ok, done_count, busy_cycles);
        for (k = 0; k < 7; k = k + 1)
            $display("PAD: target idx=%0d taps_seen=%0d valid_taps=%0d", target[k], tap_seen[k], valid_cnt[k]);

        if (illegal_addr != 0 || corner_acc_seen != 1 || corner_acc_ok != 1 ||
            done_count != 1 || busy_cycles == 0 ||
            tap_seen[0] != 9 || tap_seen[1] != 9 || tap_seen[2] != 9 || tap_seen[3] != 9 ||
            tap_seen[4] != 9 || tap_seen[5] != 9 || tap_seen[6] != 9 ||
            valid_cnt[0] != 4 || valid_cnt[1] != 4 || valid_cnt[2] != 4 || valid_cnt[3] != 4 ||
            valid_cnt[4] != 6 || valid_cnt[5] != 6 || valid_cnt[6] != 9) begin
            $fatal(1, "PAD: padding verification FAILED");
        end

        $display("PAD_ALL_PASS");
        $finish;
    end
endmodule
