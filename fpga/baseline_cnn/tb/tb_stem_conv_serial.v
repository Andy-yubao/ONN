// tb_stem_conv_serial.v - Golden-vector verification of the serial stem engine.
//
// Sample: digit 8, MNIST test index 61 (sim/vectors/golden_trace).
// Golden vectors: input_q.mem (784), conv1_acc.mem (12544), stem_q.mem (12544).
//
// Flow:
//   1. reset
//   2. load the 784 input bytes through the input write port
//   3. assert start for one cycle
//   4. on every acc_valid / q_valid pulse, compare acc_value/conv1_acc and
//      q_value/stem_q bit-exactly, and check acc_addr/q_addr == 0..12543
//   5. wait for the single done pulse
//   6. read all 12544 output bytes back through output_raddr/output_rdata and
//      compare again with stem_q
//   7. report cycle counts
//
// Any mismatch prints the first 20 failures (index / oc,y,x / expected vs
// actual / FSM state / ky,kx) and terminates with $fatal (non-zero vsim exit).
// A watchdog aborts with $fatal if done never comes (stuck FSM).
`timescale 1ns/1ps

module tb_stem_conv_serial;
    // ================= clock =================
    reg clk = 1'b0;
    always #5 clk = ~clk;             // 10 ns period

    // ================= DUT =================
    reg              rst_n = 1'b0;
    reg              input_we = 1'b0;
    reg [9:0]        input_waddr = 10'd0;
    reg signed [7:0] input_wdata = 8'sd0;
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

    // ================= golden memories =================
    reg [7:0]  input_q   [0:783];
    reg [31:0] conv1_acc [0:12543];
    reg [7:0]  stem_q    [0:12543];

    // ================= statistics / monitors =================
    integer cmp_idx;        // index of the next expected output element
    integer mismatches;     // stream compare mismatches
    integer addr_bad;       // acc_addr / q_addr ordering mismatches
    integer qcnt;           // q_valid pulse count (expect 12544)
    integer done_count;     // done pulse count (expect 1)
    integer busy_cycles;    // cycles busy == 1
    integer cyc;            // global cycle counter
    integer start_cyc, done_cyc;
    integer readback_bad;   // output RAM readback mismatches

    integer i;

    // watchdog: fires $fatal if the FSM hangs (done never asserted).  The
    // normal flow calls $finish, which terminates this thread first.
    initial begin : watchdog
        repeat (500000) @(posedge clk);
        $fatal(1, "STEM: TIMEOUT - state machine stuck, done never asserted");
    end

    // sample the debug streams every clock edge (after the state settles).
    always @(posedge clk) begin
        #1;
        cyc = cyc + 1;
        if (busy) busy_cycles = busy_cycles + 1;
        if (done) begin
            done_count = done_count + 1;
            done_cyc = cyc;
        end
        if (q_valid) begin
            qcnt = qcnt + 1;
            if (cmp_idx < 12544) begin
                if (acc_value !== $signed(conv1_acc[cmp_idx]) ||
                    q_value    !== stem_q[cmp_idx]) begin
                    if (mismatches < 20)
                        $display("STEM MISMATCH idx=%0d oc=%0d y=%0d x=%0d  acc_exp=%0d acc_act=%0d  q_exp=%0d q_act=%0d  state=%0d tap=%0d ky=%0d kx=%0d",
                                 cmp_idx, cmp_idx/784, (cmp_idx%784)/28, cmp_idx%28,
                                 $signed(conv1_acc[cmp_idx]), $signed(acc_value),
                                 stem_q[cmp_idx], q_value, dut.state, dut.tap, dut.tap/3, dut.tap%3);
                    mismatches = mismatches + 1;
                end
                if (acc_addr !== cmp_idx[13:0] || q_addr !== cmp_idx[13:0]) begin
                    if (addr_bad < 20)
                        $display("STEM ADDR  idx=%0d acc_addr=%0d q_addr=%0d (expect %0d)", cmp_idx, acc_addr, q_addr, cmp_idx);
                    addr_bad = addr_bad + 1;
                end
                cmp_idx = cmp_idx + 1;
            end
        end
    end

    initial begin
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/input_q.mem",   input_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv1_acc.mem", conv1_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/stem_q.mem",    stem_q);

        // $readmemh is a task (no return value); guard loads with sentinels so
        // a missing / wrong-path file aborts instead of vacuously passing.
        if (input_q[0]   !== 8'hED)       $fatal(1, "STEM: input_q.mem not loaded (input_q[0]=%h)", input_q[0]);
        if (conv1_acc[0] !== 32'h000012DC) $fatal(1, "STEM: conv1_acc.mem not loaded (conv1_acc[0]=%h)", conv1_acc[0]);
        if (stem_q[0]    !== 8'h24)       $fatal(1, "STEM: stem_q.mem not loaded (stem_q[0]=%h)", stem_q[0]);

        cmp_idx = 0; mismatches = 0; addr_bad = 0; qcnt = 0;
        done_count = 0; busy_cycles = 0; cyc = 0; readback_bad = 0;

        // ---- 1. reset ----
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // ---- 2. load input (784 bytes through the write port) ----
        for (i = 0; i < 784; i = i + 1) begin
            input_we   = 1'b1;
            input_waddr = i[9:0];
            input_wdata = $signed(input_q[i]);
            @(posedge clk);
        end
        input_we = 1'b0;
        @(posedge clk);

        // ---- 3. start (single-cycle pulse) ----
        start = 1'b1;
        @(posedge clk);
        start_cyc = cyc;
        start = 1'b0;

        // ---- 4/5. wait for the done pulse ----
        while (!done) @(posedge clk);
        #1;

        // ---- 6. readback through the output RAM read port ----
        for (i = 0; i < 12544; i = i + 1) begin
            output_raddr = i[13:0];
            @(posedge clk);
            #1;
            if (output_rdata !== stem_q[i]) begin
                if (readback_bad < 20)
                    $display("STEM READBACK MISMATCH idx=%0d oc=%0d y=%0d x=%0d  got=%0d expect=%0d",
                             i, i/784, (i%784)/28, i%28, output_rdata, stem_q[i]);
                readback_bad = readback_bad + 1;
            end
        end

        // ---- 7. report ----
        $display("STEM: q_valid_pulses=%0d expected=12544", qcnt);
        $display("STEM: conv1_acc mismatches=%0d / 12544", mismatches);
        $display("STEM: stem_q stream mismatches=%0d / 12544", mismatches);
        $display("STEM: acc/q_addr ordering bad=%0d", addr_bad);
        $display("STEM: output RAM readback mismatches=%0d / 12544", readback_bad);
        $display("STEM: done_count=%0d (expect 1), busy_cycles=%0d", done_count, busy_cycles);
        $display("STEM: start_cyc=%0d done_cyc=%0d total_cycles=%0d", start_cyc, done_cyc, done_cyc - start_cyc);

        if (mismatches != 0 || addr_bad != 0 || qcnt != 12544 ||
            done_count != 1 || readback_bad != 0 || busy_cycles == 0) begin
            $fatal(1, "STEM: %0d mismatches, %0d addr bad, qcnt=%0d, done=%0d, readback=%0d",
                   mismatches, addr_bad, qcnt, done_count, readback_bad);
        end

        $display("STEM_ALL_PASS");
        $finish;
    end
endmodule
