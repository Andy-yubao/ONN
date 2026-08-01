// tb_stem_pool1_pipeline.v - Golden-vector verification of the phase-2
// integration: input_q -> stem_conv_serial (STORE_OUTPUT_RAM=0) -> streaming
// MaxPool -> pool1 RAM.
//
// Sample: digit 8, MNIST test index 61 (sim/vectors/golden_trace).
// Golden vectors: input_q.mem (784), conv1_acc.mem (12544), stem_q.mem (12544),
// pool1_q.mem (3136).
//
// Flow (per pass):
//   1. reset
//   2. load the 784 input bytes through the input write port
//   3. assert ONE `start` for one cycle - it launches stem and maxpool together
//   4. compare the full stem debug stream (acc + q) against conv1_acc/stem_q,
//      and the pool1 stream against pool1_q, bit-exactly
//   5. wait for the single done pulse (= maxpool_done)
//   6. read all 3136 pool1 bytes back through pool_raddr/pool_rdata
//   7. report cycle counts and run the frozen integration assertions
//
// Frozen integration assertions (docs/rtl_microarchitecture.md §10):
//   * stem_q_valid => maxpool_busy == 1 (every cycle)
//   * on pool_valid, pool_addr is strictly continuous 0..3135
//   * exactly 12544 stem_q received and exactly 3136 pool1_q written before done
//   * no input lost (all golden values bit-identical), done_count == 1, no X/Z
//   * start and the first in_valid are never on the same sampling edge
//   * stem_done and maxpool_done fire on the same cycle; completion = maxpool_done
//
// Boundary tests:
//   * pass 1: an extra `start` pulse asserted mid-run (both FSMs busy) is ignored
//   * pass 2: full reset + second inference run; every pool output must be
//     bit-identical to pass 1 (determinism)
//
// Any mismatch prints the first 20 failures and terminates with $fatal (non-zero
// vsim exit).  A watchdog aborts with $fatal if a run never finishes.
`timescale 1ns/1ps

module tb_stem_pool1_pipeline;
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
    wire             pool_valid;
    wire [11:0]      pool_addr;
    wire [7:0]       pool_value;
    reg [11:0]       pool_raddr = 12'd0;
    wire [7:0]       pool_rdata;

    wire             stem_busy;
    wire             maxpool_busy;
    wire             stem_done;
    wire             maxpool_done;
    wire             stem_q_valid;
    wire [7:0]       stem_q_value;

    stem_pool1_pipeline dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .input_we      (input_we),
        .input_waddr   (input_waddr),
        .input_wdata   (input_wdata),
        .start         (start),
        .busy          (busy),
        .done          (done),
        .pool_valid    (pool_valid),
        .pool_addr     (pool_addr),
        .pool_value    (pool_value),
        .pool_raddr    (pool_raddr),
        .pool_rdata    (pool_rdata),
        .stem_busy     (stem_busy),
        .maxpool_busy  (maxpool_busy),
        .stem_done     (stem_done),
        .maxpool_done  (maxpool_done),
        .stem_q_valid  (stem_q_valid),
        .stem_q_value  (stem_q_value)
    );

    // ================= golden memories =================
    reg [7:0]  input_q   [0:783];
    reg [31:0] conv1_acc [0:12543];
    reg [7:0]  stem_q    [0:12543];
    reg [7:0]  pool1_q   [0:3135];

    // ================= statistics =================
    integer passid;          // 0 = pass 1, 1 = pass 2
    integer stem_cmp;        // next expected stem element index (0..12543)
    integer pool_cmp;        // next expected pool element index (0..3135)
    integer qcnt;            // stem q_valid pulse count (expect 12544)
    integer pcnt;            // pool_valid pulse count (expect 3136)
    integer mismatches;      // stem acc/q stream mismatches vs golden
    integer addr_bad;        // stem acc_addr/q_addr ordering mismatches
    integer pool_bad;        // pool_value mismatches vs golden
    integer pool_addr_bad;   // pool_addr continuity mismatches
    integer xz_bad;          // X/Z on pool/busy/done
    integer cross_bad;       // pass2 pool value != pass1 pool value
    integer busy_assn_bad;   // stem_q_valid while maxpool_busy == 0
    integer done_count;      // maxpool_done pulse count (expect 1)
    integer stem_done_count; // stem_done pulse count (expect 1)
    integer busy_cycles;     // cycles busy == 1
    integer cyc;             // global cycle counter (monitor timebase)
    integer start_cyc;       // cycle index of the start sampling edge
    integer first_stem_q_cyc;
    integer first_pool_cyc;
    integer stem_done_cyc;
    integer maxpool_done_cyc;
    integer done_cyc;
    integer readback_bad;    // pool1 RAM readback mismatches
    integer p;
    reg [7:0] poolA [0:3135];    // pass-1 recorded pool outputs (cross-check)

    // watchdog: fires $fatal if a run hangs (done never asserted).  The normal
    // flow calls $finish, which terminates this thread first.
    initial begin : watchdog
        repeat (1000000) @(posedge clk);
        $fatal(1, "PIPE: TIMEOUT - pipeline stuck, done never asserted");
    end

    // monitor: sample every clock edge (after the state settles).
    always @(posedge clk) begin
        #1;
        cyc = cyc + 1;
        if (busy) busy_cycles = busy_cycles + 1;

        // done / stem_done relative timing
        if (stem_done) begin
            stem_done_count = stem_done_count + 1;
            stem_done_cyc = cyc;
        end
        if (maxpool_done) begin
            done_count = done_count + 1;
            maxpool_done_cyc = cyc;
        end
        if (done) done_cyc = cyc;

        // X/Z on the pool stream and the status signals
        if (busy === 1'bx || done === 1'bx || pool_valid === 1'bx ||
            pool_valid === 1'bz || ^pool_addr === 1'bx || ^pool_value === 1'bx) begin
            if (xz_bad < 20)
                $display("PIPE XZ cyc=%0d busy=%b done=%b pool_valid=%b pool_addr=%0d pool_value=%h",
                         cyc, busy, done, pool_valid, pool_addr, pool_value);
            xz_bad = xz_bad + 1;
        end

        // ---- stem debug stream (hierarchical ref to the stem instance) ----
        if (dut.u_stem.q_valid) begin
            qcnt = qcnt + 1;
            if (stem_cmp < 12544) begin
                if (dut.u_stem.acc_value !== $signed(conv1_acc[stem_cmp]) ||
                    dut.u_stem.q_value    !== stem_q[stem_cmp]) begin
                    if (mismatches < 20)
                        $display("PIPE STEM MISMATCH idx=%0d oc=%0d y=%0d x=%0d acc_exp=%0d acc_act=%0d q_exp=%0d q_act=%0d",
                                 stem_cmp, stem_cmp/784, (stem_cmp%784)/28, stem_cmp%28,
                                 $signed(conv1_acc[stem_cmp]), $signed(dut.u_stem.acc_value),
                                 stem_q[stem_cmp], dut.u_stem.q_value);
                    mismatches = mismatches + 1;
                end
                if (dut.u_stem.acc_addr !== stem_cmp[13:0] ||
                    dut.u_stem.q_addr    !== stem_cmp[13:0]) begin
                    if (addr_bad < 20)
                        $display("PIPE STEM ADDR idx=%0d acc_addr=%0d q_addr=%0d (expect %0d)",
                                 stem_cmp, dut.u_stem.acc_addr, dut.u_stem.q_addr, stem_cmp);
                    addr_bad = addr_bad + 1;
                end
                if (stem_cmp == 0) first_stem_q_cyc = cyc;
                // frozen: stem_q_valid must always imply maxpool_busy
                if (!maxpool_busy) begin
                    if (busy_assn_bad < 20)
                        $display("PIPE BUSY-ASSN cyc=%0d stem_q_valid=1 but maxpool_busy=0", cyc);
                    busy_assn_bad = busy_assn_bad + 1;
                end
                stem_cmp = stem_cmp + 1;
            end
        end

        // ---- pool1 stream ----
        if (pool_valid) begin
            pcnt = pcnt + 1;
            if (pool_cmp < 3136) begin
                if (pool_value !== pool1_q[pool_cmp]) begin
                    if (pool_bad < 20)
                        $display("PIPE POOL MISMATCH idx=%0d oc=%0d py=%0d px=%0d exp=%0d act=%0d",
                                 pool_cmp, pool_cmp/196, (pool_cmp%196)/14, pool_cmp%14,
                                 pool1_q[pool_cmp], pool_value);
                    pool_bad = pool_bad + 1;
                end
                if (pool_addr !== pool_cmp[11:0]) begin
                    if (pool_addr_bad < 20)
                        $display("PIPE POOL ADDR idx=%0d oc=%0d py=%0d px=%0d pool_addr=%0d (expect %0d)",
                                 pool_cmp, pool_cmp/196, (pool_cmp%196)/14, pool_cmp%14,
                                 pool_addr, pool_cmp);
                    pool_addr_bad = pool_addr_bad + 1;
                end
                if (pool_cmp == 0) first_pool_cyc = cyc;
                if (passid == 0) begin
                    poolA[pool_cmp] = pool_value;
                end else if (pool_value !== poolA[pool_cmp]) begin
                    if (cross_bad < 20)
                        $display("PIPE CROSS pass2 idx=%0d expA=%0d act=%0d", pool_cmp, poolA[pool_cmp], pool_value);
                    cross_bad = cross_bad + 1;
                end
                pool_cmp = pool_cmp + 1;
            end
        end
    end

    // ================= one pass =================
    task run_pass;
        input integer passno;       // 1 or 2
        integer ok;
        begin
            ok = 0;

            // ---- reset the DUT ----
            rst_n = 1'b0;
            start = 1'b0;
            input_we = 1'b0;
            input_waddr = 10'd0;
            input_wdata = 8'sd0;
            pool_raddr = 12'd0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);

            // ---- reset statistics ----
            passid = passno - 1;
            stem_cmp = 0; pool_cmp = 0;
            qcnt = 0; pcnt = 0;
            mismatches = 0; addr_bad = 0; pool_bad = 0; pool_addr_bad = 0;
            xz_bad = 0; cross_bad = 0; busy_assn_bad = 0;
            done_count = 0; stem_done_count = 0;
            busy_cycles = 0; cyc = 0;
            start_cyc = 0; first_stem_q_cyc = 0; first_pool_cyc = 0;
            stem_done_cyc = 0; maxpool_done_cyc = 0; done_cyc = 0;
            readback_bad = 0;

            // ---- load input (784 bytes through the write port) ----
            for (p = 0; p < 784; p = p + 1) begin
                input_we = 1'b1;
                input_waddr = p[9:0];
                input_wdata = $signed(input_q[p]);
                @(posedge clk);
            end
            input_we = 1'b0;
            @(posedge clk);

            // ---- start (single-cycle pulse); launches stem + maxpool together ----
            start = 1'b1;
            @(posedge clk);
            start_cyc = cyc + 1;      // actual cycle index of the start edge
            start = 1'b0;

            // ---- boundary: extra start mid-run must be ignored (pass 1 only) ----
            // Both FSMs are busy, so this pulse must be dropped without corrupting
            // the run; every golden compare below still has to pass.
            if (passno == 1) begin
                while (qcnt < 200) @(posedge clk);
                start = 1'b1;
                @(posedge clk);
                start = 1'b0;
            end

            // ---- wait for the single done pulse ----
            // Use the monitor's done_count (not the live `done`): the done pulse
            // fires and is consumed asynchronously; done_count latches it.
            wait (done_count == 1);
            #1;

            // ---- readback pool1 RAM (3136 entries) ----
            for (p = 0; p < 3136; p = p + 1) begin
                pool_raddr = p[11:0];
                @(posedge clk);
                #1;
                if (pool_rdata !== pool1_q[p]) begin
                    if (readback_bad < 20)
                        $display("PIPE READBACK MISMATCH idx=%0d oc=%0d py=%0d px=%0d got=%0d expect=%0d",
                                 p, p/196, (p%196)/14, p%14, pool_rdata, pool1_q[p]);
                    readback_bad = readback_bad + 1;
                end
            end

            // ---- report ----
            $display("PIPE PASS[%0d] qcnt=%0d/12544 pcnt=%0d/3136 stem_mismatches=%0d stem_addr_bad=%0d pool_mismatches=%0d pool_addr_bad=%0d xz_bad=%0d cross_bad=%0d busy_assn_bad=%0d done=%0d stem_done=%0d readback_bad=%0d",
                     passno, qcnt, pcnt, mismatches, addr_bad, pool_bad, pool_addr_bad,
                     xz_bad, cross_bad, busy_assn_bad, done_count, stem_done_count, readback_bad);
            $display("PIPE PASS[%0d] start_cyc=%0d first_stem_q=%0d first_pool=%0d stem_done=%0d maxpool_done=%0d done=%0d",
                     passno, start_cyc, first_stem_q_cyc, first_pool_cyc, stem_done_cyc, maxpool_done_cyc, done_cyc);
            $display("PIPE PASS[%0d] cyc start->first_stem_q=%0d start->first_pool1_q=%0d start->done=%0d busy_cycles=%0d",
                     passno, first_stem_q_cyc - start_cyc, first_pool_cyc - start_cyc,
                     maxpool_done_cyc - start_cyc, busy_cycles);

            // ---- verdict ----
            ok = (qcnt == 12544 && pcnt == 3136 && mismatches == 0 && addr_bad == 0 &&
                  pool_bad == 0 && pool_addr_bad == 0 && xz_bad == 0 && cross_bad == 0 &&
                  busy_assn_bad == 0 && done_count == 1 && stem_done_count == 1 &&
                  readback_bad == 0 && busy_cycles > 0 &&
                  first_stem_q_cyc >= start_cyc + 2 &&      // start & first in_valid never same edge
                  stem_done_cyc == maxpool_done_cyc &&      // stem_done == maxpool_done cycle
                  maxpool_done_cyc == done_cyc);            // top-level done == maxpool_done
            if (!ok)
                $fatal(1, "PIPE PASS[%0d] FAILED (qcnt=%0d pcnt=%0d stem=%0d addr=%0d pool=%0d paddr=%0d xz=%0d cross=%0d busy=%0d done=%0d sdone=%0d readback=%0d)",
                       passno, qcnt, pcnt, mismatches, addr_bad, pool_bad, pool_addr_bad,
                       xz_bad, cross_bad, busy_assn_bad, done_count, stem_done_count, readback_bad);
        end
    endtask

    // ================= main =================
    initial begin
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/input_q.mem",   input_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv1_acc.mem", conv1_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/stem_q.mem",    stem_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/pool1_q.mem",   pool1_q);

        // $readmemh is a task (no return value); guard loads with sentinels so
        // a missing / wrong-path file aborts instead of vacuously passing.
        if (input_q[0]   !== 8'hED)       $fatal(1, "PIPE: input_q.mem not loaded (input_q[0]=%h)", input_q[0]);
        if (conv1_acc[0] !== 32'h000012DC) $fatal(1, "PIPE: conv1_acc.mem not loaded (conv1_acc[0]=%h)", conv1_acc[0]);
        if (stem_q[0]    !== 8'h24)       $fatal(1, "PIPE: stem_q.mem not loaded (stem_q[0]=%h)", stem_q[0]);
        if (pool1_q[0]   !== 8'h24)       $fatal(1, "PIPE: pool1_q.mem not loaded (pool1_q[0]=%h)", pool1_q[0]);
        if (pool1_q[3135] !== 8'h05)      $fatal(1, "PIPE: pool1_q.mem truncated (pool1_q[3135]=%h)", pool1_q[3135]);

        run_pass(1);            // pass 1: golden + extra-start-ignored test
        run_pass(2);            // pass 2: reset, rerun, must be identical to pass 1

        $display("PIPE_ALL_PASS");
        $finish;
    end
endmodule
