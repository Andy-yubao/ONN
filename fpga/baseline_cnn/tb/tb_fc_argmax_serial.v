// tb_fc_argmax_serial.v - Golden-vector verification of FC + argmax.
//
// Sample: digit 8, MNIST test index 61 (sim/vectors/golden_trace).
// Golden vectors: gap_q.mem (32, FC INPUT), fc_acc.mem (10), prediction.txt (=8).
//
// Topology under test:
//     gap_we/gap_waddr/gap_wdata -> fc_argmax_serial (gap_mem 32xUINT8)
//     -> fc_valid/fc_class/fc_acc (10 items) -> prediction (signed argmax)
//
// Flow:
//   1. reset
//   2. write all 32 gap values (verify 32/32 writes via hierarchical readback)
//   3. assert ONE `start`; compare the 10 fc_valid results against fc_acc
//      bit-exactly; fc_class strictly 0..9; prediction == 8; no X/Z; done single
//   4. recompute the signed argmax from the observed fc_acc values and confirm
//      it matches prediction (argmax uses SIGNED comparison)
//   Pass 1 (golden) records every output; pass 2 (golden, NO reset) must be
//   bit-identical; pass 3 uses a hand-built input (gap[0]=82, rest 0) that makes
//   classes 5 and 7 tie at 6891 (strictly above all others) - the argmax must
//   select the SMALLER class index (5), proving the "only on >" tie rule.
//
// Any mismatch prints the first 20 failures and terminates with $fatal (non-zero
// vsim exit).  A watchdog aborts with $fatal if a pass never finishes.
`timescale 1ns/1ps

module tb_fc_argmax_serial;
    // ================= clock =================
    reg clk = 1'b0;
    always #5 clk = ~clk;             // 10 ns period

    // ================= DUT =================
    reg rst_n = 1'b0;
    reg gap_we = 1'b0;
    reg [4:0] gap_waddr = 5'd0;
    reg [7:0] gap_wdata = 8'd0;
    reg start = 1'b0;
    wire busy;
    wire done;
    wire fc_valid;
    wire [3:0] fc_class;
    wire signed [31:0] fc_acc;
    wire [3:0] prediction;

    fc_argmax_serial u_fc (
        .clk        (clk),
        .rst_n      (rst_n),
        .gap_we     (gap_we),
        .gap_waddr  (gap_waddr),
        .gap_wdata  (gap_wdata),
        .start      (start),
        .busy       (busy),
        .done       (done),
        .fc_valid   (fc_valid),
        .fc_class   (fc_class),
        .fc_acc     (fc_acc),
        .prediction (prediction)
    );

    // ================= golden memories =================
    reg [7:0]  gap_q   [0:31];
    reg signed [31:0] fc_acc_gold [0:9];

    // hand-built tie input: gap[0]=82, rest 0 -> classes 5 and 7 tie at 6891
    // (all other classes strictly lower); argmax must pick class 5.
    reg [7:0]  tie_gap [0:31];
    reg signed [31:0] tie_acc [0:9];

    // ================= statistics =================
    integer passid;          // 1..3
    integer fc_cnt;          // fc_valid pulse count (expect 10)
    integer cmp_oc;          // expected fc_class (0..9)
    integer mismatches;      // fc_acc mismatches vs expected
    integer class_bad;       // fc_class ordering mismatches
    integer gap_wr_bad;      // gap_mem write verification mismatches
    integer argmax_bad;      // recomputed signed argmax != prediction
    integer pred_bad;        // prediction != golden / tie expectation
    integer xz_bad;          // X/Z on the outputs / status
    integer done_count;      // done pulse count (expect 1)
    integer busy_cycles;     // cycles busy == 1
    integer cyc;             // global cycle counter
    integer start_cyc;
    integer first_fc_cyc;
    integer done_cyc;
    integer cross_bad;       // pass 2 output != pass 1 output
    integer gap_bad;         // busy dropped before done (busy=0 && done=0)
    reg started;             // a start has been asserted (per pass)
    reg done_seen;           // the done pulse has been observed (per pass)
    reg signed [31:0] obs_acc [0:9];  // observed fc_acc (argmax recompute)
    reg signed [31:0] accA   [0:9];   // pass-1 recorded outputs (cross-check)
    integer i;

    // watchdog: fires $fatal if a pass hangs (done never asserted).
    initial begin : watchdog
        repeat (200000) @(posedge clk);
        $fatal(1, "FC: TIMEOUT - fc run stuck, done never asserted");
    end

    // monitor: sample every clock edge (after the state settles).
    always @(posedge clk) begin
        #1;
        cyc = cyc + 1;
        if (busy) busy_cycles = busy_cycles + 1;
        if (done) begin
            done_count = done_count + 1;
            done_cyc = cyc;
            done_seen = 1;
        end

        // frozen: between start and the done pulse there must be NO cycle with
        // busy=0 && done=0 (busy holds 1 from start through S_DONE).
        if (started && !done_seen && !busy && !done) begin
            if (gap_bad < 20)
                $display("FC BUSY-GAP cyc=%0d busy=0 done=0 before done", cyc);
            gap_bad = gap_bad + 1;
        end

        // X/Z on the observable bus
        if (busy === 1'bx || done === 1'bx || fc_valid === 1'bx || fc_valid === 1'bz ||
            ^fc_class === 1'bx || ^fc_acc === 1'bx || ^prediction === 1'bx) begin
            if (xz_bad < 20)
                $display("FC XZ cyc=%0d busy=%b done=%b fc_valid=%b", cyc, busy, done, fc_valid);
            xz_bad = xz_bad + 1;
        end

        // fc acc/class stream (10 items, class order)
        if (fc_valid) begin
            fc_cnt = fc_cnt + 1;
            if (cmp_oc < 10) begin
                if (fc_class !== cmp_oc[3:0]) begin
                    if (class_bad < 20)
                        $display("FC CLASS fc_cnt=%0d class=%0d (expect %0d)", fc_cnt, fc_class, cmp_oc);
                    class_bad = class_bad + 1;
                end
                if (fc_acc !== ((passid <= 2) ? $signed(fc_acc_gold[cmp_oc]) : $signed(tie_acc[cmp_oc]))) begin
                    if (mismatches < 20)
                        $display("FC ACC class=%0d exp=%0d act=%0d", cmp_oc,
                                 (passid <= 2) ? $signed(fc_acc_gold[cmp_oc]) : $signed(tie_acc[cmp_oc]),
                                 $signed(fc_acc));
                    mismatches = mismatches + 1;
                end
                obs_acc[cmp_oc] = fc_acc;
                if (passid == 1) accA[cmp_oc] = fc_acc;
                if (passid == 2 && fc_acc !== $signed(accA[cmp_oc])) begin
                    if (cross_bad < 20)
                        $display("FC CROSS pass2 class=%0d expA=%0d act=%0d", cmp_oc, $signed(accA[cmp_oc]), $signed(fc_acc));
                    cross_bad = cross_bad + 1;
                end
                if (cmp_oc == 0) first_fc_cyc = cyc;
                cmp_oc = cmp_oc + 1;
            end
        end
    end

    // ================= one pass =================
    task run_pass;
        input integer passno;       // 1 = golden, 2 = golden (no reset), 3 = tie
        input integer do_reset;     // 1 = reset the DUT first
        integer ok;
        reg signed [31:0] best2;
        integer pred2;
        begin
            // ---- reset the DUT (optional) ----
            if (do_reset) begin
                rst_n = 1'b0;
                gap_we = 1'b0;
                start = 1'b0;
                repeat (4) @(posedge clk);
                rst_n = 1'b1;
                @(posedge clk);
            end

            // ---- reset statistics ----
            passid = passno;
            fc_cnt = 0; cmp_oc = 0; mismatches = 0; class_bad = 0;
            gap_wr_bad = 0; argmax_bad = 0; pred_bad = 0; xz_bad = 0;
            done_count = 0; busy_cycles = 0; cyc = 0;
            start_cyc = 0; first_fc_cyc = 0; done_cyc = 0;
            cross_bad = 0; gap_bad = 0; started = 0; done_seen = 0;

            // ---- write the 32 gap values (verify 32/32 writes) ----
            gap_we = 1'b1;
            for (i = 0; i < 32; i = i + 1) begin
                gap_waddr = i[4:0];
                gap_wdata = (passno <= 2) ? gap_q[i] : tie_gap[i];
                @(posedge clk);
            end
            gap_we = 1'b0;
            @(posedge clk);
            for (i = 0; i < 32; i = i + 1) begin
                if (u_fc.gap_mem[i] !== ((passno <= 2) ? gap_q[i] : tie_gap[i])) begin
                    if (gap_wr_bad < 20)
                        $display("FC GAP-WR ch=%0d exp=%0d act=%0d", i,
                                 (passno <= 2) ? gap_q[i] : tie_gap[i], u_fc.gap_mem[i]);
                    gap_wr_bad = gap_wr_bad + 1;
                end
            end

            // ---- start ----
            start = 1'b1;
            @(posedge clk);
            start_cyc = cyc + 1;      // actual cycle index of the DUT start edge
            started = 1;              // busy must hold 1 until done from now on
            start = 1'b0;
            @(posedge clk);

            // ---- wait for the single done pulse ----
            while (!done) @(posedge clk);
            #1;

            // ---- recompute the signed argmax from the observed fc_acc values
            //      and confirm it matches prediction ----
            best2 = obs_acc[0]; pred2 = 0;
            for (i = 1; i < 10; i = i + 1) begin
                if ($signed(obs_acc[i]) > $signed(best2)) begin
                    best2 = obs_acc[i];
                    pred2 = i;
                end
            end
            if (pred2 !== prediction) argmax_bad = argmax_bad + 1;

            // ---- expected prediction: 8 (golden), 5 (tie: class 5 < 7) ----
            if (passno <= 2) begin
                if (prediction !== 4'd8) pred_bad = pred_bad + 1;
            end else begin
                if (prediction !== 4'd5) pred_bad = pred_bad + 1;
            end

            // ---- report ----
            $display("FC PASS[%0d] fc_cnt=%0d/10 mismatches=%0d class_bad=%0d gap_wr_bad=%0d xz_bad=%0d done=%0d",
                     passno, fc_cnt, mismatches, class_bad, gap_wr_bad, xz_bad, done_count);
            $display("FC PASS[%0d] start=%0d first_fc=%0d done=%0d start->done=%0d busy_cycles=%0d cross=%0d gap=%0d argmax=%0d pred_bad=%0d prediction=%0d",
                     passno, start_cyc, first_fc_cyc, done_cyc,
                     done_cyc - start_cyc, busy_cycles, cross_bad, gap_bad, argmax_bad, pred_bad, prediction);

            if (fc_cnt != 10 || mismatches != 0 || class_bad != 0 || gap_wr_bad != 0 ||
                xz_bad != 0 || done_count != 1 || busy_cycles == 0 ||
                busy_cycles != done_cyc - start_cyc || cross_bad != 0 || gap_bad != 0 ||
                argmax_bad != 0 || pred_bad != 0)
                $fatal(1, "FC PASS[%0d] FAILED (fc=%0d mism=%0d cls=%0d wr=%0d xz=%0d done=%0d cross=%0d gap=%0d argmax=%0d pred=%0d)",
                       passno, fc_cnt, mismatches, class_bad, gap_wr_bad, xz_bad,
                       done_count, cross_bad, gap_bad, argmax_bad, pred_bad);
        end
    endtask

    // ================= main =================
    initial begin
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/gap_q.mem",   gap_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/fc_acc.mem",  fc_acc_gold);

        // sentinel guards (missing / truncated file aborts instead of passing)
        if (gap_q[0]      !== 8'h04)         $fatal(1, "FC: gap_q.mem not loaded (gap_q[0]=%h)", gap_q[0]);
        if (gap_q[31]     !== 8'h07)         $fatal(1, "FC: gap_q.mem truncated (gap_q[31]=%h)", gap_q[31]);
        if (fc_acc_gold[0] !== 32'hFFFFF168) $fatal(1, "FC: fc_acc.mem not loaded (fc_acc[0]=%h)", fc_acc_gold[0]);
        if (fc_acc_gold[8] !== 32'h00003D8A) $fatal(1, "FC: fc_acc.mem wrong (fc_acc[8]=%h)", fc_acc_gold[8]);
        if (fc_acc_gold[9] !== 32'hFFFFF193) $fatal(1, "FC: fc_acc.mem truncated (fc_acc[9]=%h)", fc_acc_gold[9]);

        // hand-built tie vector: gap[0]=82 (rest 0) -> fc_acc[5] == fc_acc[7] == 6891
        // (computed offline from the frozen fc_weight/fc_bias; strictly above all
        //  other classes).  The argmax must select the smaller index, class 5.
        for (i = 0; i < 32; i = i + 1) tie_gap[i] = 8'd0;
        tie_gap[0] = 8'd82;
        tie_acc[0] = -32'sd4444;  tie_acc[1] = -32'sd3640;  tie_acc[2] = 32'sd312;
        tie_acc[3] = 32'sd4544;   tie_acc[4] = -32'sd5480;  tie_acc[5] = 32'sd6891;
        tie_acc[6] = -32'sd3924;  tie_acc[7] = 32'sd6891;   tie_acc[8] = -32'sd8306;
        tie_acc[9] = -32'sd4717;
        // sanity: the tie really is a tie and dominates (guard against a broken
        // hard-coded vector silently passing because both expectations are wrong)
        if (tie_acc[5] !== tie_acc[7] || tie_acc[5] <= 32'sd4544)
            $fatal(1, "FC: tie vector sanity check failed (tie_acc[5]=%0d [7]=%0d)", tie_acc[5], tie_acc[7]);

        run_pass(1, 1);      // pass 1: golden, reset, record outputs
        run_pass(2, 0);      // pass 2: golden, NO reset, must be identical
        run_pass(3, 1);      // pass 3: artificial tie (classes 5 and 7)

        $display("FC_ALL_PASS");
        $finish;
    end
endmodule
