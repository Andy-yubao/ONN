// tb_maxpool2x2_stream.v - Golden-vector verification of the streaming maxpool.
//
// Sample: digit 8, MNIST test index 61 (sim/vectors/golden_trace).
// Golden vectors: stem_q.mem (12544, the pool INPUT stream) and
// pool1_q.mem (3136, the pool OUTPUT).  Two passes over the same stream:
//
//   Test A (continuous): start, then in_valid=1 for 12544 consecutive cycles.
//   Test B (with gaps):  every 5 valid inputs followed by 2 gap cycles
//                        (in_valid=0); outputs and addresses must be identical
//                        to Test A, and internal oc/y/x/out_cnt must NOT move
//                        during any gap.
//
// On every out_valid pulse the output is compared bit-exactly against pool1_q,
// out_addr must equal the running index (0..3135), and the output bus must be
// free of X/Z.  Five windows are additionally recomputed by hand in the
// testbench (4 stem_q values -> max): oc0 pool(0,0) / pool(0,13) / pool(13,0)
// / pool(13,13) and oc15 pool(13,13).
//
// Any mismatch prints the first 20 failures (input count, output index,
// oc/pool_y/pool_x, the four input values, expected vs actual) and terminates
// with $fatal (non-zero vsim exit).  A watchdog aborts with $fatal if a run
// never finishes.
`timescale 1ns/1ps

module tb_maxpool2x2_stream;
    // ================= clock =================
    reg clk = 1'b0;
    always #5 clk = ~clk;             // 10 ns period

    // ================= DUT =================
    reg       rst_n = 1'b0;
    reg       start = 1'b0;
    reg       in_valid = 1'b0;
    reg [7:0] in_q = 8'd0;
    wire      busy;
    wire      out_valid;
    wire [11:0] out_addr;
    wire [7:0]  out_q;
    wire      done;

    maxpool2x2_stream dut (.*);

    // ================= golden memories =================
    reg [7:0] stem_q  [0:12543];   // pool INPUT stream (oc -> y -> x)
    reg [7:0] pool1_q [0:3135];    // pool OUTPUT golden (oc -> py -> px)

    // ================= statistics =================
    integer i;
    integer passid;         // 0 = Test A, 1 = Test B
    integer cmp_idx;        // next expected output index
    integer mismatches;
    integer addr_bad;
    integer outcnt;         // out_valid pulse count (expect 3136)
    integer incnt;          // valid inputs fed (expect 12544)
    integer done_count;     // done pulse count (expect 1)
    integer busy_cycles;    // cycles busy == 1
    integer cyc;
    integer start_cyc, done_cyc;
    integer xz_bad;         // X/Z observed on the output bus
    integer manual_bad;     // hand-recomputed window mismatches
    integer gap_bad;        // internal counters moved during a gap
    integer cross_bad;      // Test B output != Test A output
    integer oc_, py_, px_, w0, w1, w2, w3;
    integer m;
    reg [7:0] outA [0:3135];        // Test A recorded outputs
    reg       prev_was_gap;
    reg [3:0]  prev_gap_oc;
    reg [4:0]  prev_gap_y, prev_gap_x;
    reg [11:0] prev_gap_cnt;

    // watchdog: fires $fatal if a run hangs (done never asserted).  The normal
    // flow calls $finish, which terminates this thread first.
    initial begin : watchdog
        repeat (500000) @(posedge clk);
        $fatal(1, "MAXPOOL: TIMEOUT - run stuck, done never asserted");
    end

    // window input index helper: stem_q address of tap (dy,dx) in window
    // (oc, py, px).
    function [13:0] wi;
        input integer oc, py, px, dy, dx;
        begin
            wi = (oc * 28 + py * 2 + dy) * 28 + px * 2 + dx;
        end
    endfunction

    // monitor: sample the DUT outputs every edge (after the state settles).
    always @(posedge clk) begin
        #1;
        cyc = cyc + 1;
        if (busy) busy_cycles = busy_cycles + 1;
        if (done) begin
            done_count = done_count + 1;
            done_cyc = cyc;
        end

        // X/Z on any output port (busy/done/out_valid/out_addr/out_q)
        if (busy === 1'bx || done === 1'bx ||
            out_valid === 1'bx || out_valid === 1'bz ||
            ^out_addr === 1'bx || ^out_q === 1'bx) begin
            if (xz_bad < 20)
                $display("MAXPOOL XZ cyc=%0d busy=%b done=%b out_valid=%b out_addr=%0d out_q=%h",
                         cyc, busy, done, out_valid, out_addr, out_q);
            xz_bad = xz_bad + 1;
        end

        // compare each pooled result
        if (out_valid) begin
            outcnt = outcnt + 1;
            if (cmp_idx < 3136) begin
                oc_ = cmp_idx / 196;
                py_ = (cmp_idx % 196) / 14;
                px_ = cmp_idx % 14;
                w0 = wi(oc_, py_, px_, 0, 0);
                w1 = wi(oc_, py_, px_, 0, 1);
                w2 = wi(oc_, py_, px_, 1, 0);
                w3 = wi(oc_, py_, px_, 1, 1);
                if (out_q !== pool1_q[cmp_idx]) begin
                    if (mismatches < 20)
                        $display("MAXPOOL MISMATCH input_count=%0d out_idx=%0d oc=%0d py=%0d px=%0d  in4=%0d,%0d,%0d,%0d exp=%0d act=%0d state=%0d y=%0d x=%0d",
                                 incnt, cmp_idx, oc_, py_, px_,
                                 stem_q[w0], stem_q[w1], stem_q[w2], stem_q[w3],
                                 pool1_q[cmp_idx], out_q, dut.state, dut.y, dut.x);
                    mismatches = mismatches + 1;
                end
                if (out_addr !== cmp_idx[11:0]) begin
                    if (addr_bad < 20)
                        $display("MAXPOOL ADDR out_idx=%0d oc=%0d py=%0d px=%0d out_addr=%0d (expect %0d)",
                                 cmp_idx, oc_, py_, px_, out_addr, cmp_idx);
                    addr_bad = addr_bad + 1;
                end
                // record Test A outputs / cross-check Test B
                if (passid == 0) begin
                    outA[cmp_idx] = out_q;
                end else if (out_q !== outA[cmp_idx]) begin
                    if (cross_bad < 20)
                        $display("MAXPOOL CROSS passB out_idx=%0d expA=%0d actB=%0d", cmp_idx, outA[cmp_idx], out_q);
                    cross_bad = cross_bad + 1;
                end
                // hand-recompute the five special windows
                case (cmp_idx)
                    0, 13, 182, 195, 3135: begin
                        m = stem_q[w0];
                        if (stem_q[w1] > m) m = stem_q[w1];
                        if (stem_q[w2] > m) m = stem_q[w2];
                        if (stem_q[w3] > m) m = stem_q[w3];
                        if (out_q !== m[7:0]) begin
                            if (manual_bad < 20)
                                $display("MAXPOOL MANUAL out_idx=%0d oc=%0d py=%0d px=%0d exp=%0d act=%0d",
                                         cmp_idx, oc_, py_, px_, m, out_q);
                            manual_bad = manual_bad + 1;
                        end
                    end
                endcase
                cmp_idx = cmp_idx + 1;
            end
        end

        // gap check: while busy and the input is deasserted, internal counters
        // must not move from one gap cycle to the next.
        if (!in_valid && busy) begin
            if (prev_was_gap) begin
                if (dut.oc !== prev_gap_oc || dut.y !== prev_gap_y ||
                    dut.x !== prev_gap_x || dut.out_cnt !== prev_gap_cnt) begin
                    if (gap_bad < 20)
                        $display("MAXPOOL GAP-MOVE prev oc=%0d y=%0d x=%0d cnt=%0d now oc=%0d y=%0d x=%0d cnt=%0d",
                                 prev_gap_oc, prev_gap_y, prev_gap_x, prev_gap_cnt,
                                 dut.oc, dut.y, dut.x, dut.out_cnt);
                    gap_bad = gap_bad + 1;
                end
            end
            prev_gap_oc  = dut.oc;
            prev_gap_y   = dut.y;
            prev_gap_x   = dut.x;
            prev_gap_cnt = dut.out_cnt;
            prev_was_gap = 1'b1;
        end else begin
            prev_was_gap = 1'b0;
        end
    end

    // ================= one run =================
    task run_pass;
        input integer burst;    // 0 = continuous, else valid inputs per burst
        input integer gap;      // gap cycles after each burst
        integer pass_ok;
        begin
            pass_ok = 0;

            // ---- reset the module ----
            rst_n = 1'b0;
            in_valid = 1'b0;
            start = 1'b0;
            in_q = 8'd0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);

            // ---- reset statistics ----
            passid = (burst == 0) ? 0 : 1;
            cmp_idx = 0; mismatches = 0; addr_bad = 0; outcnt = 0;
            incnt = 0; done_count = 0; busy_cycles = 0; cyc = 0;
            xz_bad = 0; manual_bad = 0; gap_bad = 0; cross_bad = 0;
            start_cyc = 0; done_cyc = 0;
            prev_was_gap = 1'b0;

            // ---- start (single-cycle pulse) ----
            start = 1'b1;
            @(posedge clk);
            start_cyc = cyc;
            start = 1'b0;

            // ---- feed the stream ----
            i = 0;
            while (i < 12544) begin
                if (burst == 0) begin
                    // Test A: continuous
                    in_q = stem_q[i];
                    in_valid = 1'b1;
                    incnt = incnt + 1;
                    i = i + 1;
                    @(posedge clk);
                end else begin
                    // burst of `burst` valid inputs
                    repeat (burst) begin
                        if (i >= 12544) begin
                            in_valid = 1'b0;
                            @(posedge clk);
                        end else begin
                            in_q = stem_q[i];
                            in_valid = 1'b1;
                            incnt = incnt + 1;
                            i = i + 1;
                            @(posedge clk);
                        end
                    end
                    // gap of `gap` cycles
                    if (gap > 0) begin
                        in_valid = 1'b0;
                        repeat (gap) @(posedge clk);
                    end
                end
            end
            in_valid = 1'b0;
            @(posedge clk);

            // ---- wait for the single done pulse ----
            // Use the monitor's done_count, NOT the live `done` signal: in the
            // gap test the done pulse can fire during a trailing gap cycle that
            // the feed loop already consumed, so sampling `done` afterwards
            // would wait forever.  done_count latches the pulse regardless.
            wait (done_count == 1);
            #1;

            // ---- report ----
            $display("MAXPOOL PASS[%0d] burst=%0d gap=%0d  outcnt=%0d/3136 incnt=%0d/12544 mismatches=%0d addr_bad=%0d xz_bad=%0d manual_bad=%0d gap_bad=%0d cross_bad=%0d done=%0d busy_cycles=%0d",
                     passid, burst, gap, outcnt, incnt, mismatches, addr_bad,
                     xz_bad, manual_bad, gap_bad, cross_bad, done_count, busy_cycles);
            $display("MAXPOOL PASS[%0d] start_cyc=%0d done_cyc=%0d total_cycles=%0d",
                     passid, start_cyc, done_cyc, done_cyc - start_cyc);

            pass_ok = (outcnt == 3136 && incnt == 12544 && mismatches == 0 &&
                       addr_bad == 0 && xz_bad == 0 && manual_bad == 0 &&
                       gap_bad == 0 && cross_bad == 0 && done_count == 1 &&
                       busy_cycles > 0);
            if (!pass_ok)
                $fatal(1, "MAXPOOL PASS[%0d] FAILED (outcnt=%0d incnt=%0d mism=%0d addr=%0d xz=%0d manual=%0d gap=%0d cross=%0d done=%0d busy=%0d)",
                       passid, outcnt, incnt, mismatches, addr_bad, xz_bad,
                       manual_bad, gap_bad, cross_bad, done_count, busy_cycles);
        end
    endtask

    // ================= main =================
    initial begin
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/stem_q.mem",  stem_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/pool1_q.mem", pool1_q);

        // $readmemh is a task (no return value); guard loads with sentinels so
        // a missing / wrong-path file aborts instead of vacuously passing.
        if (stem_q[0]  !== 8'h24)  $fatal(1, "MAXPOOL: stem_q.mem not loaded (stem_q[0]=%h)", stem_q[0]);
        if (pool1_q[0] !== 8'h24)  $fatal(1, "MAXPOOL: pool1_q.mem not loaded (pool1_q[0]=%h)", pool1_q[0]);
        if (pool1_q[3135] !== 8'h05) $fatal(1, "MAXPOOL: pool1_q.mem truncated (pool1_q[3135]=%h)", pool1_q[3135]);

        run_pass(0, 0);            // Test A: continuous input
        run_pass(5, 2);            // Test B: 5 valid + 2 gap cycles

        $display("MAXPOOL_ALL_PASS");
        $finish;
    end
endmodule
