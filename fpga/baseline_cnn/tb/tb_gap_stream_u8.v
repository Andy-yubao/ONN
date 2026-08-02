// tb_gap_stream_u8.v - Golden-vector verification of the streaming GAP.
//
// Sample: digit 8, MNIST test index 61 (sim/vectors/golden_trace).
// Golden vectors: conv3_q.mem (1568, GAP INPUT, 32x7x7 CHW), gap_q.mem (32).
//
// Topology under test:
//     conv3_q stream -> gap_stream_u8 -> out_valid/out_channel/out_q (32 items)
//
// Flow:
//   1. reset
//   2. assert ONE `start`, then feed all 1568 conv3_q bytes with a gap pattern
//      (0 = continuous, one value per cycle; 1 = 5 values then 2 idle cycles)
//   3. compare the 32 out_valid results against gap_q bit-exactly; out_channel
//      strictly 0..31; done single pulse; busy holds until done (no premature
//      `busy=0 && done=0` window); no X/Z
//   Pass 1 (continuous) records every output; pass 2 (continuous, NO reset)
//   must be bit-identical; pass 3 (fixed gaps, reset first) re-checks golden.
//
// Any mismatch prints the first 20 failures and terminates with $fatal (non-zero
// vsim exit).  A watchdog aborts with $fatal if a pass never finishes.
`timescale 1ns/1ps

module tb_gap_stream_u8;
    // ================= clock =================
    reg clk = 1'b0;
    always #5 clk = ~clk;             // 10 ns period

    // ================= DUT =================
    reg rst_n = 1'b0;
    reg start = 1'b0;
    reg in_valid = 1'b0;
    reg [7:0] in_q = 8'd0;
    wire busy;
    wire out_valid;
    wire [4:0] out_channel;
    wire [7:0] out_q;
    wire done;

    gap_stream_u8 u_gap (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .in_valid    (in_valid),
        .in_q        (in_q),
        .busy        (busy),
        .out_valid   (out_valid),
        .out_channel (out_channel),
        .out_q       (out_q),
        .done        (done)
    );

    // ================= golden memories =================
    reg [7:0] conv3_q [0:1567];
    reg [7:0] gap_q   [0:31];

    // ================= statistics =================
    integer passid;          // 1..3
    integer ocnt;            // out_valid pulse count (expect 32)
    integer exp_chn;         // expected out_channel (0..31)
    integer val_bad;         // out_q mismatches vs golden
    integer ch_bad;          // out_channel ordering mismatches
    integer xz_bad;          // X/Z on the outputs / status
    integer done_count;      // done pulse count (expect 1)
    integer busy_cycles;     // cycles busy == 1
    integer cyc;             // global cycle counter
    integer start_cyc;
    integer first_out_cyc;
    integer done_cyc;
    integer cross_bad;       // pass 2 output != pass 1 output
    integer gap_bad;         // busy dropped before done (busy=0 && done=0)
    reg started;             // a start has been asserted (per pass)
    reg done_seen;           // the done pulse has been observed (per pass)
    reg [7:0] outA [0:31];   // pass-1 recorded outputs (cross-check)
    integer i;

    // watchdog: fires $fatal if a pass hangs (done never asserted).
    initial begin : watchdog
        repeat (200000) @(posedge clk);
        $fatal(1, "GAP: TIMEOUT - stream run stuck, done never asserted");
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
                $display("GAP BUSY-GAP cyc=%0d busy=0 done=0 before done", cyc);
            gap_bad = gap_bad + 1;
        end

        // X/Z on the observable bus
        if (busy === 1'bx || done === 1'bx || out_valid === 1'bx || out_valid === 1'bz ||
            ^out_channel === 1'bx || ^out_q === 1'bx) begin
            if (xz_bad < 20)
                $display("GAP XZ cyc=%0d busy=%b done=%b out_valid=%b", cyc, busy, done, out_valid);
            xz_bad = xz_bad + 1;
        end

        // GAP output stream (32 items, channel order)
        if (out_valid) begin
            ocnt = ocnt + 1;
            if (ocnt < 32) begin
                if (out_channel !== exp_chn[4:0]) begin
                    if (ch_bad < 20)
                        $display("GAP CHANNEL ocnt=%0d ch=%0d (expect %0d)", ocnt, out_channel, exp_chn);
                    ch_bad = ch_bad + 1;
                end
                if (out_q !== gap_q[exp_chn]) begin
                    if (val_bad < 20)
                        $display("GAP VALUE ch=%0d exp=%0d act=%0d", exp_chn, gap_q[exp_chn], out_q);
                    val_bad = val_bad + 1;
                end
                if (passid == 1) outA[exp_chn] = out_q;
                if (passid == 2 && out_q !== outA[exp_chn]) begin
                    if (cross_bad < 20)
                        $display("GAP CROSS pass2 ch=%0d expA=%0d act=%0d", exp_chn, outA[exp_chn], out_q);
                    cross_bad = cross_bad + 1;
                end
                if (ocnt == 1) first_out_cyc = cyc;
                exp_chn = exp_chn + 1;
            end
        end
    end

    // ================= one pass =================
    task run_pass;
        input integer passno;       // 1, 2 or 3
        input integer gap_mode;     // 0 = continuous, 1 = 5-on / 2-off
        input integer do_reset;     // 1 = reset the DUT first
        integer i;
        begin
            // ---- reset the DUT (optional) ----
            if (do_reset) begin
                rst_n = 1'b0;
                start = 1'b0;
                in_valid = 1'b0;
                repeat (4) @(posedge clk);
                rst_n = 1'b1;
                @(posedge clk);
            end

            // ---- reset statistics ----
            passid = passno;
            ocnt = 0; exp_chn = 0; val_bad = 0; ch_bad = 0;
            xz_bad = 0; done_count = 0; busy_cycles = 0; cyc = 0;
            start_cyc = 0; first_out_cyc = 0; done_cyc = 0;
            cross_bad = 0; gap_bad = 0; started = 0; done_seen = 0;

            // ---- start ----
            start = 1'b1;
            @(posedge clk);
            start_cyc = cyc + 1;      // actual cycle index of the DUT start edge
            started = 1;              // busy must hold 1 until done from now on
            start = 1'b0;
            @(posedge clk);

            // ---- feed the conv3_q stream with the requested gap pattern ----
            i = 0;
            while (i < 1568) begin
                in_valid = 1'b1;
                in_q = conv3_q[i];
                @(posedge clk);
                i = i + 1;
                if (gap_mode == 1) begin
                    // every 5 consecutive inputs, insert 2 idle cycles
                    if ((i % 5) == 0 && i < 1568) begin
                        in_valid = 1'b0;
                        in_q = 8'hXX;
                        @(posedge clk);
                        @(posedge clk);
                    end
                end
            end
            in_valid = 1'b0;

            // ---- wait for the single done pulse ----
            while (!done) @(posedge clk);
            #1;

            // ---- report ----
            $display("GAP PASS[%0d] gap_mode=%0d ocnt=%0d/32 val_bad=%0d ch_bad=%0d xz_bad=%0d done=%0d",
                     passno, gap_mode, ocnt, val_bad, ch_bad, xz_bad, done_count);
            $display("GAP PASS[%0d] start=%0d first_out=%0d done=%0d start->done=%0d busy_cycles=%0d cross_bad=%0d gap_bad=%0d",
                     passno, start_cyc, first_out_cyc, done_cyc,
                     done_cyc - start_cyc, busy_cycles, cross_bad, gap_bad);

            if (ocnt != 32 || val_bad != 0 || ch_bad != 0 || xz_bad != 0 ||
                done_count != 1 || busy_cycles == 0 || busy_cycles != done_cyc - start_cyc ||
                cross_bad != 0 || gap_bad != 0)
                $fatal(1, "GAP PASS[%0d] FAILED (ocnt=%0d val=%0d ch=%0d xz=%0d done=%0d cross=%0d gap=%0d)",
                       passno, ocnt, val_bad, ch_bad, xz_bad, done_count, cross_bad, gap_bad);
        end
    endtask

    // ================= main =================
    initial begin
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_q.mem", conv3_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/gap_q.mem",   gap_q);

        // sentinel guards (missing / truncated file aborts instead of passing)
        if (gap_q[0]  !== 8'h04)        $fatal(1, "GAP: gap_q.mem not loaded (gap_q[0]=%h)", gap_q[0]);
        if (gap_q[31] !== 8'h07)        $fatal(1, "GAP: gap_q.mem truncated (gap_q[31]=%h)", gap_q[31]);
        if (conv3_q[0] !== 8'h00 || conv3_q[1567] !== 8'h00)
            $fatal(1, "GAP: conv3_q.mem not loaded (conv3_q[0]=%h [1567]=%h)", conv3_q[0], conv3_q[1567]);

        run_pass(1, 0, 1);      // pass 1: continuous, record outputs
        run_pass(2, 0, 0);      // pass 2: continuous, NO reset, must be identical
        run_pass(3, 1, 1);      // pass 3: fixed gaps, reset first

        $display("GAP_ALL_PASS");
        $finish;
    end
endmodule
