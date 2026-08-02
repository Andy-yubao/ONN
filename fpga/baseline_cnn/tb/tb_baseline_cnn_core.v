// tb_baseline_cnn_core.v - End-to-end golden verification of the complete core.
//
// Part A (golden trace, digit 8 / MNIST test index 61):
//   loads sim/vectors/golden_trace/*.mem, writes input_q into the core, runs ONE
//   inference, and compares EVERY internal node stream bit-exactly:
//       conv1_acc 12544/12544, stem_q 12544/12544, pool1_q 3136/3136,
//       conv2_acc 6272/6272,   conv2_q 6272/6272,   pool2_q 1568/1568,
//       conv3_acc 1568/1568,   conv3_q 1568/1568,   gap_q 32/32,
//       fc_acc 10/10, prediction 1/1 (== 8).
//   Also asserts: every stream address strictly 0..N-1; the controller stage
//   sequence IDLE..DONE..IDLE; conv2+pool2 started on the same cycle; conv3+GAP
//   started on the same cycle; conv2_done==pool2_done and conv3_done==gap_done
//   on the same cycle; GAP feeds FC exactly 32 values; FC emits exactly 10;
//   no X/Z; no illegal RAM/ROM address; done once; prediction == 8.
//   Records the stage cycle spans and the total start->done cycle count.
//
// Part B (10 smoke samples, digit0..digit9):
//   no global reset between runs: each run writes its input_q in IDLE, starts,
//   waits done, and compares its 10 fc_acc + prediction.  10/10 must pass; every
//   frame asserts done exactly once; later frames must not leak pool/gap/argmax
//   state from earlier ones (fc_acc matching is the leak check).
//
// Any mismatch prints the first 20 failures and terminates with $fatal (non-zero
// vsim exit).  A large watchdog aborts with $fatal if a run never finishes.
`timescale 1ns/1ps

module tb_baseline_cnn_core;
    // ================= clock =================
    reg clk = 1'b0;
    always #5 clk = ~clk;             // 10 ns period

    // ================= DUT =================
    reg rst_n = 1'b0;
    reg input_we = 1'b0;
    reg [9:0] input_waddr = 10'd0;
    reg signed [7:0] input_wdata = 8'sd0;
    reg start = 1'b0;
    wire busy, done;
    wire [3:0] prediction;
    wire [3:0] dbg_stage;
    wire stem_acc_valid;
    wire [13:0] stem_acc_addr;
    wire signed [31:0] stem_acc_value;
    wire stem_q_valid;
    wire [13:0] stem_q_addr;
    wire [7:0] stem_q_value;
    wire pool1_valid;
    wire [11:0] pool1_addr;
    wire [7:0] pool1_value;
    wire conv_layer;
    wire conv_acc_valid;
    wire [12:0] conv_acc_addr;
    wire signed [31:0] conv_acc_value;
    wire conv_q_valid;
    wire [12:0] conv_q_addr;
    wire [7:0] conv_q_value;
    wire pool2_valid;
    wire [10:0] pool2_addr;
    wire [7:0] pool2_value;
    wire gap_valid;
    wire [4:0] gap_channel;
    wire [7:0] gap_value;
    wire fc_valid;
    wire [3:0] fc_class;
    wire signed [31:0] fc_acc;

    baseline_cnn_core u_core (
        .clk            (clk),
        .rst_n          (rst_n),
        .input_we       (input_we),
        .input_waddr    (input_waddr),
        .input_wdata    (input_wdata),
        .start          (start),
        .busy           (busy),
        .done           (done),
        .prediction     (prediction),
        .dbg_stage      (dbg_stage),
        .stem_acc_valid (stem_acc_valid),
        .stem_acc_addr  (stem_acc_addr),
        .stem_acc_value (stem_acc_value),
        .stem_q_valid   (stem_q_valid),
        .stem_q_addr    (stem_q_addr),
        .stem_q_value   (stem_q_value),
        .pool1_valid    (pool1_valid),
        .pool1_addr     (pool1_addr),
        .pool1_value    (pool1_value),
        .conv_layer     (conv_layer),
        .conv_acc_valid (conv_acc_valid),
        .conv_acc_addr  (conv_acc_addr),
        .conv_acc_value (conv_acc_value),
        .conv_q_valid   (conv_q_valid),
        .conv_q_addr    (conv_q_addr),
        .conv_q_value   (conv_q_value),
        .pool2_valid    (pool2_valid),
        .pool2_addr     (pool2_addr),
        .pool2_value    (pool2_value),
        .gap_valid      (gap_valid),
        .gap_channel    (gap_channel),
        .gap_value      (gap_value),
        .fc_valid       (fc_valid),
        .fc_class       (fc_class),
        .fc_acc         (fc_acc)
    );

    // ================= golden memories (Part A) =================
    reg [7:0]  input_q    [0:783];
    reg [31:0] conv1_acc  [0:12543];
    reg [7:0]  stem_q     [0:12543];
    reg [7:0]  pool1_q    [0:3135];
    reg [31:0] conv2_acc  [0:6271];
    reg [7:0]  conv2_q    [0:6271];
    reg [7:0]  pool2_q    [0:1567];
    reg [31:0] conv3_acc  [0:1567];
    reg [7:0]  conv3_q    [0:1567];
    reg [7:0]  gap_q      [0:31];
    reg signed [31:0] fc_acc_gold [0:9];

    // current frame's expected fc_acc (golden or smoke)
    reg signed [31:0] fc_acc_cur [0:9];

    // ================= smoke samples (Part B) =================
    reg [7:0]  sm_input [0:783];
    reg signed [31:0] sm_fc_acc [0:9];

    // ================= statistics =================
    integer golden_mode;      // 1 = Part A compares every node; 0 = fc/pred only
    integer run_no;           // 0 = Part A, 1..10 = smoke digits
    integer s_cnt;            // stem acc/q pulse count
    integer p1_cnt;           // pool1 pulse count
    integer c2_cnt;           // conv2 acc/q pulse count
    integer c3_cnt;           // conv3 acc/q pulse count
    integer p2_cnt;           // pool2 pulse count
    integer g_cnt;            // gap pulse count
    integer f_cnt;            // fc pulse count
    integer s_bad, s_addr_bad;
    integer p1_bad, p1_addr_bad;
    integer c2_bad, c2_addr_bad;
    integer p2_bad, p2_addr_bad;
    integer c3_bad, c3_addr_bad;
    integer g_bad, g_ch_bad;
    integer fc_bad, fc_cls_bad;
    integer xz_bad;           // X/Z on status / valid-qualified buses
    integer ill_addr_bad;     // illegal RAM/ROM addresses
    integer done_count;       // core done pulses (expect 1 per run)
    integer stage_bad;        // stage sequence violations
    integer gap_bad;          // busy=0 && done=0 window before done
    integer started;          // a run is in progress (per run)
    integer done_seen;        // done observed (per run)
    integer same_start_bad;   // conv2/pool2 or conv3/gap not co-started
    integer same_done_bad;    // conv2/pool2 or conv3/gap not co-done
    integer retime_bad;       // engine EMIT / pipe out_valid misalignment
    integer cyc;              // global cycle counter
    integer start_cyc;
    integer core_done_cyc;
    integer stage_cyc [0:9];  // first cycle each controller stage was entered
    integer stem_done_cyc, conv2_start_cyc, pool2_done_cyc;
    integer conv3_start_cyc, gap_done_cyc, fc_start_cyc, fc_done_cyc;
    integer last_stage;
    integer stage_seen;       // distinct stage count in the run
    integer i;

    // watchdog: fires $fatal if any run hangs (about 1.53M cycles per run, 11
    // runs -> ~17M; give plenty of headroom but still abort a stuck run).
    initial begin : watchdog
        repeat (30000000) @(posedge clk);
        $fatal(1, "CORE: TIMEOUT - an inference run never finished");
    end

    // monitor: sample every clock edge (after the state settles).
    always @(posedge clk) begin
        #1;
        cyc = cyc + 1;

        // ---- per-stage first-entry recording + sequence check ----
        if (last_stage != dbg_stage) begin
            last_stage = dbg_stage;
            if (dbg_stage >= 0 && dbg_stage <= 9 && stage_seen < 10) begin
                if (stage_cyc[dbg_stage] == 0) stage_cyc[dbg_stage] = cyc;
                // expected sequence: 0,1,2,3,4,5,6,7,8,9,0
                if (stage_seen == 0) begin
                    if (dbg_stage !== 4'd0) stage_bad = stage_bad + 1;
                end else if (dbg_stage !== (stage_seen % 10)) begin
                    stage_bad = stage_bad + 1;
                end
                stage_seen = stage_seen + 1;
            end
        end

        // ---- core-level status ----
        if (done) begin
            done_count = done_count + 1;
            core_done_cyc = cyc;
            done_seen = 1;
        end
        if (started && !done_seen && !busy && !done) begin
            if (gap_bad < 20)
                $display("CORE BUSY-GAP cyc=%0d busy=0 done=0 before done", cyc);
            gap_bad = gap_bad + 1;
        end

        // ---- X/Z on status and valid-qualified buses ----
        if (busy === 1'bx || done === 1'bx || dbg_stage === 4'bx || dbg_stage === 4'bz ||
            (stem_acc_valid && (^stem_acc_value === 1'bx || ^stem_acc_addr === 1'bx)) ||
            (stem_q_valid  && (^stem_q_value === 1'bx || ^stem_q_addr === 1'bx)) ||
            (pool1_valid   && (^pool1_value === 1'bx || ^pool1_addr === 1'bx)) ||
            (conv_q_valid  && (^conv_acc_value === 1'bx || ^conv_q_value === 1'bx ||
                               ^conv_acc_addr === 1'bx || ^conv_q_addr === 1'bx)) ||
            (pool2_valid   && (^pool2_value === 1'bx || ^pool2_addr === 1'bx)) ||
            (gap_valid     && (^gap_value === 1'bx || ^gap_channel === 1'bx)) ||
            (fc_valid      && (^fc_acc === 1'bx || ^fc_class === 1'bx))) begin
            if (xz_bad < 20)
                $display("CORE XZ cyc=%0d busy=%b done=%b stage=%0d", cyc, busy, done, dbg_stage);
            xz_bad = xz_bad + 1;
        end

        // ---- no illegal RAM/ROM addresses (hierarchical into the core) ----
        if (u_core.u_conv.busy && u_core.u_conv.fm_raddr > 12'd3135) ill_addr_bad = ill_addr_bad + 1;
        if (u_core.u_conv.u_wt2_rom.addr > 13'd4607) ill_addr_bad = ill_addr_bad + 1;
        if (u_core.u_conv.u_wt3_rom.addr > 14'd9215) ill_addr_bad = ill_addr_bad + 1;
        if (u_core.u_fc.u_wt_rom.addr > 9'd319)      ill_addr_bad = ill_addr_bad + 1;

        // ---- co-start / co-done coordination checks ----
        // co-start: in CONV2_START both conv_start AND pool2_start are high in
        // the same cycle; in CONV3_START both conv_start AND gap_start are high.
        if (dbg_stage == 4'd3 && !(u_core.conv_start && u_core.pool2_start))
            same_start_bad = same_start_bad + 1;
        if (dbg_stage == 4'd5 && !(u_core.conv_start && u_core.gap_start))
            same_start_bad = same_start_bad + 1;
        // co-done: conv2_done == pool2_done on the same cycle (conv_layer=0),
        // conv3_done == gap_done on the same cycle (conv_layer=1).  Gate by
        // layer so each run only checks its own partner.
        if (!conv_layer && (u_core.u_conv.done != u_core.u_pool2.done))
            same_done_bad = same_done_bad + 1;
        if (conv_layer && (u_core.u_conv.done != u_core.u_gap.done))
            same_done_bad = same_done_bad + 1;

        // A+ retiming contract: neither engine may expose a state-count-only
        // EMIT.  EMIT, pipe out_valid and both debug valids are one-to-one.
        if ((u_core.u_stem_pipe.u_stem.state == 4'd7) !==
            u_core.u_stem_pipe.u_stem.req_out_valid)
            retime_bad = retime_bad + 1;
        if (stem_q_valid !== ((u_core.u_stem_pipe.u_stem.state == 4'd7) &&
                              u_core.u_stem_pipe.u_stem.req_out_valid))
            retime_bad = retime_bad + 1;
        if ((u_core.u_conv.state == 4'd7) !== u_core.u_conv.req_out_valid)
            retime_bad = retime_bad + 1;
        if (conv_q_valid !== ((u_core.u_conv.state == 4'd7) &&
                              u_core.u_conv.req_out_valid))
            retime_bad = retime_bad + 1;

        // ---- stage completion cycle capture ----
        if (u_core.u_stem_pipe.done && stem_done_cyc == 0) stem_done_cyc = cyc;
        if (dbg_stage == 4'd3 && conv2_start_cyc == 0)     conv2_start_cyc = cyc;
        if (u_core.u_pool2.done && pool2_done_cyc == 0)    pool2_done_cyc = cyc;
        if (dbg_stage == 4'd5 && conv3_start_cyc == 0)     conv3_start_cyc = cyc;
        if (u_core.u_gap.done && gap_done_cyc == 0)        gap_done_cyc = cyc;
        if (dbg_stage == 4'd7 && fc_start_cyc == 0)        fc_start_cyc = cyc;
        if (u_core.u_fc.done && fc_done_cyc == 0)          fc_done_cyc = cyc;

        // ---- stem acc/q stream (both pulse on valid-qualified S_EMIT) ----
        if (stem_q_valid && golden_mode) begin
            if (s_cnt < 12544) begin
                if (stem_acc_value !== $signed(conv1_acc[s_cnt]) ||
                    stem_q_value    !== stem_q[s_cnt]) begin
                    if (s_bad < 20)
                        $display("CORE STEM MISMATCH idx=%0d acc_exp=%0d acc_act=%0d q_exp=%0d q_act=%0d",
                                 s_cnt, $signed(conv1_acc[s_cnt]), $signed(stem_acc_value),
                                 stem_q[s_cnt], stem_q_value);
                    s_bad = s_bad + 1;
                end
                if (stem_acc_addr !== s_cnt[13:0] || stem_q_addr !== s_cnt[13:0]) begin
                    if (s_addr_bad < 20)
                        $display("CORE STEM ADDR idx=%0d acc_addr=%0d q_addr=%0d", s_cnt, stem_acc_addr, stem_q_addr);
                    s_addr_bad = s_addr_bad + 1;
                end
                s_cnt = s_cnt + 1;
            end
        end

        // ---- pool1 stream ----
        if (pool1_valid && golden_mode) begin
            if (p1_cnt < 3136) begin
                if (pool1_value !== pool1_q[p1_cnt]) begin
                    if (p1_bad < 20)
                        $display("CORE POOL1 MISMATCH idx=%0d exp=%0d act=%0d", p1_cnt, pool1_q[p1_cnt], pool1_value);
                    p1_bad = p1_bad + 1;
                end
                if (pool1_addr !== p1_cnt[11:0]) begin
                    if (p1_addr_bad < 20)
                        $display("CORE POOL1 ADDR idx=%0d addr=%0d", p1_cnt, pool1_addr);
                    p1_addr_bad = p1_addr_bad + 1;
                end
                p1_cnt = p1_cnt + 1;
            end
        end

        // ---- conv acc/q stream (conv2 during layer 0, conv3 during layer 1) ----
        if (conv_q_valid) begin
            if (conv_layer) begin
                if (golden_mode && c3_cnt < 1568) begin
                    if (conv_acc_value !== $signed(conv3_acc[c3_cnt]) ||
                        conv_q_value    !== conv3_q[c3_cnt]) begin
                        if (c3_bad < 20)
                            $display("CORE CONV3 MISMATCH idx=%0d acc_exp=%0d acc_act=%0d q_exp=%0d q_act=%0d",
                                     c3_cnt, $signed(conv3_acc[c3_cnt]), $signed(conv_acc_value),
                                     conv3_q[c3_cnt], conv_q_value);
                        c3_bad = c3_bad + 1;
                    end
                    if (conv_acc_addr !== c3_cnt[12:0] || conv_q_addr !== c3_cnt[12:0]) begin
                        if (c3_addr_bad < 20)
                            $display("CORE CONV3 ADDR idx=%0d acc_addr=%0d q_addr=%0d", c3_cnt, conv_acc_addr, conv_q_addr);
                        c3_addr_bad = c3_addr_bad + 1;
                    end
                end
                if (c3_cnt < 1568) c3_cnt = c3_cnt + 1;
            end else begin
                if (golden_mode && c2_cnt < 6272) begin
                    if (conv_acc_value !== $signed(conv2_acc[c2_cnt]) ||
                        conv_q_value    !== conv2_q[c2_cnt]) begin
                        if (c2_bad < 20)
                            $display("CORE CONV2 MISMATCH idx=%0d acc_exp=%0d acc_act=%0d q_exp=%0d q_act=%0d",
                                     c2_cnt, $signed(conv2_acc[c2_cnt]), $signed(conv_acc_value),
                                     conv2_q[c2_cnt], conv_q_value);
                        c2_bad = c2_bad + 1;
                    end
                    if (conv_acc_addr !== c2_cnt[12:0] || conv_q_addr !== c2_cnt[12:0]) begin
                        if (c2_addr_bad < 20)
                            $display("CORE CONV2 ADDR idx=%0d acc_addr=%0d q_addr=%0d", c2_cnt, conv_acc_addr, conv_q_addr);
                        c2_addr_bad = c2_addr_bad + 1;
                    end
                end
                if (c2_cnt < 6272) c2_cnt = c2_cnt + 1;
            end
        end

        // ---- pool2 stream ----
        if (pool2_valid && golden_mode) begin
            if (p2_cnt < 1568) begin
                if (pool2_value !== pool2_q[p2_cnt]) begin
                    if (p2_bad < 20)
                        $display("CORE POOL2 MISMATCH idx=%0d exp=%0d act=%0d", p2_cnt, pool2_q[p2_cnt], pool2_value);
                    p2_bad = p2_bad + 1;
                end
                if (pool2_addr !== p2_cnt[10:0]) begin
                    if (p2_addr_bad < 20)
                        $display("CORE POOL2 ADDR idx=%0d addr=%0d", p2_cnt, pool2_addr);
                    p2_addr_bad = p2_addr_bad + 1;
                end
                p2_cnt = p2_cnt + 1;
            end
        end

        // ---- gap stream (32 items; also counts the FC gap writes) ----
        if (gap_valid) begin
            if (golden_mode && g_cnt < 32) begin
                if (gap_value !== gap_q[g_cnt]) begin
                    if (g_bad < 20)
                        $display("CORE GAP MISMATCH ch=%0d exp=%0d act=%0d", g_cnt, gap_q[g_cnt], gap_value);
                    g_bad = g_bad + 1;
                end
                if (gap_channel !== g_cnt[4:0]) begin
                    if (g_ch_bad < 20)
                        $display("CORE GAP CHANNEL idx=%0d ch=%0d", g_cnt, gap_channel);
                    g_ch_bad = g_ch_bad + 1;
                end
            end
            if (g_cnt < 32) g_cnt = g_cnt + 1;
        end

        // ---- fc stream (10 items) ----
        if (fc_valid) begin
            if (f_cnt < 10) begin
                if (fc_acc !== $signed(fc_acc_cur[f_cnt])) begin
                    if (fc_bad < 20)
                        $display("CORE FC MISMATCH class=%0d exp=%0d act=%0d", f_cnt, $signed(fc_acc_cur[f_cnt]), $signed(fc_acc));
                    fc_bad = fc_bad + 1;
                end
                if (fc_class !== f_cnt[3:0]) begin
                    if (fc_cls_bad < 20)
                        $display("CORE FC CLASS idx=%0d class=%0d", f_cnt, fc_class);
                    fc_cls_bad = fc_cls_bad + 1;
                end
                f_cnt = f_cnt + 1;
            end
        end
    end

    // ================= write 784 input bytes (must be done in IDLE) =================
    task write_input;
        input [0:0] from_golden;    // 1 = golden input_q, 0 = sm_input
        integer i;
        begin
            input_we = 1'b1;
            for (i = 0; i < 784; i = i + 1) begin
                input_waddr = i[9:0];
                input_wdata = from_golden ? input_q[i] : sm_input[i];
                @(posedge clk);
            end
            input_we = 1'b0;
            @(posedge clk);
        end
    endtask

    // ================= one inference run (start -> done) =================
    task run_inference;
        input [3:0] expect_pred;
        integer ok;
        integer k;
        begin
            // reset per-run statistics (engines are idle now, safe)
            s_cnt = 0; p1_cnt = 0; c2_cnt = 0; c3_cnt = 0; p2_cnt = 0; g_cnt = 0; f_cnt = 0;
            s_bad = 0; s_addr_bad = 0; p1_bad = 0; p1_addr_bad = 0;
            c2_bad = 0; c2_addr_bad = 0; p2_bad = 0; p2_addr_bad = 0;
            c3_bad = 0; c3_addr_bad = 0; g_bad = 0; g_ch_bad = 0;
            fc_bad = 0; fc_cls_bad = 0; xz_bad = 0; ill_addr_bad = 0;
            done_count = 0; stage_bad = 0; gap_bad = 0; same_start_bad = 0; same_done_bad = 0;
            retime_bad = 0;
            start_cyc = 0; core_done_cyc = 0; stem_done_cyc = 0; conv2_start_cyc = 0;
            pool2_done_cyc = 0; conv3_start_cyc = 0; gap_done_cyc = 0; fc_start_cyc = 0; fc_done_cyc = 0;
            last_stage = -1; stage_seen = 0;
            for (k = 0; k < 10; k = k + 1) stage_cyc[k] = 0;
            started = 0; done_seen = 0;

            // ---- start ----
            start = 1'b1;
            @(posedge clk);
            start_cyc = cyc + 1;      // actual cycle index of the DUT start edge
            started = 1;              // busy must hold 1 until done from now on
            start = 1'b0;
            @(posedge clk);

            // ---- wait for the single core done pulse ----
            while (!done) @(posedge clk);
            #1;

            // ---- verdict ----
            ok = 1;
            if (golden_mode) begin
                if (s_cnt != 12544 || s_bad != 0 || s_addr_bad != 0 ||
                    p1_cnt != 3136 || p1_bad != 0 || p1_addr_bad != 0 ||
                    c2_cnt != 6272 || c2_bad != 0 || c2_addr_bad != 0 ||
                    p2_cnt != 1568 || p2_bad != 0 || p2_addr_bad != 0 ||
                    c3_cnt != 1568 || c3_bad != 0 || c3_addr_bad != 0 ||
                    g_cnt != 32 || g_bad != 0 || g_ch_bad != 0) ok = 0;
            end
            if (f_cnt != 10 || fc_bad != 0 || fc_cls_bad != 0) ok = 0;
            if (prediction !== expect_pred) ok = 0;
            if (done_count != 1 || xz_bad != 0 || ill_addr_bad != 0 ||
                stage_bad != 0 || gap_bad != 0 || same_start_bad != 0 ||
                same_done_bad != 0 || retime_bad != 0) ok = 0;
            // Every frame has a data-independent schedule under the serial
            // engines; explicitly assert the A+ P=3 end-to-end contract.
            if (core_done_cyc - start_cyc != 1590315) ok = 0;

            $display("CORE RUN[%0d] golden=%0d s=%0d/12544 p1=%0d/3136 c2=%0d/6272 p2=%0d/1568 c3=%0d/1568 g=%0d/32 f=%0d/10 pred=%0d (expect %0d)",
                     run_no, golden_mode, s_cnt, p1_cnt, c2_cnt, p2_cnt, c3_cnt, g_cnt, f_cnt, prediction, expect_pred);
            $display("CORE RUN[%0d] bad s=%0d/%0d p1=%0d/%0d c2=%0d/%0d p2=%0d/%0d c3=%0d/%0d g=%0d/%0d fc=%0d/%0d xz=%0d ill=%0d",
                     run_no, s_bad, s_addr_bad, p1_bad, p1_addr_bad, c2_bad, c2_addr_bad,
                     p2_bad, p2_addr_bad, c3_bad, c3_addr_bad, g_bad, g_ch_bad, fc_bad, fc_cls_bad, xz_bad, ill_addr_bad);
            $display("CORE RUN[%0d] status done=%0d stage_bad=%0d gap_bad=%0d same_start=%0d same_done=%0d",
                     run_no, done_count, stage_bad, gap_bad, same_start_bad, same_done_bad);
            $display("CORE RUN[%0d] retime_bad=%0d start->done=%0d (expect 1590315)",
                     run_no, retime_bad, core_done_cyc - start_cyc);
            if (golden_mode) begin
                $display("CORE RUN[%0d] cycles start->stem_done=%0d conv2_start->pool2_done=%0d conv3_start->gap_done=%0d fc_start->fc_done=%0d start->core_done=%0d",
                         run_no, stem_done_cyc - start_cyc, pool2_done_cyc - conv2_start_cyc,
                         gap_done_cyc - conv3_start_cyc, fc_done_cyc - fc_start_cyc, core_done_cyc - start_cyc);
            end

            if (!ok)
                $fatal(1, "CORE RUN[%0d] FAILED (pred=%0d expect=%0d s=%0d p1=%0d c2=%0d p2=%0d c3=%0d g=%0d f=%0d done=%0d stage=%0d gap=%0d ss=%0d sd=%0d xz=%0d ill=%0d)",
                       run_no, prediction, expect_pred, s_cnt, p1_cnt, c2_cnt, p2_cnt, c3_cnt,
                       g_cnt, f_cnt, done_count, stage_bad, gap_bad, same_start_bad, same_done_bad, xz_bad, ill_addr_bad);
            started = 0;
        end
    endtask

    // ================= main =================
    initial begin
        // ---- load the golden trace (Part A) ----
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/input_q.mem",   input_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv1_acc.mem", conv1_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/stem_q.mem",    stem_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/pool1_q.mem",   pool1_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv2_acc.mem", conv2_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv2_q.mem",   conv2_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/pool2_q.mem",   pool2_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_acc.mem", conv3_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_q.mem",   conv3_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/gap_q.mem",     gap_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/fc_acc.mem",    fc_acc_gold);

        // sentinel guards (missing / truncated files abort instead of passing)
        if (input_q[0]   !== 8'hED)      $fatal(1, "CORE: input_q.mem not loaded (input_q[0]=%h)", input_q[0]);
        if (conv1_acc[0] !== 32'h000012DC) $fatal(1, "CORE: conv1_acc.mem not loaded (conv1_acc[0]=%h)", conv1_acc[0]);
        if (conv1_acc[12543] !== 32'h000002BB) $fatal(1, "CORE: conv1_acc.mem truncated (conv1_acc[12543]=%h)", conv1_acc[12543]);
        if (stem_q[0]    !== 8'h24)      $fatal(1, "CORE: stem_q.mem not loaded (stem_q[0]=%h)", stem_q[0]);
        if (stem_q[12543] !== 8'h05)     $fatal(1, "CORE: stem_q.mem truncated (stem_q[12543]=%h)", stem_q[12543]);
        if (pool1_q[0]   !== 8'h24)      $fatal(1, "CORE: pool1_q.mem not loaded (pool1_q[0]=%h)", pool1_q[0]);
        if (conv2_q[0]   !== 8'h10)      $fatal(1, "CORE: conv2_q.mem not loaded (conv2_q[0]=%h)", conv2_q[0]);
        if (pool2_q[0]   !== 8'h1F)      $fatal(1, "CORE: pool2_q.mem not loaded (pool2_q[0]=%h)", pool2_q[0]);
        if (conv3_acc[0] !== 32'hFFFFF7E6) $fatal(1, "CORE: conv3_acc.mem not loaded (conv3_acc[0]=%h)", conv3_acc[0]);
        if (gap_q[0]     !== 8'h04)      $fatal(1, "CORE: gap_q.mem not loaded (gap_q[0]=%h)", gap_q[0]);
        if (fc_acc_gold[8] !== 32'h00003D8A) $fatal(1, "CORE: fc_acc.mem wrong (fc_acc[8]=%h)", fc_acc_gold[8]);
        if (fc_acc_gold[9] !== 32'hFFFFF193) $fatal(1, "CORE: fc_acc.mem truncated (fc_acc[9]=%h)", fc_acc_gold[9]);

        cyc = 0;              // the monitor increments cyc from the first posedge

        // ---- reset ----
        rst_n = 1'b0;
        start = 1'b0;
        input_we = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // ---- Part A: golden digit8 trace ----
        run_no = 0;
        golden_mode = 1;
        for (i = 0; i < 10; i = i + 1) fc_acc_cur[i] = fc_acc_gold[i];
        write_input(1'b1);
        run_inference(4'd8);

        // ---- Part B: 10 smoke samples, NO global reset between runs ----
        golden_mode = 0;
        begin : smoke_loop
            integer d;
            for (d = 0; d < 10; d = d + 1) begin
                case (d)
                    0: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit0_idx00003/input_q.mem", sm_input);
                    1: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit1_idx00002/input_q.mem", sm_input);
                    2: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit2_idx00001/input_q.mem", sm_input);
                    3: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit3_idx00018/input_q.mem", sm_input);
                    4: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit4_idx00004/input_q.mem", sm_input);
                    5: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit5_idx00008/input_q.mem", sm_input);
                    6: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit6_idx00011/input_q.mem", sm_input);
                    7: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit7_idx00000/input_q.mem", sm_input);
                    8: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit8_idx00061/input_q.mem", sm_input);
                    9: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit9_idx00007/input_q.mem", sm_input);
                endcase
                case (d)
                    0: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit0_idx00003/fc_acc.mem", sm_fc_acc);
                    1: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit1_idx00002/fc_acc.mem", sm_fc_acc);
                    2: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit2_idx00001/fc_acc.mem", sm_fc_acc);
                    3: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit3_idx00018/fc_acc.mem", sm_fc_acc);
                    4: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit4_idx00004/fc_acc.mem", sm_fc_acc);
                    5: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit5_idx00008/fc_acc.mem", sm_fc_acc);
                    6: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit6_idx00011/fc_acc.mem", sm_fc_acc);
                    7: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit7_idx00000/fc_acc.mem", sm_fc_acc);
                    8: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit8_idx00061/fc_acc.mem", sm_fc_acc);
                    9: $readmemh("fpga/baseline_cnn/sim/vectors/smoke/digit9_idx00007/fc_acc.mem", sm_fc_acc);
                endcase
                // sentinel: the smoke file must have loaded (all-X => empty/truncated)
                if (^sm_fc_acc[0] === 1'bx) $fatal(1, "CORE: smoke fc_acc.mem not loaded for digit %0d", d);
                for (i = 0; i < 10; i = i + 1) fc_acc_cur[i] = sm_fc_acc[i];
                run_no = d + 1;
                write_input(1'b0);
                run_inference(d[3:0]);
            end
        end

        $display("CORE_ALL_PASS");
        $finish;
    end
endmodule
