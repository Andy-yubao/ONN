// tb_conv2_pool2.v - Golden-vector verification of conv2 + pool2.
//
// Sample: digit 8, MNIST test index 61 (sim/vectors/golden_trace).
// Golden vectors: pool1_q.mem (3136, conv2 INPUT), conv2_acc.mem (6272),
// conv2_q.mem (6272), pool2_q.mem (1568).
//
// Topology under test:
//     pool1_q RAM (external) -> conv_u8_serial (layer_sel=0) -> q stream
//     -> maxpool2x2_stream (N_CH=32, IN_H=14, IN_W=14, OC_W=5, XY_W=4,
//        OUT_ADDR_W=11) -> pool2_q
//
// Flow:
//   1. reset
//   2. load 3136 pool1_q bytes into the external FM RAM (write port)
//   3. assert ONE `start` - it launches conv2 AND pool2 TOGETHER (pool is NOT
//      started by conv2_done; conv2_done fires only after all 6272 q have gone)
//   4. compare the conv2 acc/q streams against conv2_acc/conv2_q and the pool2
//      stream against pool2_q, bit-exactly; addresses strictly 0..6271 / 0..1567
//   5. frozen assertions: q_valid => pool_busy; no illegal fm_raddr (>3135);
//      conv_done and pool_done fire on the same cycle; done single pulse; no X/Z
//   6. report per-output / total cycle counts
//
// Any mismatch prints the first 20 failures and terminates with $fatal (non-zero
// vsim exit).  A watchdog aborts with $fatal if the run never finishes.
`timescale 1ns/1ps

module tb_conv2_pool2;
    // ================= clock =================
    reg clk = 1'b0;
    always #5 clk = ~clk;             // 10 ns period

    // ================= DUT =================
    reg rst_n = 1'b0;
    reg start = 1'b0;
    reg layer_sel = 1'b0;   // 0 = conv2
    wire busy;
    wire done;
    wire acc_valid;
    wire [12:0] acc_addr;
    wire signed [31:0] acc_value;
    wire q_valid;
    wire [12:0] q_addr;
    wire [7:0] q_value;
    wire [11:0] fm_raddr;
    wire [7:0] fm_rdata;
    wire conv_busy, conv_done;
    wire pool_busy, pool_done;
    wire pool_valid;
    wire [10:0] pool_addr;
    wire [7:0] pool_value;

    // external input feature-map RAM (sync_ram_u8, one-cycle read latency)
    reg fm_we = 1'b0;
    reg [11:0] fm_waddr = 12'd0;
    reg [7:0]  fm_wdata = 8'd0;

    sync_ram_u8 #(.DEPTH(3136), .ADDR_W(12)) u_fm_ram (
        .clk   (clk),
        .we    (fm_we),
        .waddr (fm_waddr),
        .wdata (fm_wdata),
        .raddr (fm_raddr),
        .rdata (fm_rdata)
    );

    conv_u8_serial u_conv (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start),
        .layer_sel     (layer_sel),
        .busy          (conv_busy),
        .done          (conv_done),
        .fm_raddr      (fm_raddr),
        .fm_rdata      (fm_rdata),
        .acc_valid     (acc_valid),
        .acc_addr      (acc_addr),
        .acc_value     (acc_value),
        .q_valid       (q_valid),
        .q_addr        (q_addr),
        .q_value       (q_value)
    );

    maxpool2x2_stream #(
        .N_CH(32), .IN_H(14), .IN_W(14),
        .OC_W(5), .XY_W(4), .OUT_ADDR_W(11)
    ) u_pool (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),          // SAME start as conv2 (together)
        .in_valid  (q_valid),
        .in_q      (q_value),
        .busy      (pool_busy),
        .out_valid (pool_valid),
        .out_addr  (pool_addr),
        .out_q     (pool_value),
        .done      (pool_done)
    );

    assign busy = conv_busy || pool_busy;
    assign done = pool_done;         // completion = pool2 done (same cycle as conv_done)

    // ================= golden memories =================
    reg [7:0]  pool1_q   [0:3135];
    reg [31:0] conv2_acc [0:6271];
    reg [7:0]  conv2_q   [0:6271];
    reg [7:0]  pool2_q   [0:1567];

    // ================= statistics / monitors =================
    integer cmp_idx;          // next expected conv2 output index (0..6271)
    integer pool_cmp;         // next expected pool2 output index (0..1567)
    integer qcnt;             // conv2 q_valid pulse count (expect 6272)
    integer pcnt;             // pool2 out_valid pulse count (expect 1568)
    integer mismatches;       // conv2 acc/q stream mismatches vs golden
    integer addr_bad;         // conv2 acc_addr/q_addr ordering mismatches
    integer pool_bad;         // pool2 value mismatches vs golden
    integer pool_addr_bad;    // pool2 addr continuity mismatches
    integer xz_bad;           // X/Z on the streams / status
    integer busy_assn_bad;    // q_valid while pool_busy == 0
    integer illegal_addr_bad; // fm_raddr > 3135 during the run
    integer rom_ovr_bad;      // weight-ROM gating / out-of-depth read
    integer post_bias_bad;    // ADD_BIAS edge captured pre-bias/stale S64
    integer emit_valid_bad;   // S_EMIT and requant pipe out_valid misaligned
    integer done_count;       // pool_done pulse count (expect 1)
    integer conv_done_count;  // conv_done pulse count (expect 1)
    integer busy_cycles;      // cycles busy == 1
    integer cyc;              // global cycle counter
    integer start_cyc;
    integer first_q_cyc, first_pool_cyc;
    integer conv_done_cyc, pool_done_cyc, done_cyc;
    integer i;

    // watchdog: fires $fatal if the run hangs (done never asserted).
    initial begin : watchdog
        repeat (2000000) @(posedge clk);
        $fatal(1, "C2P2: TIMEOUT - conv2+pool2 run stuck, done never asserted");
    end

    // monitor: sample every clock edge (after the state settles).
    always @(posedge clk) begin
        #1;
        cyc = cyc + 1;
        if (busy) busy_cycles = busy_cycles + 1;
        if (conv_done) begin
            conv_done_count = conv_done_count + 1;
            conv_done_cyc = cyc;
        end
        if (pool_done) begin
            done_count = done_count + 1;
            pool_done_cyc = cyc;
        end
        if (done) done_cyc = cyc;

        // X/Z on the observable bus
        if (busy === 1'bx || done === 1'bx || q_valid === 1'bx || q_valid === 1'bz ||
            pool_valid === 1'bx || pool_valid === 1'bz ||
            ^acc_addr === 1'bx || ^q_addr === 1'bx || ^pool_addr === 1'bx ||
            ^q_value === 1'bx || ^pool_value === 1'bx) begin
            if (xz_bad < 20)
                $display("C2P2 XZ cyc=%0d busy=%b done=%b q_valid=%b pool_valid=%b", cyc, busy, done, q_valid, pool_valid);
            xz_bad = xz_bad + 1;
        end

        // no illegal FM address while conv2 is running (padding forces 0)
        if (conv_busy && fm_raddr > 12'd3135) begin
            if (illegal_addr_bad < 20)
                $display("C2P2 ILLEGAL-FM cyc=%0d fm_raddr=%0d (>3135)", cyc, fm_raddr);
            illegal_addr_bad = illegal_addr_bad + 1;
        end

        // weight-ROM gating / out-of-depth (hierarchical into u_conv):
        //   - the conv3 weight ROM must be fixed at 0 during a conv2 run
        //     (layer-gated reads, never a stray address)
        //   - no weight ROM may ever be addressed beyond its own depth
        //     (conv2 4608, conv3 9216)
        if (conv_busy) begin
            if (u_conv.u_wt2_rom.addr > 13'd4607) begin
                if (rom_ovr_bad < 20)
                    $display("C2P2 WT2-OVR cyc=%0d wt2_addr=%0d (>4607)", cyc, u_conv.u_wt2_rom.addr);
                rom_ovr_bad = rom_ovr_bad + 1;
            end
            if (u_conv.u_wt3_rom.addr !== 14'd0) begin
                if (rom_ovr_bad < 20)
                    $display("C2P2 WT3-GATE cyc=%0d wt3_addr=%0d (expect 0 during conv2)", cyc, u_conv.u_wt3_rom.addr);
                rom_ovr_bad = rom_ovr_bad + 1;
            end
        end

        // P=3 retiming invariants.  S_SAT32 holds the token captured on the
        // preceding ADD_BIAS edge; for these frozen vectors it must equal the
        // golden post-bias accumulator exactly (all values fit S32).  S_EMIT
        // and the pipeline out_valid must be strictly one-to-one.
        if (u_conv.state == 4'd4 && u_conv.token_addr < 6272 &&
            u_conv.token_acc64 !== $signed(conv2_acc[u_conv.token_addr]))
            post_bias_bad = post_bias_bad + 1;
        if ((u_conv.state == 4'd7) !== u_conv.req_out_valid)
            emit_valid_bad = emit_valid_bad + 1;
        if (acc_valid !== ((u_conv.state == 4'd7) && u_conv.req_out_valid))
            emit_valid_bad = emit_valid_bad + 1;
        if (q_valid !== ((u_conv.state == 4'd7) && u_conv.req_out_valid))
            emit_valid_bad = emit_valid_bad + 1;
        if (!u_conv.req_out_valid &&
            (acc_valid !== 1'b0 || q_valid !== 1'b0 ||
             acc_addr !== 13'd0 || q_addr !== 13'd0 ||
             acc_value !== 32'sd0 || q_value !== 8'd0))
            emit_valid_bad = emit_valid_bad + 1;

        // ---- conv2 acc/q debug stream ----
        if (q_valid) begin
            qcnt = qcnt + 1;
            if (cmp_idx < 6272) begin
                if (acc_value !== $signed(conv2_acc[cmp_idx]) ||
                    q_value    !== conv2_q[cmp_idx]) begin
                    if (mismatches < 20)
                        $display("C2P2 CONV2 MISMATCH idx=%0d oc=%0d y=%0d x=%0d acc_exp=%0d acc_act=%0d q_exp=%0d q_act=%0d",
                                 cmp_idx, cmp_idx/196, (cmp_idx%196)/14, cmp_idx%14,
                                 $signed(conv2_acc[cmp_idx]), $signed(acc_value),
                                 conv2_q[cmp_idx], q_value);
                    mismatches = mismatches + 1;
                end
                if (acc_addr !== cmp_idx[12:0] || q_addr !== cmp_idx[12:0]) begin
                    if (addr_bad < 20)
                        $display("C2P2 CONV2 ADDR idx=%0d acc_addr=%0d q_addr=%0d (expect %0d)",
                                 cmp_idx, acc_addr, q_addr, cmp_idx);
                    addr_bad = addr_bad + 1;
                end
                if (cmp_idx == 0) first_q_cyc = cyc;
                // frozen: every conv2 q_valid must arrive while pool2 is busy
                if (!pool_busy) begin
                    if (busy_assn_bad < 20)
                        $display("C2P2 BUSY-ASSN cyc=%0d q_valid=1 but pool_busy=0", cyc);
                    busy_assn_bad = busy_assn_bad + 1;
                end
                cmp_idx = cmp_idx + 1;
            end
        end

        // ---- pool2 stream ----
        if (pool_valid) begin
            pcnt = pcnt + 1;
            if (pool_cmp < 1568) begin
                if (pool_value !== pool2_q[pool_cmp]) begin
                    if (pool_bad < 20)
                        $display("C2P2 POOL2 MISMATCH idx=%0d oc=%0d py=%0d px=%0d exp=%0d act=%0d",
                                 pool_cmp, pool_cmp/49, (pool_cmp%49)/7, pool_cmp%7,
                                 pool2_q[pool_cmp], pool_value);
                    pool_bad = pool_bad + 1;
                end
                if (pool_addr !== pool_cmp[10:0]) begin
                    if (pool_addr_bad < 20)
                        $display("C2P2 POOL2 ADDR idx=%0d oc=%0d py=%0d px=%0d pool_addr=%0d (expect %0d)",
                                 pool_cmp, pool_cmp/49, (pool_cmp%49)/7, pool_cmp%7,
                                 pool_addr, pool_cmp);
                    pool_addr_bad = pool_addr_bad + 1;
                end
                if (pool_cmp == 0) first_pool_cyc = cyc;
                pool_cmp = pool_cmp + 1;
            end
        end
    end

    initial begin
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/pool1_q.mem",  pool1_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv2_acc.mem", conv2_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv2_q.mem",   conv2_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/pool2_q.mem",   pool2_q);

        // sentinel guards (missing / truncated file aborts instead of passing)
        if (pool1_q[0]    !== 8'h24)        $fatal(1, "C2P2: pool1_q.mem not loaded (pool1_q[0]=%h)", pool1_q[0]);
        if (pool1_q[3135] !== 8'h05)        $fatal(1, "C2P2: pool1_q.mem truncated (pool1_q[3135]=%h)", pool1_q[3135]);
        if (conv2_acc[0]  !== 32'h00000F8A) $fatal(1, "C2P2: conv2_acc.mem not loaded (conv2_acc[0]=%h)", conv2_acc[0]);
        if (conv2_q[0]    !== 8'h10)        $fatal(1, "C2P2: conv2_q.mem not loaded (conv2_q[0]=%h)", conv2_q[0]);
        if (conv2_q[6271] !== 8'h11)        $fatal(1, "C2P2: conv2_q.mem truncated (conv2_q[6271]=%h)", conv2_q[6271]);
        if (pool2_q[0]    !== 8'h1F)        $fatal(1, "C2P2: pool2_q.mem not loaded (pool2_q[0]=%h)", pool2_q[0]);
        if (pool2_q[1567] !== 8'h1F)        $fatal(1, "C2P2: pool2_q.mem truncated (pool2_q[1567]=%h)", pool2_q[1567]);

        cmp_idx = 0; pool_cmp = 0; qcnt = 0; pcnt = 0;
        mismatches = 0; addr_bad = 0; pool_bad = 0; pool_addr_bad = 0;
        xz_bad = 0; busy_assn_bad = 0; illegal_addr_bad = 0; rom_ovr_bad = 0;
        post_bias_bad = 0; emit_valid_bad = 0;
        done_count = 0; conv_done_count = 0; busy_cycles = 0; cyc = 0;
        start_cyc = 0; first_q_cyc = 0; first_pool_cyc = 0;
        conv_done_cyc = 0; pool_done_cyc = 0; done_cyc = 0;

        // ---- 1. reset ----
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // ---- 2. load pool1_q into the external FM RAM ----
        for (i = 0; i < 3136; i = i + 1) begin
            fm_we    = 1'b1;
            fm_waddr = i[11:0];
            fm_wdata = pool1_q[i];
            @(posedge clk);
        end
        fm_we = 1'b0;
        @(posedge clk);

        // ---- 3. start: launches conv2 AND pool2 together ----
        start = 1'b1;
        @(posedge clk);
        start_cyc = cyc + 1;      // actual cycle index of the DUT start edge
        start = 1'b0;

        // ---- 4/5. wait for the single done pulse ----
        while (!done) @(posedge clk);
        #1;

        // ---- report ----
        $display("C2P2: qcnt=%0d/6272 pcnt=%0d/1568 conv2_mismatches=%0d addr_bad=%0d pool_mismatches=%0d pool_addr_bad=%0d",
                 qcnt, pcnt, mismatches, addr_bad, pool_bad, pool_addr_bad);
        $display("C2P2: xz_bad=%0d busy_assn_bad=%0d illegal_addr_bad=%0d done=%0d conv_done=%0d busy_cycles=%0d",
                 xz_bad, busy_assn_bad, illegal_addr_bad, done_count, conv_done_count, busy_cycles);
        $display("C2P2: start=%0d first_q=%0d first_pool=%0d conv_done=%0d pool_done=%0d done=%0d",
                 start_cyc, first_q_cyc, first_pool_cyc, conv_done_cyc, pool_done_cyc, done_cyc);
        $display("C2P2: first-q=%0d (expect 149) start->done=%0d (expect 940801) busy_cycles=%0d post_bias_bad=%0d emit_valid_bad=%0d",
                 first_q_cyc - start_cyc, done_cyc - start_cyc, busy_cycles,
                 post_bias_bad, emit_valid_bad);

        // ---- verdict ----
        if (qcnt != 6272 || pcnt != 1568 || mismatches != 0 || addr_bad != 0 ||
            pool_bad != 0 || pool_addr_bad != 0 || xz_bad != 0 || busy_assn_bad != 0 ||
            illegal_addr_bad != 0 || rom_ovr_bad != 0 || post_bias_bad != 0 || emit_valid_bad != 0 ||
            done_count != 1 || conv_done_count != 1 ||
            busy_cycles != done_cyc - start_cyc ||
            conv_done_cyc != pool_done_cyc || pool_done_cyc != done_cyc ||
            first_q_cyc - start_cyc != 149 || done_cyc - start_cyc != 940801) begin
            $fatal(1, "C2P2 FAILED (qcnt=%0d pcnt=%0d mism=%0d addr=%0d pool=%0d paddr=%0d xz=%0d busy=%0d ill=%0d rom=%0d done=%0d cdone=%0d)",
                   qcnt, pcnt, mismatches, addr_bad, pool_bad, pool_addr_bad,
                   xz_bad, busy_assn_bad, illegal_addr_bad, rom_ovr_bad, done_count, conv_done_count);
        end

        $display("C2P2_ALL_PASS");
        $finish;
    end
endmodule
