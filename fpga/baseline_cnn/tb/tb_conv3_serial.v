// tb_conv3_serial.v - Golden-vector verification of conv3 (shared engine).
//
// Sample: digit 8, MNIST test index 61 (sim/vectors/golden_trace).
// Golden vectors: pool2_q.mem (1568, conv3 INPUT), conv3_acc.mem (1568),
// conv3_q.mem (1568).
//
// Topology under test:
//     pool2_q RAM (external) -> conv_u8_serial (layer_sel=1)
//
// Flow (per pass):
//   1. reset
//   2. load 1568 pool2_q bytes into the external FM RAM (write port)
//   3. assert start with layer_sel=1 (conv3)
//   4. compare the acc/q streams against conv3_acc/conv3_q bit-exactly;
//      addresses strictly 0..1567; no illegal fm_raddr (>1567); no X/Z
//   5. wait for the single done pulse
//   6. pass 1 records every output; pass 2 (full reset + rerun) must be
//      bit-identical to pass 1
//
// Cycle budget: per-output 1+288+1+1 = 291 (Cin=32); first q at start+290;
// start->done = 1568*291 + 1 = 456289 (the +1 is the S_DONE cycle; the done
// pulse itself is the cycle after busy drops).
//
// Any mismatch prints the first 20 failures and terminates with $fatal (non-zero
// vsim exit).  A watchdog aborts with $fatal if a pass never finishes.
`timescale 1ns/1ps

module tb_conv3_serial;
    // ================= clock =================
    reg clk = 1'b0;
    always #5 clk = ~clk;             // 10 ns period

    // ================= DUT =================
    reg rst_n = 1'b0;
    reg start = 1'b0;
    reg layer_sel = 1'b1;   // 1 = conv3
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

    // external input feature-map RAM (sync_ram_u8, one-cycle read latency)
    reg fm_we = 1'b0;
    reg [11:0] fm_waddr = 12'd0;
    reg [7:0]  fm_wdata = 8'd0;

    sync_ram_u8 #(.DEPTH(1568), .ADDR_W(12)) u_fm_ram (
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
        .busy          (busy),
        .done          (done),
        .fm_raddr      (fm_raddr),
        .fm_rdata      (fm_rdata),
        .acc_valid     (acc_valid),
        .acc_addr      (acc_addr),
        .acc_value     (acc_value),
        .q_valid       (q_valid),
        .q_addr        (q_addr),
        .q_value       (q_value)
    );

    // ================= golden memories =================
    reg [7:0]  pool2_q   [0:1567];
    reg [31:0] conv3_acc [0:1567];
    reg [7:0]  conv3_q   [0:1567];

    // ================= statistics =================
    integer passid;          // 0 = pass 1, 1 = pass 2
    integer cmp_idx;         // next expected conv3 output index (0..1567)
    integer qcnt;            // q_valid pulse count (expect 1568)
    integer mismatches;      // acc/q stream mismatches vs golden
    integer addr_bad;        // acc_addr/q_addr ordering mismatches
    integer xz_bad;          // X/Z on the streams / status
    integer illegal_addr_bad;// fm_raddr > 1567 during the run
    integer cross_bad;       // pass 2 output != pass 1 output
    integer done_count;      // done pulse count (expect 1)
    integer busy_cycles;     // cycles busy == 1
    integer cyc;             // global cycle counter
    integer start_cyc;
    integer first_q_cyc;
    integer done_cyc;
    integer i;
    reg [31:0] accA [0:1567];    // pass-1 recorded acc (cross-check)
    reg [7:0]  qA    [0:1567];   // pass-1 recorded q (cross-check)

    // watchdog: fires $fatal if a pass hangs (done never asserted).
    initial begin : watchdog
        repeat (1000000) @(posedge clk);
        $fatal(1, "C3: TIMEOUT - conv3 pass stuck, done never asserted");
    end

    // monitor: sample every clock edge (after the state settles).
    always @(posedge clk) begin
        #1;
        cyc = cyc + 1;
        if (busy) busy_cycles = busy_cycles + 1;
        if (done) begin
            done_count = done_count + 1;
            done_cyc = cyc;
        end

        // X/Z on the observable bus
        if (busy === 1'bx || done === 1'bx || q_valid === 1'bx || q_valid === 1'bz ||
            ^acc_addr === 1'bx || ^q_addr === 1'bx || ^q_value === 1'bx ||
            ^acc_value === 1'bx) begin
            if (xz_bad < 20)
                $display("C3 XZ cyc=%0d busy=%b done=%b q_valid=%b", cyc, busy, done, q_valid);
            xz_bad = xz_bad + 1;
        end

        // no illegal FM address while conv3 is running (padding forces 0)
        if (busy && fm_raddr > 12'd1567) begin
            if (illegal_addr_bad < 20)
                $display("C3 ILLEGAL-FM cyc=%0d fm_raddr=%0d (>1567)", cyc, fm_raddr);
            illegal_addr_bad = illegal_addr_bad + 1;
        end

        // conv3 acc/q stream
        if (q_valid) begin
            qcnt = qcnt + 1;
            if (cmp_idx < 1568) begin
                if (acc_value !== $signed(conv3_acc[cmp_idx]) ||
                    q_value    !== conv3_q[cmp_idx]) begin
                    if (mismatches < 20)
                        $display("C3 MISMATCH idx=%0d oc=%0d y=%0d x=%0d acc_exp=%0d acc_act=%0d q_exp=%0d q_act=%0d",
                                 cmp_idx, cmp_idx/49, (cmp_idx%49)/7, cmp_idx%7,
                                 $signed(conv3_acc[cmp_idx]), $signed(acc_value),
                                 conv3_q[cmp_idx], q_value);
                    mismatches = mismatches + 1;
                end
                if (acc_addr !== cmp_idx[12:0] || q_addr !== cmp_idx[12:0]) begin
                    if (addr_bad < 20)
                        $display("C3 ADDR idx=%0d acc_addr=%0d q_addr=%0d (expect %0d)",
                                 cmp_idx, acc_addr, q_addr, cmp_idx);
                    addr_bad = addr_bad + 1;
                end
                if (cmp_idx == 0) first_q_cyc = cyc;
                if (passid == 0) begin
                    accA[cmp_idx] = acc_value;
                    qA[cmp_idx]   = q_value;
                end else if (acc_value !== $signed(accA[cmp_idx]) ||
                             q_value    !== qA[cmp_idx]) begin
                    if (cross_bad < 20)
                        $display("C3 CROSS pass2 idx=%0d acc_expA=%0d acc_act=%0d q_expA=%0d q_act=%0d",
                                 cmp_idx, $signed(accA[cmp_idx]), $signed(acc_value), qA[cmp_idx], q_value);
                    cross_bad = cross_bad + 1;
                end
                cmp_idx = cmp_idx + 1;
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
            fm_we = 1'b0;
            fm_waddr = 12'd0;
            fm_wdata = 8'd0;
            layer_sel = 1'b1;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);

            // ---- reset statistics ----
            passid = passno - 1;
            cmp_idx = 0; qcnt = 0; mismatches = 0; addr_bad = 0;
            xz_bad = 0; illegal_addr_bad = 0; cross_bad = 0;
            done_count = 0; busy_cycles = 0; cyc = 0;
            start_cyc = 0; first_q_cyc = 0; done_cyc = 0;

            // ---- load pool2_q into the external FM RAM ----
            for (i = 0; i < 1568; i = i + 1) begin
                fm_we    = 1'b1;
                fm_waddr = i[11:0];
                fm_wdata = pool2_q[i];
                @(posedge clk);
            end
            fm_we = 1'b0;
            @(posedge clk);

            // ---- start (single-cycle pulse), layer_sel=1 = conv3 ----
            start = 1'b1;
            @(posedge clk);
            start_cyc = cyc + 1;      // actual cycle index of the DUT start edge
            start = 1'b0;

            // ---- wait for the single done pulse ----
            while (!done) @(posedge clk);
            #1;

            // ---- report ----
            $display("C3 PASS[%0d] qcnt=%0d/1568 mismatches=%0d addr_bad=%0d xz_bad=%0d illegal_addr_bad=%0d cross_bad=%0d done=%0d",
                     passno, qcnt, mismatches, addr_bad, xz_bad, illegal_addr_bad, cross_bad, done_count);
            $display("C3 PASS[%0d] start=%0d first_q=%0d done=%0d per-output=%0d start->done=%0d busy_cycles=%0d",
                     passno, start_cyc, first_q_cyc, done_cyc,
                     first_q_cyc - start_cyc, done_cyc - start_cyc, busy_cycles);

            ok = (qcnt == 1568 && mismatches == 0 && addr_bad == 0 &&
                  xz_bad == 0 && illegal_addr_bad == 0 && cross_bad == 0 &&
                  done_count == 1 && busy_cycles > 0 &&
                  busy_cycles == done_cyc - start_cyc &&
                  first_q_cyc - start_cyc == 290 && done_cyc - start_cyc == 456289);
            if (!ok)
                $fatal(1, "C3 PASS[%0d] FAILED (qcnt=%0d mism=%0d addr=%0d xz=%0d ill=%0d cross=%0d done=%0d)",
                       passno, qcnt, mismatches, addr_bad, xz_bad, illegal_addr_bad, cross_bad, done_count);
        end
    endtask

    // ================= main =================
    initial begin
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/pool2_q.mem",  pool2_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_acc.mem", conv3_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_q.mem",   conv3_q);

        // sentinel guards (missing / truncated file aborts instead of passing)
        if (pool2_q[0]    !== 8'h1F)        $fatal(1, "C3: pool2_q.mem not loaded (pool2_q[0]=%h)", pool2_q[0]);
        if (pool2_q[1567] !== 8'h1F)        $fatal(1, "C3: pool2_q.mem truncated (pool2_q[1567]=%h)", pool2_q[1567]);
        if (conv3_acc[0]  !== 32'hFFFFF7E6) $fatal(1, "C3: conv3_acc.mem not loaded (conv3_acc[0]=%h)", conv3_acc[0]);
        if (conv3_q[0]    !== 8'h00)        $fatal(1, "C3: conv3_q.mem not loaded (conv3_q[0]=%h)", conv3_q[0]);
        if (conv3_q[1567] !== 8'h00)        $fatal(1, "C3: conv3_q.mem truncated (conv3_q[1567]=%h)", conv3_q[1567]);

        run_pass(1);            // pass 1: golden + record outputs
        run_pass(2);            // pass 2: reset + rerun, must be identical

        $display("C3_ALL_PASS");
        $finish;
    end
endmodule
