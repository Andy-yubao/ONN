// tb_ac620_cnn_selftest.v - End-to-end verification of the AC620 fixed-digit
// BaselineCNN board self-test top.
//
// Instantiates the REAL ac620_cnn_selftest_top -> baseline_cnn_core -> every
// sub-block (no core stub).  Two instances:
//     u_pass  EXPECTED_PRED = 8  (the frozen digit; the run must PASS)
//     u_fail  EXPECTED_PRED = 7  (prediction stays 8, expectation mismatch;
//                                 must FAIL and show the 1010/0101 animation)
// Both use shortened POR / display-divider parameters so the simulation shows
// the result display without simulating real seconds (the CNN compute itself is
// NOT shortened - each instance runs the full 1,590,315-cycle inference).
//
// Assertions (any violation -> $fatal, non-zero vsim exit):
//   1.  loader starts after POR and reaches the LOAD state;
//   2.  input ROM read succeeds (write data matches input_q.mem bit-exactly);
//   3.  exactly 784 input writes;
//   4.  write addresses strictly 0..783 (each address written exactly once);
//   5.  write data bit-equal to input_q.mem;
//   6.  input_we never asserted while the core is busy;
//   7.  start is a single-cycle pulse (count == 1);
//   8.  start issued strictly after the last (784th) input write;
//   9.  core done pulse exactly once;
//   10. start->done cycle count matches the A+ P=3 result (1,590,315);
//   11. prediction == 8 (latched);
//   12. selftest_pass latched 1 on the PASS instance, 0 on the FAIL instance;
//   13. no watchdog timeout on either instance;
//   14. LEDs free of X/Z;
//   15. LED enters the raw-prediction display phase (led == 4'b1000);
//   16. LED enters the PASS sync-blink animation (all four together);
//   17. PASS animation keeps all four LED bits equal;
//   18. FAIL instance shows the 1010 / 0101 alternating animation;
//   19. prints AC620_CNN_SELFTEST_PASS (all failures are $fatal).
//
// Run from the repo root (the $readmemh / core file paths are repo-relative).
`timescale 1ns/1ps

module tb_ac620_cnn_selftest;
    // ================= clock (10 ns period == 50 MHz) =================
    reg clk_50m = 1'b0;
    always #5 clk_50m = ~clk_50m;

    // ================= reference input (golden digit 8) =================
    reg [7:0] ref_input [0:783];

    // ================= PASS instance (EXPECTED_PRED = 8) =================
    wire [3:0] led_pass;
    ac620_cnn_selftest_top #(
        .INPUT_MEM_FILE     ("fpga/baseline_cnn/sim/vectors/golden_trace/input_q.mem"),
        .STEM_WT_MEM_FILE   ("fpga/baseline_cnn/params/weights/stem_weight.mem"),
        .STEM_BIAS_MEM_FILE ("fpga/baseline_cnn/params/biases/stem_bias.mem"),
        .CONV2_WT_MEM_FILE  ("fpga/baseline_cnn/params/weights/conv2_weight.mem"),
        .CONV2_BIAS_MEM_FILE("fpga/baseline_cnn/params/biases/conv2_bias.mem"),
        .CONV3_WT_MEM_FILE  ("fpga/baseline_cnn/params/weights/conv3_weight.mem"),
        .CONV3_BIAS_MEM_FILE("fpga/baseline_cnn/params/biases/conv3_bias.mem"),
        .FC_WT_MEM_FILE     ("fpga/baseline_cnn/params/weights/fc_weight.mem"),
        .FC_BIAS_MEM_FILE   ("fpga/baseline_cnn/params/biases/fc_bias.mem"),
        .EXPECTED_PRED      (4'd8),
        .POR_CYCLES         (8'd8),
        .DISP_DIV           (25'd4),
        .WATCHDOG_LIMIT     (32'd2_000_000)   // > 1,590,315 run
    ) u_pass (
        .clk_50m (clk_50m),
        .led     (led_pass)
    );

    // ================= FAIL instance (EXPECTED_PRED = 7) =================
    wire [3:0] led_fail;
    ac620_cnn_selftest_top #(
        .INPUT_MEM_FILE     ("fpga/baseline_cnn/sim/vectors/golden_trace/input_q.mem"),
        .STEM_WT_MEM_FILE   ("fpga/baseline_cnn/params/weights/stem_weight.mem"),
        .STEM_BIAS_MEM_FILE ("fpga/baseline_cnn/params/biases/stem_bias.mem"),
        .CONV2_WT_MEM_FILE  ("fpga/baseline_cnn/params/weights/conv2_weight.mem"),
        .CONV2_BIAS_MEM_FILE("fpga/baseline_cnn/params/biases/conv2_bias.mem"),
        .CONV3_WT_MEM_FILE  ("fpga/baseline_cnn/params/weights/conv3_weight.mem"),
        .CONV3_BIAS_MEM_FILE("fpga/baseline_cnn/params/biases/conv3_bias.mem"),
        .FC_WT_MEM_FILE     ("fpga/baseline_cnn/params/weights/fc_weight.mem"),
        .FC_BIAS_MEM_FILE   ("fpga/baseline_cnn/params/biases/fc_bias.mem"),
        .EXPECTED_PRED      (4'd7),           // mismatch -> FAIL animation
        .POR_CYCLES         (8'd8),
        .DISP_DIV           (25'd4),
        .WATCHDOG_LIMIT     (32'd2_000_000)
    ) u_fail (
        .clk_50m (clk_50m),
        .led     (led_fail)
    );

    // ================= statistics / flags =================
    integer cyc;                 // global cycle counter
    integer wr_count;            // PASS instance write-enable cycles
    integer waddr_bad;           // write address not strictly 0..783
    integer wdata_bad;           // write data != input_q.mem
    integer we_busy_bad;         // input_we asserted while core busy
    integer start_count;         // core start pulses (expect exactly 1)
    integer done_count;          // core done pulses (expect exactly 1)
    integer wd_fired;            // watchdog fired on either instance
    integer xz_bad;              // X/Z on led_pass / led_fail
    integer anim_sync_bad;       // PASS animation not all-four-equal
    integer pred_phase_bad;      // prediction phase led != 4'b1000
    integer fail_anim_bad;       // FAIL animation not 1010/0101
    integer start_cyc;           // cycle the core sampled start
    integer done_cyc;            // cycle the core pulsed done
    integer last_write_cyc;      // cycle of the last input write
    integer saw_pred_phase;      // saw led == 4'b1000 (prediction phase)
    integer saw_pass_on;         // saw led == 4'b1111 (PASS blink)
    integer saw_pass_off;        // saw led == 4'b0000 (PASS blink)
    integer saw_fail_a;          // saw led == 4'b1010 (FAIL blink)
    integer saw_fail_b;          // saw led == 4'b0101 (FAIL blink)
    integer i, ok;               // main-loop scratch (declared at module level)

    reg busy_prev;               // core busy from the previous posedge (rising edge)
    reg phase_b_prev_pass;       // PASS instance phase_b from the previous posedge
    reg phase_b_prev_fail;       // FAIL instance phase_b from the previous posedge
    reg [2:0] state_prev_pass;   // PASS instance loader state, previous posedge
    reg [2:0] state_prev_fail;   // FAIL instance loader state, previous posedge

    localparam S_RESULT = 3'd4;

    // ================= monitor: sample every clock edge =================
    // Write-port checks run on the NEGEDGE: mid-cycle the combinational write
    // bus (input_we_w / input_waddr_w / input_wdata_w) is the stable value the
    // core samples at the next posedge (the ROM rdata has not yet been updated
    // by that posedge), so it is the actual written data.  Sampling these on the
    // posedge would see the ROM's freshly-updated rdata one cycle early.
    always @(negedge clk_50m) begin
        if (u_pass.input_we_w) begin
            if (wr_count < 784) begin
                if (u_pass.input_waddr_w !== wr_count[9:0])
                    waddr_bad = waddr_bad + 1;
                if (u_pass.input_wdata_w !== ref_input[wr_count])
                    wdata_bad = wdata_bad + 1;
            end
            wr_count = wr_count + 1;
            last_write_cyc = cyc;
        end
        // input_we must be 0 while the core is busy
        if (u_pass.u_core.busy && u_pass.input_we_w)
            we_busy_bad = we_busy_bad + 1;
    end

    always @(posedge clk_50m) begin
        #1;
        cyc = cyc + 1;

        // ---- start: count the loader start pulse; record the core busy rising
        //      edge (the cycle the core samples start) as start_cyc ----
        if (u_pass.start_w) begin
            start_count = start_count + 1;
        end
        if (u_pass.u_core.busy && !busy_prev)
            start_cyc = cyc;                 // core just left IDLE on this edge
        busy_prev = u_pass.u_core.busy;

        // ---- done ----
        if (u_pass.done_w) begin
            done_count = done_count + 1;
            done_cyc   = cyc;
        end

        // ---- watchdog ----
        if (u_pass.watchdog_fired || u_fail.watchdog_fired) wd_fired = wd_fired + 1;

        // ---- X/Z on the LEDs ----
        if (^led_pass === 1'bx || ^led_fail === 1'bx) begin
            if (xz_bad < 20)
                $display("SELFTEST XZ cyc=%0d led_pass=%b led_fail=%b", cyc, led_pass, led_fail);
            xz_bad = xz_bad + 1;
        end

        // ---- PASS instance result display ----
        // led_r is computed from the PREVIOUS phase_b (non-blocking), so the
        // cycle where phase_b changes still shows the old phase.  Skip those
        // transition cycles and assert only on stable half-periods.
        if (u_pass.state == S_RESULT) begin
            if (u_pass.state !== state_prev_pass) begin
                // just entered the result display; led_r still shows the old
                // phase - skip strict checks this cycle
            end else if (u_pass.phase_b !== phase_b_prev_pass) begin
                // phase transition cycle: led_r still shows the old phase
            end else if (u_pass.phase_b) begin
                // stable animation half-period: all four LEDs must be equal
                if (!(led_pass[0] === led_pass[1] &&
                      led_pass[1] === led_pass[2] &&
                      led_pass[2] === led_pass[3]))
                    anim_sync_bad = anim_sync_bad + 1;
                if (led_pass === 4'b1111) saw_pass_on  = 1;
                if (led_pass === 4'b0000) saw_pass_off = 1;
            end else begin
                // stable prediction half-period: digit 8 raw output is 4'b1000
                if (led_pass === 4'b1000) saw_pred_phase = 1;
                else                      pred_phase_bad = pred_phase_bad + 1;
            end
            phase_b_prev_pass = u_pass.phase_b;
        end
        state_prev_pass = u_pass.state;

        // ---- FAIL instance result display ----
        if (u_fail.state == S_RESULT) begin
            if (u_fail.state !== state_prev_fail) begin
                // just entered the result display; skip
            end else if (u_fail.phase_b !== phase_b_prev_fail) begin
                // phase transition cycle; skip strict checks
            end else if (u_fail.phase_b) begin
                if (led_fail === 4'b1010) saw_fail_a = 1;
                if (led_fail === 4'b0101) saw_fail_b = 1;
                if (led_fail !== 4'b1010 && led_fail !== 4'b0101)
                    fail_anim_bad = fail_anim_bad + 1;
            end
            phase_b_prev_fail = u_fail.phase_b;
        end
        state_prev_fail = u_fail.state;
    end

    // ================= whole-simulation timeout =================
    initial begin : sim_watchdog
        // PASS+FAIL runs complete at ~1.59M cycles; display verification is a
        // few hundred more.  3M with a $fatal aborts a stuck run.
        repeat (3000000) @(posedge clk_50m);
        $display("SELFTEST TIMEOUT DEBUG: pass.state=%0d fail.state=%0d pass.busy=%b pass.done=%b pass.wd=%0d fail.wd=%0d pass.pred=%0d writes=%0d waddr_bad=%0d wdata_bad=%0d start=%0d done=%0d start->done=%0d",
                 u_pass.state, u_fail.state, u_pass.u_core.busy, u_pass.done_w,
                 u_pass.watchdog_fired, u_fail.watchdog_fired, u_pass.prediction_latched,
                 wr_count, waddr_bad, wdata_bad, start_count, done_count, done_cyc - start_cyc);
        $fatal(1, "SELFTEST: TIMEOUT - an instance never finished");
    end

    // ================= main =================
    initial begin
        // ---- load the golden input reference ----
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/input_q.mem", ref_input);
        if (ref_input[0] !== 8'hED)
            $fatal(1, "SELFTEST: input_q.mem not loaded (ref_input[0]=%h)", ref_input[0]);

        cyc = 0; wr_count = 0; waddr_bad = 0; wdata_bad = 0; we_busy_bad = 0;
        start_count = 0; done_count = 0; wd_fired = 0; xz_bad = 0;
        anim_sync_bad = 0; pred_phase_bad = 0; fail_anim_bad = 0;
        start_cyc = 0; done_cyc = 0; last_write_cyc = 0;
        saw_pred_phase = 0; saw_pass_on = 0; saw_pass_off = 0;
        saw_fail_a = 0; saw_fail_b = 0;
        busy_prev = 0; phase_b_prev_pass = 0; phase_b_prev_fail = 0;
        state_prev_pass = 0; state_prev_fail = 0;

        // ---- wait for both runs to finish AND the display to be seen ----
        wait (u_pass.state == S_RESULT && u_fail.state == S_RESULT);
        // from here the two DUTs are in their result display; make sure we have
        // seen the prediction phase and both blink patterns on each instance.
        wait (saw_pred_phase && saw_pass_on && saw_pass_off &&
              saw_fail_a && saw_fail_b);
        repeat (16) @(posedge clk_50m);

        // ---- verdict ----
        ok = 1;
        if (wr_count != 784) begin
            $display("SELFTEST FAIL: writes=%0d expect 784", wr_count); ok = 0;
        end
        if (waddr_bad != 0) begin
            $display("SELFTEST FAIL: write-address violations=%0d", waddr_bad); ok = 0;
        end
        if (wdata_bad != 0) begin
            $display("SELFTEST FAIL: write-data mismatches=%0d (ROM read broken?)", wdata_bad); ok = 0;
        end
        if (we_busy_bad != 0) begin
            $display("SELFTEST FAIL: input_we asserted while busy=%0d", we_busy_bad); ok = 0;
        end
        if (start_count != 1) begin
            $display("SELFTEST FAIL: start pulses=%0d expect 1", start_count); ok = 0;
        end
        if (done_count != 1) begin
            $display("SELFTEST FAIL: done pulses=%0d expect 1", done_count); ok = 0;
        end
        if (start_cyc <= last_write_cyc) begin
            $display("SELFTEST FAIL: start at cyc %0d not after last write cyc %0d",
                     start_cyc, last_write_cyc); ok = 0;
        end
        if (done_cyc - start_cyc != 1590315) begin
            $display("SELFTEST FAIL: start->done=%0d expect 1590315", done_cyc - start_cyc); ok = 0;
        end
        if (u_pass.prediction_latched !== 4'd8) begin
            $display("SELFTEST FAIL: prediction_latched=%0d expect 8", u_pass.prediction_latched); ok = 0;
        end
        if (u_pass.selftest_pass !== 1'b1) begin
            $display("SELFTEST FAIL: selftest_pass=%0d expect 1", u_pass.selftest_pass); ok = 0;
        end
        if (u_fail.prediction_latched !== 4'd8 || u_fail.selftest_pass !== 1'b0) begin
            $display("SELFTEST FAIL: fail-instance pred=%0d pass=%0d (expect 8/0)",
                     u_fail.prediction_latched, u_fail.selftest_pass); ok = 0;
        end
        if (wd_fired != 0) begin
            $display("SELFTEST FAIL: watchdog fired=%0d", wd_fired); ok = 0;
        end
        if (xz_bad != 0) begin
            $display("SELFTEST FAIL: X/Z on LEDs=%0d", xz_bad); ok = 0;
        end
        if (saw_pred_phase != 1 || pred_phase_bad != 0) begin
            $display("SELFTEST FAIL: prediction phase seen=%0d bad=%0d (led != 4'b1000)",
                     saw_pred_phase, pred_phase_bad); ok = 0;
        end
        if (saw_pass_on != 1 || saw_pass_off != 1) begin
            $display("SELFTEST FAIL: PASS blink seen 1111=%0d 0000=%0d",
                     saw_pass_on, saw_pass_off); ok = 0;
        end
        if (anim_sync_bad != 0) begin
            $display("SELFTEST FAIL: PASS animation not sync=%0d", anim_sync_bad); ok = 0;
        end
        if (saw_fail_a != 1 || saw_fail_b != 1 || fail_anim_bad != 0) begin
            $display("SELFTEST FAIL: FAIL anim seen 1010=%0d 0101=%0d bad=%0d",
                     saw_fail_a, saw_fail_b, fail_anim_bad); ok = 0;
        end

        $display("AC620_CNN_SELFTEST stat writes=%0d start=%0d done=%0d start->done=%0d pred=%0d pass=%0d",
                 wr_count, start_cyc, done_cyc, done_cyc - start_cyc,
                 u_pass.prediction_latched, u_pass.selftest_pass);
        $display("AC620_CNN_SELFTEST stat led PASS[1111=%0d 0000=%0d sync_bad=%0d pred=%0d/%0d] FAIL[1010=%0d 0101=%0d bad=%0d] xz=%0d wd=%0d",
                 saw_pass_on, saw_pass_off, anim_sync_bad, saw_pred_phase, pred_phase_bad,
                 saw_fail_a, saw_fail_b, fail_anim_bad, xz_bad, wd_fired);

        if (!ok)
            $fatal(1, "AC620_CNN_SELFTEST FAILED");

        $display("AC620_CNN_SELFTEST_PASS ALL_PASS");
        $finish;
    end
endmodule
