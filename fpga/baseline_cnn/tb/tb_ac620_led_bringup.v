// tb_ac620_led_bringup.v - Simulation of the minimal AC620 board bring-up top.
//
// The DUT uses a shortened configuration (CNT_WIDTH=8, LED taps 4..7) so the
// whole run is two full periods of the slowest LED tap = 2*256 = 512 clock
// edges (~5 us at 10 ns) instead of tens of millions of cycles.
//
// Assertions (frozen):
//   1. the free-running counter keeps running (increments by exactly 1 on
//      every clock edge, wrapping at 2^CNT_WIDTH);
//   2. all four LEDs toggle at least once;
//   3. adjacent LEDs obey the halving relationship: over a whole-period window
//      the number of transitions of led[i] is exactly twice led[i+1];
//   4. no X/Z on the LEDs or the counter.
//
// Any violation -> $fatal (non-zero vsim exit).  On success prints
// "AC620_LED_BRINGUP_PASS ALL_PASS" (the run_questa.ps1 gate greps "ALL_PASS").
`timescale 1ns/1ps

module tb_ac620_led_bringup;
    // ---- shortened configuration (must not be the 50 MHz hardware default) ----
    localparam CNT_WIDTH = 8;
    localparam LED0_TAP  = 4;
    localparam LED1_TAP  = 5;
    localparam LED2_TAP  = 6;
    localparam LED3_TAP  = 7;

    localparam CYCLES = 2 * (1 << CNT_WIDTH);   // two full periods of LED3

    // ---- clock: 10 ns period (equivalent to the 50 MHz board clock) ----
    reg clk_50m = 1'b0;
    always #5 clk_50m = ~clk_50m;

    // ---- DUT ----
    wire [3:0] led;

    ac620_led_bringup_top #(
        .CNT_WIDTH (CNT_WIDTH),
        .LED0_TAP  (LED0_TAP),
        .LED1_TAP  (LED1_TAP),
        .LED2_TAP  (LED2_TAP),
        .LED3_TAP  (LED3_TAP)
    ) u_dut (
        .clk_50m (clk_50m),
        .led     (led)
    );

    // ---- statistics ----
    integer i, j;
    reg  [3:0] prev_led;              // led sampled at the previous edge
    reg  [CNT_WIDTH-1:0] cnt_prev;    // counter sampled at the previous edge
    integer toggles [0:3];            // led[j] transition count in the window
    integer xz_bad;                   // X/Z sightings on led / counter
    integer cnt_bad;                  // counter-increment violations
    integer ratio_bad;                // halving-relationship violations

    // ---- main ----
    initial begin
        for (j = 0; j < 4; j = j + 1) toggles[j] = 0;
        xz_bad = 0; cnt_bad = 0; ratio_bad = 0;

        // first edge: sample the post-edge state (cnt == 1) as the baseline.
        // The #1 settle avoids a race with the DUT's non-blocking counter
        // update (both the TB and the DUT trigger on the same posedge).
        @(posedge clk_50m);
        #1;
        prev_led = led;
        cnt_prev = u_dut.cnt;

        for (i = 1; i <= CYCLES; i = i + 1) begin
            @(posedge clk_50m);
            #1;                                // settle after the edge

            // ---- 4. no X/Z on the observable outputs / counter ----
            if (^led === 1'bx) begin
                if (xz_bad < 20)
                    $display("LED XZ cyc=%0d led=%b", i, led);
                xz_bad = xz_bad + 1;
            end
            if (^u_dut.cnt === 1'bx) begin
                if (xz_bad < 20)
                    $display("LED XZ cyc=%0d cnt=%b", i, u_dut.cnt);
                xz_bad = xz_bad + 1;
            end

            // ---- toggle counting ----
            for (j = 0; j < 4; j = j + 1)
                if (led[j] !== prev_led[j]) toggles[j] = toggles[j] + 1;

            // ---- 1. counter keeps running: increments by exactly 1 (wraps) ----
            if (u_dut.cnt !== (cnt_prev + 1'b1)) begin
                if (cnt_bad < 20)
                    $display("LED CNT cyc=%0d cnt=%0d (expect %0d)",
                             i, u_dut.cnt, cnt_prev + 1'b1);
                cnt_bad = cnt_bad + 1;
            end

            prev_led = led;
            cnt_prev = u_dut.cnt;
        end

        // ---- 2. all four LEDs toggle ----
        if (toggles[0] == 0 || toggles[1] == 0 || toggles[2] == 0 || toggles[3] == 0)
            $display("LED NO-TOGGLE toggles=%0d/%0d/%0d/%0d",
                     toggles[0], toggles[1], toggles[2], toggles[3]);

        // ---- 3. adjacent LEDs halve: toggles[i] == 2 * toggles[i+1] ----
        for (j = 0; j < 3; j = j + 1)
            if (toggles[j] != 2 * toggles[j + 1]) begin
                if (ratio_bad < 20)
                    $display("LED RATIO led[%0d]=%0d led[%0d]=%0d (expect x2)",
                             j, toggles[j], j + 1, toggles[j + 1]);
                ratio_bad = ratio_bad + 1;
            end

        $display("LED BRINGUP toggles=%0d/%0d/%0d/%0d xz_bad=%0d cnt_bad=%0d ratio_bad=%0d",
                 toggles[0], toggles[1], toggles[2], toggles[3], xz_bad, cnt_bad, ratio_bad);

        if (xz_bad != 0 || cnt_bad != 0 || ratio_bad != 0 ||
            toggles[0] == 0 || toggles[1] == 0 || toggles[2] == 0 || toggles[3] == 0)
            $fatal(1, "AC620 LED BRING-UP FAILED (toggles=%0d/%0d/%0d/%0d xz=%0d cnt=%0d ratio=%0d)",
                   toggles[0], toggles[1], toggles[2], toggles[3], xz_bad, cnt_bad, ratio_bad);

        $display("AC620_LED_BRINGUP_PASS ALL_PASS");
        $finish;
    end
endmodule
