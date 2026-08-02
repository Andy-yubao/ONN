// ac620_led_bringup_top.v - Minimal physical top for AC620 board bring-up.
//
// Real 50 MHz board clock (PIN_E1) drives a free-running binary counter whose
// high bits drive the four onboard LEDs (LED0=PIN_A2, LED1=PIN_B3,
// LED2=PIN_A4, LED3=PIN_A3).  The four LEDs toggle at strictly halved
// frequencies so all four blink at different, human-visible rates (with the
// 25-bit default, LED0..LED3 toggle at ~11.9 / 5.96 / 2.98 / 1.49 Hz).
//
// Constraints (frozen from the AC620 V2 board back-silkscreen, 2026-08-02):
//     clk_50m -> PIN_E1  (50 MHz, 3.3-V LVTTL)
//     led[0]  -> PIN_A2  (3.3-V LVTTL)
//     led[1]  -> PIN_B3  (3.3-V LVTTL)
//     led[2]  -> PIN_A4  (3.3-V LVTTL)
//     led[3]  -> PIN_A3  (3.3-V LVTTL)
//
// Design rules (frozen):
//   - Verilog-2001 only.
//   - NO reset port: this board has no installed independent RST_N button, so
//     we must not invent one.  The counter is declared with an initial value
//     of 0, which Quartus synthesises as a deterministic power-up state.
//   - No PLL / vendor IP / divider / gated clock.
//   - The counter width and LED taps are parameters so the testbench can run a
//     shortened simulation; the hardware defaults must suit 50 MHz.
//   - No assumption about LED active-high vs active-low: as long as the taps
//     keep toggling, either polarity is observable as blinking.
`timescale 1ns/1ps

module ac620_led_bringup_top #(
    parameter CNT_WIDTH = 25,
    parameter LED0_TAP  = 21,   // fastest  (~11.9 Hz at 50 MHz with width 25)
    parameter LED1_TAP  = 22,   // /2
    parameter LED2_TAP  = 23,   // /4
    parameter LED3_TAP  = 24    // slowest  (~1.49 Hz at 50 MHz with width 25)
) (
    input  wire       clk_50m,
    output wire [3:0] led
);

    // Free-running counter.  The declaration-time initial value gives the
    // register a deterministic power-up state (no reset port on this board).
    reg [CNT_WIDTH-1:0] cnt = {CNT_WIDTH{1'b0}};

    always @(posedge clk_50m)
        cnt <= cnt + 1'b1;

    // Four LEDs on consecutive high counter bits -> strictly halved toggle
    // frequencies, LED0 fastest .. LED3 slowest.
    assign led[0] = cnt[LED0_TAP];
    assign led[1] = cnt[LED1_TAP];
    assign led[2] = cnt[LED2_TAP];
    assign led[3] = cnt[LED3_TAP];

endmodule
