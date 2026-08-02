// ac620_cnn_selftest_top.v - AC620 fixed-digit BaselineCNN board self-test.
//
// The FIRST real board-level top that runs the complete CNN on the AC620.
// On power-up it loads the frozen digit-8 input_q.mem (784 bytes) into the
// baseline_cnn_core input RAM, starts ONE inference automatically, latches the
// prediction, and drives the four onboard LEDs with a polarity-independent
// PASS / FAIL display.
//
// Physical constraints (frozen 2026-08-02 from the AC620 V2 back-silkscreen):
//     clk_50m -> PIN_E1  (50 MHz, 3.3-V LVTTL)
//     led[0]  -> PIN_A2  (3.3-V LVTTL)
//     led[1]  -> PIN_B3  (3.3-V LVTTL)
//     led[2]  -> PIN_A4  (3.3-V LVTTL)
//     led[3]  -> PIN_A3  (3.3-V LVTTL)
//
// Top-level ports ONLY clk_50m / led: no external rst_n, no keys, no UART, no
// debug physical pins, no virtual pins.  There is no installed independent
// reset button on this board, so a synthesizable internal power-on reset (POR)
// is generated here.
//
// Block structure (single self-contained top):
//     POR                 - internal rst_n held low POR_CYCLES clocks after
//                           configuration, released synchronously (see below)
//     u_input_rom         - sync_rom_s8 (784 x 8) initialized from INPUT_MEM_FILE
//     loader FSM          - POR -> LOAD(784 writes) -> START -> WAIT_DONE -> RESULT
//     u_core              - baseline_cnn_core (every parameter path overridden)
//     prediction latch    - prediction_latched / selftest_pass on done
//     LED display         - prediction half-period + PASS/FAIL animation
//
// Design rules (frozen):
//   - Verilog-2001 only, no PLL, no gated clock, no vendor IP.
//   - The loader drives the core input write port ONLY in LOAD (before start);
//     input_we is never asserted while the core is busy.
//   - start is a single-cycle pulse issued strictly after the last (784th)
//     input write and only while the core is idle (busy == 0).
//   - done is accepted exactly once; prediction is latched on that cycle.
//   - The self-test runs ONCE per power-up (single-shot, no looping).
//   - A watchdog aborts a stuck inference into the FAIL display; the limit is
//     far above the 1,529,163-cycle digit-8 run.
//   - Register power-up initialization is used ONLY for the POR counter /
//     rst_n (declared with an initial value); Quartus synthesises this as a
//     deterministic power-up state.  Every other register is reset by rst_n.
//   - The LED protocol must be readable under both active-high and active-low
//     hardware, so it relies on CHANGE patterns, not static levels:
//       while loading / computing  : all four LEDs heartbeat together
//       result, prediction phase   : led = prediction_latched (4'b1000 for 8)
//       result, animation (PASS)   : 4'b1111 / 4'b0000 together   (sync blink)
//       result, animation (FAIL)   : 4'b1010 / 4'b0101 alternating
//     => PASS reads as "all four LEDs blink together", FAIL as "two groups
//        alternate" under either polarity.  The FINAL pass criterion is the
//        synchronous all-four blink, not any static polarity.
//
// Relative $readmemh paths: Quartus resolves them against the project dir
// (fpga/baseline_cnn/quartus), so the defaults below are ".." and "../.."
// relative.  The testbench overrides INPUT_MEM_FILE and every core file
// parameter with paths that resolve from the Questa working directory (the
// repo root).  No absolute paths are used anywhere.
`timescale 1ns/1ps

module ac620_cnn_selftest_top #(
    // ---- parameter ROM / input ROM paths (Quartus project-dir defaults) ----
    parameter INPUT_MEM_FILE     = "../../sim/vectors/golden_trace/input_q.mem",
    parameter STEM_WT_MEM_FILE   = "../params/weights/stem_weight.mem",
    parameter STEM_BIAS_MEM_FILE = "../params/biases/stem_bias.mem",
    parameter CONV2_WT_MEM_FILE  = "../params/weights/conv2_weight.mem",
    parameter CONV2_BIAS_MEM_FILE= "../params/biases/conv2_bias.mem",
    parameter CONV3_WT_MEM_FILE  = "../params/weights/conv3_weight.mem",
    parameter CONV3_BIAS_MEM_FILE= "../params/biases/conv3_bias.mem",
    parameter FC_WT_MEM_FILE     = "../params/weights/fc_weight.mem",
    parameter FC_BIAS_MEM_FILE   = "../params/biases/fc_bias.mem",

    // ---- self-test configuration (hardware defaults suit 50 MHz) ----
    parameter EXPECTED_PRED      = 4'd8,     // the fixed digit this self-test expects
    parameter POR_CYCLES         = 8'd32,    // internal POR length in clocks
    parameter POR_W              = 8,        // POR counter width (>= log2(POR_CYCLES+1))
    parameter DISP_DIV           = 25'd12_500_000, // 0.25 s per display half-period
    parameter DISP_W             = 25,       // display divider width
    parameter WATCHDOG_LIMIT     = 32'd5_000_000, // 100 ms @50 MHz (>> 1.53 M run)
    parameter WATCHDOG_W         = 32        // watchdog counter width
) (
    input  wire       clk_50m,
    output wire [3:0] led
);

    // ================= internal power-on reset =================
    // `rst_n` is declared low and released synchronously by a free-running POR
    // counter (declaration-time initial value => deterministic power-up state,
    // which is how Quartus implements register power-up initialization).  No
    // PLL, no gated clock, no reliance on uninitialised registers.
    reg [POR_W-1:0] por_cnt = {POR_W{1'b0}};
    reg             rst_n   = 1'b0;
    always @(posedge clk_50m) begin
        if (por_cnt != POR_CYCLES) begin
            por_cnt <= por_cnt + 1'b1;
            rst_n   <= 1'b0;
        end else begin
            rst_n   <= 1'b1;
        end
    end

    // ================= loader / self-test controller =================
    localparam S_PRIME  = 3'd0;   // present ROM addr 0, let rdata settle to mem[0]
    localparam S_LOAD   = 3'd1;   // write all 784 input bytes (0..783, once each)
    localparam S_START  = 3'd2;   // single-cycle core start (only if idle)
    localparam S_WAIT   = 3'd3;   // wait for core done (watchdog-guarded)
    localparam S_RESULT = 3'd4;   // latch prediction / pass-fail, drive LEDs

    reg [2:0]  state;
    reg [9:0]  ld_cnt;                 // LOAD index: item being written (0..783)
    reg [9:0]  rom_addr_r;             // ROM address presented (sampled by ROM)
    reg [WATCHDOG_W-1:0] wd_cnt;       // WAIT cycle counter
    reg [3:0]  prediction_latched;     // latched at done
    reg        selftest_pass;          // (prediction_latched == EXPECTED_PRED)
    reg        watchdog_fired;         // stuck-inference abort flag

    // ================= fixed digit-8 input ROM =================
    // sync_rom_s8: synchronous read, rdata valid one cycle after addr is
    // sampled.  Contents = golden_trace/input_q.mem (784 bytes, verbatim 8-bit
    // patterns, NOT re-quantised).  The signed [7:0] interpretation is applied
    // at the core input port; the stored bit pattern is unchanged.  The read
    // address comes straight from the loader's rom_addr_r register (the ROM has
    // one-cycle read latency, so rom_addr_r runs one ahead of the write index).
    wire [7:0]  rom_rdata;
    sync_rom_s8 #(.DEPTH(784), .ADDR_W(10), .FILE(INPUT_MEM_FILE)) u_input_rom (
        .clk   (clk_50m),
        .addr  (rom_addr_r),
        .rdata (rom_rdata)
    );

    // ---- core status wires (declared before the loader FSM that uses them) ----
    wire core_busy, done_w;
    wire [3:0] prediction_w;

    // ---- loader combinational write/start ports (sampled by the core) ----
    // input_we is high for exactly the 784 LOAD cycles.  waddr is ld_cnt
    // (0..783, strictly once each).  wdata is rom_rdata.
    // ROM timing: rom_addr_r runs AHEAD of the write index by one (set to 1 in
    // S_PRIME, then rom_addr_r <= rom_addr_r+1 each LOAD cycle).  The ROM is a
    // one-cycle-latency sync read, so on LOAD cycle T (writing waddr=T-1) rdata
    // holds mem[T-1]: the ROM sampled addr = rom_addr_r = T-1 on the previous
    // posedge.  This keeps address/data aligned with no off-by-one.
    wire input_we_w      = (state == S_LOAD);
    wire [9:0] input_waddr_w = ld_cnt;
    wire signed [7:0] input_wdata_w = rom_rdata;   // 8-bit pattern, signed at core
    wire start_w         = (state == S_START) && !core_busy;   // only while idle

    always @(posedge clk_50m or negedge rst_n) begin
        if (!rst_n) begin
            state             <= S_PRIME;
            ld_cnt            <= 10'd0;
            rom_addr_r        <= 10'd0;
            wd_cnt            <= {WATCHDOG_W{1'b0}};
            prediction_latched<= 4'd0;
            selftest_pass     <= 1'b0;
            watchdog_fired    <= 1'b0;
        end else begin
            case (state)
                S_PRIME: begin
                    // Set rom_addr_r = 1 so it runs one ahead of the write index
                    // (ld_cnt=0).  The POR window held rom_addr_r at 0 for
                    // POR_CYCLES clocks, so rdata == mem[0] is already stable and
                    // the first LOAD write uses it; rdata becomes mem[1] for the
                    // second write because the ROM samples addr=1 on the first
                    // LOAD posedge.
                    rom_addr_r <= 10'd1;
                    state      <= S_LOAD;
                end
                S_LOAD: begin
                    // Advance the ROM read pointer one per LOAD cycle; the write
                    // of the item whose data is in rdata happens this cycle.
                    if (ld_cnt == 10'd783) begin
                        // 784th and last write done here.  Clamp the ROM address
                        // to the legal maximum (0..783) so rdata never reads
                        // past the ROM depth (mem[784] would be X) afterwards.
                        rom_addr_r <= 10'd783;
                        state      <= S_START;
                    end else begin
                        rom_addr_r <= rom_addr_r + 1'b1;
                        ld_cnt     <= ld_cnt + 1'b1;
                    end
                end
                S_START: begin
                    state <= S_WAIT;             // start_w was high this cycle
                end
                S_WAIT: begin
                    if (done_w) begin
                        prediction_latched <= prediction_w; // valid at done
                        selftest_pass      <= (prediction_w == EXPECTED_PRED);
                        state              <= S_RESULT;
                    end else if (wd_cnt >= WATCHDOG_LIMIT) begin
                        watchdog_fired <= 1'b1;   // stuck inference -> FAIL
                        selftest_pass  <= 1'b0;
                        state          <= S_RESULT;
                    end else begin
                        wd_cnt <= wd_cnt + 1'b1;
                    end
                end
                S_RESULT: begin
                    // single-shot: hold the result display forever
                end
                default: state <= S_PRIME;
            endcase
        end
    end

    // ================= complete BaselineCNN core =================
    // Every parameter file path is overridden for the Quartus project dir
    // (mirrors baseline_cnn_core_smoke_top); the testbench overrides them again
    // for the Questa working dir.  All unused debug outputs are left unconnected
    // (no physical debug pins).  The frozen core internals are NOT modified.
    baseline_cnn_core #(
        .STEM_WT_MEM_FILE   (STEM_WT_MEM_FILE),
        .STEM_BIAS_MEM_FILE (STEM_BIAS_MEM_FILE),
        .CONV2_WT_MEM_FILE  (CONV2_WT_MEM_FILE),
        .CONV2_BIAS_MEM_FILE(CONV2_BIAS_MEM_FILE),
        .CONV3_WT_MEM_FILE  (CONV3_WT_MEM_FILE),
        .CONV3_BIAS_MEM_FILE(CONV3_BIAS_MEM_FILE),
        .FC_WT_MEM_FILE     (FC_WT_MEM_FILE),
        .FC_BIAS_MEM_FILE   (FC_BIAS_MEM_FILE)
    ) u_core (
        .clk            (clk_50m),
        .rst_n          (rst_n),
        .input_we       (input_we_w),
        .input_waddr    (input_waddr_w),
        .input_wdata    (input_wdata_w),
        .start          (start_w),
        .busy           (core_busy),
        .done           (done_w),
        .prediction     (prediction_w),
        .dbg_stage      (),
        .stem_acc_valid (),
        .stem_acc_addr  (),
        .stem_acc_value (),
        .stem_q_valid   (),
        .stem_q_addr    (),
        .stem_q_value   (),
        .pool1_valid    (),
        .pool1_addr     (),
        .pool1_value    (),
        .conv_layer     (),
        .conv_acc_valid (),
        .conv_acc_addr  (),
        .conv_acc_value (),
        .conv_q_valid   (),
        .conv_q_addr    (),
        .conv_q_value   (),
        .pool2_valid    (),
        .pool2_addr     (),
        .pool2_value    (),
        .gap_valid      (),
        .gap_channel    (),
        .gap_value      (),
        .fc_valid       (),
        .fc_class       (),
        .fc_acc         ()
    );

    // ================= LED display control =================
    // A free-running display divider (from reset) provides the human-visible
    // heartbeat during loading/compute and the repeating two half-periods of the
    // result display:
    //     half-period A : led = prediction_latched
    //     half-period B : PASS -> 4'b1111 / 4'b0000 (all four together),
    //                     FAIL -> 4'b1010 / 4'b0101 (alternating groups)
    // anim_on flips every time a half-period B completes, so successive B
    // half-periods show the complementary pattern (the blink).
    reg [DISP_W-1:0] disp_cnt;
    reg              phase_b;   // 0 = prediction half-period, 1 = animation
    reg              anim_on;   // alternates each animation half-period
    always @(posedge clk_50m or negedge rst_n) begin
        if (!rst_n) begin
            disp_cnt <= {DISP_W{1'b0}};
            phase_b  <= 1'b0;
            anim_on  <= 1'b0;
        end else begin
            if (disp_cnt == (DISP_DIV - 1)) begin
                disp_cnt <= {DISP_W{1'b0}};
                phase_b  <= ~phase_b;
                if (phase_b) anim_on <= ~anim_on;   // entering a new A half-period
            end else begin
                disp_cnt <= disp_cnt + 1'b1;
            end
        end
    end

    reg [3:0] led_r;
    always @(posedge clk_50m or negedge rst_n) begin
        if (!rst_n) begin
            led_r <= 4'd0;
        end else if (state == S_RESULT) begin
            if (phase_b)
                // animation half-period: polarity-independent CHANGE pattern
                led_r <= selftest_pass ? (anim_on ? 4'b1111 : 4'b0000)
                                       : (anim_on ? 4'b1010 : 4'b0101);
            else
                // raw prediction: digit 8 => 4'b1000 (LED3 lit active-high)
                led_r <= prediction_latched;
        end else begin
            // loading / computing: all four LEDs heartbeat together
            led_r <= {4{disp_cnt[DISP_W-1]}};
        end
    end
    assign led = led_r;

endmodule
