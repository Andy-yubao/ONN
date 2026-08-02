// conv_u8_serial.v - Shared serial single-MAC convolution engine for conv2/conv3.
//
// Implements exactly, for both layers:
//     conv2 (layer_sel=0):  pool1_q 16x14x14 (UINT8) -> Conv2d 16->32, 3x3, pad1
//                           -> +INT32 bias -> saturate INT32 -> requantize_u8
//                           (0x416335B9 = 1097020857, shift 38) -> conv2_q 32x14x14
//     conv3 (layer_sel=1):  pool2_q 32x7x7  (UINT8) -> Conv2d 32->32, 3x3, pad1
//                           -> +INT32 bias -> saturate INT32 -> requantize_u8
//                           (0x4D6CC8EC = 1298974956, shift 38) -> conv3_q 32x7x7
//
// This engine ONLY serves conv2/conv3; the verified stem engine
// (stem_conv_serial.v) is NOT replaced.  The input feature map lives in an
// EXTERNAL synchronous RAM: the engine drives `fm_raddr`, the external RAM
// presents `fm_rdata` one cycle later (one-cycle read latency, same contract as
// the sync_ram_u8 / sync_rom_* templates).  The weight/bias ROMs for BOTH
// layers are instantiated internally and selected by the layer_sel latched at
// `start` (a mid-run layer_sel change cannot affect the current inference).
//
// Traversal is fixed oc -> y -> x (outer) -> ic -> ky -> kx (inner), outputs in
// CHW order, addresses:
//     out_addr    = (oc*H + y)*W + x          (H,W = 14/14 or 7/7)
//     in_addr     = (ic*H + iy)*W + ix
//     weight_addr = ((oc*Cin + ic)*3 + ky)*3 + kx
// Padding: out-of-bounds taps contribute 0 and never generate an illegal RAM
// address (the address bus is forced to 0 for those taps).
//
// No division / modulo anywhere: oc/y/x/ic/ky/kx are plain counters (the 3x3
// tap loop advances kx->ky->ic directly, never a tap index re-decomposed with
// /3 %3), and every address constant multiply is written as shifts + adds
// (196=128+64+4, 49=32+16+1, 144=128+16, 288=256+32, 14=16-2, 7=8-1, 9=8+1,
// 3=2+1).  No general divider/modulo is ever inferred.
//
// Numeric semantics per output (frozen, Int8Reference scheme A):
//     acc64   = sum over Cin*9 taps of UINT8(input) * SINT8(weight)
//     acc64  += sign_extend(bias)
//     acc32   = saturate_int32(acc64)          (clamp, never wraparound)
//     q       = requantize_u8(acc32, layer_mult, 38)
// The 8x8 tap product is widened to 17 signed bits; the exact accumulator is
// signed 64-bit so the INT32 saturation happens AFTER bias and never wraps.
//
// Busy/done protocol (frozen): `busy` covers PROLOGUE..DONE so it stays 1 from
// start until the done pulse; `done` is a single-cycle pulse.  A start while
// busy is ignored; the next start after done is accepted reliably.
//
// Cycle budget: PROLOGUE(1) + Cin*9 MAC + ADD_BIAS(1) + REQUANT(1) per output
//   conv2: 1+144+1+1 = 147 cyc/output -> 6272*147 + 1 (DONE state) = 921985 cyc
//   conv3: 1+288+1+1 = 291 cyc/output -> 1568*291 + 1 (DONE state) = 456289 cyc
// (the +1 is the single S_DONE cycle before the done pulse; the done pulse
// itself is the cycle after busy drops, so busy_cycles == done_cyc - start_cyc).
//
// Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module conv_u8_serial #(
    // Relative init paths, resolved from the Questa working directory (repo
    // root).  The Quartus smoke wrapper overrides them with paths that resolve
    // from the project directory.  No absolute paths.
    parameter WT2_MEM_FILE   = "fpga/baseline_cnn/params/weights/conv2_weight.mem",
    parameter BIAS2_MEM_FILE = "fpga/baseline_cnn/params/biases/conv2_bias.mem",
    parameter WT3_MEM_FILE   = "fpga/baseline_cnn/params/weights/conv3_weight.mem",
    parameter BIAS3_MEM_FILE = "fpga/baseline_cnn/params/biases/conv3_bias.mem"
) (
    // ---- clock / reset ----
    input  wire              clk,
    input  wire              rst_n,

    // ---- control ----
    input  wire              start,     // single-cycle pulse; ignored while busy
    input  wire              layer_sel, // 0=conv2, 1=conv3; latched at start
    output reg               busy,      // 1 from start until done (never drops first)
    output reg               done,      // single-cycle pulse at end of run

    // ---- external input feature-map RAM read port (one-cycle latency) ----
    output reg  [11:0]       fm_raddr,  // CHW 0..3135 (conv2) / 0..1567 (conv3)
    input  wire        [7:0] fm_rdata,  // UINT8, valid one cycle after fm_raddr

    // ---- conv2/conv3 acc debug stream (one pulse per output element) ----
    output reg               acc_valid,
    output reg  [12:0]       acc_addr,    // CHW, strictly 0..6271 / 0..1567
    output reg signed [31:0] acc_value,   // INT32 accumulator (post-bias)

    // ---- conv2/conv3 q debug stream (one pulse per output element) ----
    output reg               q_valid,
    output reg  [12:0]       q_addr,      // CHW, strictly 0..6271 / 0..1567
    output reg         [7:0] q_value      // UINT8 requant result
);
    // ================= layer geometry =================
    // conv2: Cin=16, H=W=14,  Cout=32, weight 32*16*9=4608, out 6272
    // conv3: Cin=32, H=W=7,   Cout=32, weight 32*32*9=9216, out 1568
    localparam C2_CIN  = 16;
    localparam C3_CIN  = 32;
    localparam C2_HW   = 14;   // conv2 rows/cols
    localparam C3_HW   = 7;    // conv3 rows/cols
    localparam C2_WT_D = 4608;
    localparam C3_WT_D = 9216;

    // Frozen requant constants (params/baseline_cnn_params.vh).
    localparam signed [31:0] C2_MULT = 32'sd1097020857;  // 32'h416335B9
    localparam signed [31:0] C3_MULT = 32'sd1298974956;  // 32'h4D6CC8EC
    localparam [5:0]         CONV_SHIFT = 6'd38;

    // INT32 saturation bounds (clamp, no wraparound).
    localparam signed [63:0] INT32_MAX =  64'sd2147483647;
    localparam signed [63:0] INT32_MIN = -64'sd2147483648;

    // ================= FSM states =================
    localparam S_IDLE    = 3'd0;
    localparam S_PROLOGUE = 3'd1;
    localparam S_MAC     = 3'd2;
    localparam S_ADD_BIAS = 3'd3;
    localparam S_REQUANT = 3'd4;
    localparam S_DONE    = 3'd5;

    reg [2:0] state;
    reg       layer;         // layer_sel latched at start (0=conv2, 1=conv3)
    // current output element (oc -> y -> x)
    reg [4:0] oc;
    reg [3:0] y;
    reg [3:0] x;
    // tap counters (ic -> ky -> kx, kx fastest); the tap currently being ISSUED
    reg [5:0] ic;
    reg [1:0] ky;
    reg [1:0] kx;
    reg       tap_valid_r;   // validity of the tap data currently in the pipeline
    reg signed [63:0] acc64; // exact accumulator (S64)

    // ================= layer-dependent bounds =================
    wire [3:0] hw_max  = layer ? 4'd6  : 4'd13;   // y/x max index (IN_H-1)
    wire [5:0] cin_m1  = layer ? 6'd31 : 6'd15;   // max channel index (Cin-1)

    // ================= combinational output addressing =================
    // conv2 out_addr = (oc*14 + y)*14 + x ; conv3 out_addr = (oc*7 + y)*7 + x
    //   oc*14 = oc<<4 - oc<<1 ; oc*7 = oc<<3 - oc ; row*14 = row<<4 - row<<1
    //   row*7  = row<<3 - row
    wire [9:0] oc14   = ({5'd0, oc} << 4) - ({5'd0, oc} << 1);  // oc*14 (<=434)
    wire [9:0] oc7    = ({5'd0, oc} << 3) - {5'd0, oc};         // oc*7  (<=217)
    wire [9:0] c2_row = oc14 + {5'd0, y};                       // oc*14+y (<=447)
    wire [9:0] c3_row = oc7  + {5'd0, y};                       // oc*7+y (<=223)
    wire [13:0] c2_out = ({4'd0, c2_row} << 4) - ({4'd0, c2_row} << 1) + {10'd0, x}; // row*14+x (<=6271)
    wire [13:0] c3_out = ({4'd0, c3_row} << 3) - {4'd0, c3_row}        + {9'd0, x};  // row*7+x (<=1567)
    wire [13:0] out_addr = layer ? c3_out : c2_out;

    // next output coordinates (oc -> y -> x)
    wire [3:0] x_next = (x == hw_max) ? 4'd0 : (x + 4'd1);
    wire [3:0] y_next = (x == hw_max) ? ((y == hw_max) ? 4'd0 : (y + 4'd1)) : y;
    wire [4:0] oc_next = (x == hw_max) ? ((y == hw_max) ? (oc + 5'd1) : oc) : oc;
    wire last_output = (oc == 5'd31) && (y == hw_max) && (x == hw_max);  // Cout=32 both

    // ================= combinational tap iteration =================
    // next tap counters (kx fastest): (kx->2 => ky+1; ky->2 => ic+1)
    wire [1:0] kx_n = (kx == 2'd2) ? 2'd0 : (kx + 2'd1);
    wire [1:0] ky_n = (kx == 2'd2) ? ((ky == 2'd2) ? 2'd0 : (ky + 2'd1)) : ky;
    wire [5:0] ic_n = (kx == 2'd2) ? ((ky == 2'd2) ? (ic + 6'd1) : ic) : ic;
    wire last_tap = (kx == 2'd2) && (ky == 2'd2) && (ic == cin_m1);

    // tap currently being ISSUED: in PROLOGUE tap0 (counters reset); in MAC the
    // NEXT tap (counters advance combinationally).  On the last MAC tap nothing
    // is issued.
    wire [5:0] iss_ic = (state == S_PROLOGUE) ? ic : ic_n;
    wire [1:0] iss_ky = (state == S_PROLOGUE) ? ky : ky_n;
    wire [1:0] iss_kx = (state == S_PROLOGUE) ? kx : kx_n;
    wire issuing = (state == S_PROLOGUE) || ((state == S_MAC) && !last_tap);

    // ================= input sample coords (with padding) =================
    // iy/ix in {-1..14} for conv2, {-1..8} for conv3; -1 reads as 31 unsigned
    // (two's complement) and is excluded by the hw_max bound test.
    wire signed [4:0] iss_iy = $signed({1'b0, y}) + $signed({2'b00, iss_ky}) - 5'sd1;
    wire signed [4:0] iss_ix = $signed({1'b0, x}) + $signed({2'b00, iss_kx}) - 5'sd1;
    wire [4:0] iss_iyu = iss_iy[4:0];
    wire [4:0] iss_ixu = iss_ix[4:0];
    wire valid_tap = (iss_iyu <= {1'b0, hw_max}) && (iss_ixu <= {1'b0, hw_max});
    wire [3:0] iy_v = iss_iyu[3:0];   // in-bounds row value
    wire [3:0] ix_v = iss_ixu[3:0];   // in-bounds col value

    // ================= input address (CHW, shift-add only) =================
    // conv2: in_addr = ic*196 + iy*14 + ix   (ic*196 = ic<<7 + ic<<6 + ic<<2)
    // conv3: in_addr = ic*49  + iy*7  + ix   (ic*49  = ic<<5 + ic<<4 + ic)
    wire [12:0] fm_c2 = ({7'd0, iss_ic} << 7) + ({7'd0, iss_ic} << 6) + ({7'd0, iss_ic} << 2)
                      + ({6'd0, iy_v} << 4) - ({6'd0, iy_v} << 1) + {8'd0, ix_v};
    wire [12:0] fm_c3 = ({7'd0, iss_ic} << 5) + ({7'd0, iss_ic} << 4) + {7'd0, iss_ic}
                      + ({6'd0, iy_v} << 3) - {6'd0, iy_v} + {8'd0, ix_v};

    // ================= weight address (OIHW, shift-add only) =================
    // conv2: ((oc*16 + ic)*3 + ky)*3 + kx = oc*144 + ic*9 + ky*3 + kx
    //        oc*144 = oc<<7 + oc<<4 ; ic*9 = ic<<3 + ic ; ky*3 = ky<<1 + ky
    // conv3: ((oc*32 + ic)*3 + ky)*3 + kx = oc*288 + ic*9 + ky*3 + kx
    //        oc*288 = oc<<8 + oc<<5
    wire [13:0] wt2_oc = ({6'd0, oc} << 7) + ({6'd0, oc} << 4);  // oc*144 (max 4464)
    wire [13:0] wt3_oc = ({6'd0, oc} << 8) + ({6'd0, oc} << 5);  // oc*288 (max 8928)
    wire [8:0]  ic9    = {3'd0, iss_ic};                          // 0..32
    wire [8:0]  ic9m   = (ic9 << 3) + ic9;                        // iss_ic*9 (max 288)
    wire [2:0]  ky3    = ({1'b0, iss_ky} << 1) + {1'b0, iss_ky};  // iss_ky*3 (max 6)
    wire [12:0] wt2_addr = wt2_oc[12:0] + {4'd0, ic9m} + {10'd0, ky3} + {10'd0, iss_kx};
    wire [13:0] wt3_addr = wt3_oc       + {5'd0, ic9m} + {11'd0, ky3} + {11'd0, iss_kx};

    // ================= memory output wires =================
    wire signed [7:0]  wt2_rdata, wt3_rdata;
    wire signed [31:0] bias2_rdata, bias3_rdata;
    wire signed [7:0]  wt_rdata   = layer ? wt3_rdata : wt2_rdata;
    wire signed [31:0] bias_rdata = layer ? bias3_rdata : bias2_rdata;

    // ================= datapath =================
    // UINT8 input (0..255) x SINT8 weight (-128..127), widened to 17 signed
    // bits so the product never truncates (max |255*127| = 32385 < 2^15).
    wire signed [16:0] tap_product = $signed({9'd0, fm_rdata}) *
                                     $signed({{9{wt_rdata[7]}}, wt_rdata});

    // INT32 saturation AFTER bias is folded into acc64 (no wraparound).
    wire signed [31:0] acc32 = (acc64 > INT32_MAX) ? 32'sh7FFFFFFF :
                               (acc64 < INT32_MIN) ? 32'sh80000000 :
                               acc64[31:0];

    // Reuse the frozen, already-verified requantize_u8 (never reimplement).
    wire [7:0] q_w;
    requantize_u8 u_req (
        .acc        (acc32),
        .multiplier (layer ? C3_MULT : C2_MULT),
        .shift      (CONV_SHIFT),
        .q          (q_w)
    );

    // ================= weight-ROM read gating =================
    // Each weight ROM is only addressed while issuing THAT layer's taps; the
    // OTHER layer's ROM is forced to address 0 so it can never be read
    // out-of-range.  This matters because the two layers run the same counters:
    // during a conv3 run `wt2_addr` would climb to oc*144 + ic*9 + ... = 4751,
    // beyond the conv2 ROM depth of 4608; during a conv2 run `wt3_addr` would
    // climb to 4607 (still < 9216 but forced to 0 per the frozen gating rule).
    // Outside issuing cycles both ROMs are addressed at 0 too.  `layer` is
    // latched at start, so the gate is stable for the whole run.
    wire wt2_read = issuing && !layer;
    wire wt3_read = issuing &&  layer;

    // ================= memory instances =================
    // Weight ROMs for BOTH layers (conv2 4608 S8, conv3 9216 S8); bias ROMs for
    // BOTH layers (32 S32 each).
    sync_rom_s8 #(.DEPTH(C2_WT_D), .ADDR_W(13), .FILE(WT2_MEM_FILE))
        u_wt2_rom (.clk(clk), .addr(wt2_read ? wt2_addr : 13'd0), .rdata(wt2_rdata));
    sync_rom_s8 #(.DEPTH(C3_WT_D), .ADDR_W(14), .FILE(WT3_MEM_FILE))
        u_wt3_rom (.clk(clk), .addr(wt3_read ? wt3_addr : 14'd0), .rdata(wt3_rdata));
    sync_rom_s32 #(.DEPTH(32), .ADDR_W(5), .FILE(BIAS2_MEM_FILE))
        u_bias2_rom (.clk(clk), .addr(oc[4:0]), .rdata(bias2_rdata));
    sync_rom_s32 #(.DEPTH(32), .ADDR_W(5), .FILE(BIAS3_MEM_FILE))
        u_bias3_rom (.clk(clk), .addr(oc[4:0]), .rdata(bias3_rdata));

    // ================= control outputs (all combinational) =================
    always @* begin
        // busy covers S_DONE too: once the run has started it stays 1 until the
        // done pulse, never a `busy=0 && done=0` window between start and done.
        busy = (state != S_IDLE);
    end
    always @* begin
        acc_valid = (state == S_REQUANT);
        acc_addr  = out_addr[12:0];
        acc_value = acc32;
        q_valid   = (state == S_REQUANT);
        q_addr    = out_addr[12:0];
        q_value   = q_w;
    end
    always @* begin
        if (issuing)
            fm_raddr = valid_tap ? (layer ? fm_c3[11:0] : fm_c2[11:0]) : 12'd0;
        else
            fm_raddr = 12'd0;
    end

    // ================= FSM =================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            oc          <= 5'd0;
            y           <= 4'd0;
            x           <= 4'd0;
            ic          <= 6'd0;
            ky          <= 2'd0;
            kx          <= 2'd0;
            layer       <= 1'b0;
            tap_valid_r <= 1'b0;
            acc64       <= 64'sd0;
            done        <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        layer <= layer_sel;      // latch the layer for this run
                        oc    <= 5'd0; y <= 4'd0; x <= 4'd0;
                        ic    <= 6'd0; ky <= 2'd0; kx <= 2'd0;
                        acc64 <= 64'sd0;
                        state <= S_PROLOGUE;
                    end
                end
                S_PROLOGUE: begin
                    // clear the accumulator; tap0 addresses are presented this
                    // cycle, so the next cycle holds tap0 data.
                    acc64      <= 64'sd0;
                    tap_valid_r <= valid_tap;    // tap0 validity (iss = counters)
                    state      <= S_MAC;
                end
                S_MAC: begin
                    // accumulate the tap whose data arrived this cycle, and
                    // issue the next tap's read (software-pipelined).  The last
                    // tap drops out of the loop with no new issue.
                    acc64 <= acc64 + (tap_valid_r ? tap_product : 17'sd0);
                    if (last_tap) begin
                        state <= S_ADD_BIAS;
                    end else begin
                        ic <= ic_n; ky <= ky_n; kx <= kx_n;
                        tap_valid_r <= valid_tap;    // next tap validity (iss = next)
                    end
                end
                S_ADD_BIAS: begin
                    // bias ROM sampled oc long ago -> data valid now; add once
                    // after all MACs.
                    acc64 <= acc64 + $signed(bias_rdata);
                    state <= S_REQUANT;
                end
                S_REQUANT: begin
                    // acc32 = saturate(acc64) and q = requant are combinational
                    // and valid this whole cycle; advance to the next output.
                    if (last_output) begin
                        state <= S_DONE;
                    end else begin
                        oc  <= oc_next;
                        y   <= y_next;
                        x   <= x_next;
                        ic  <= 6'd0; ky <= 2'd0; kx <= 2'd0;
                        state <= S_PROLOGUE;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;      // single-cycle pulse; busy stays 1 through DONE
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
