// stem_conv_serial.v - Serial single-MAC stem convolution engine (v1, frozen).
//
// Implements exactly:
//     input_q 1x28x28 (S8)
//     -> stem Conv2d 1->16, 3x3, stride 1, padding 1
//     -> + INT32 bias (sign-extended)
//     -> saturate INT32  ->  conv1_acc (debug stream)
//     -> requantize_u8(0x7A999012, 38)  ->  stem_q (debug stream + output RAM)
//
// Design decisions are frozen in docs/rtl_microarchitecture.md.  This is the
// first correctness-first version: 1 MAC lane, oc->y->x output order,
// ic->ky->kx convolution order, fixed 9 taps per output.  Out-of-bounds taps
// contribute 0 and never generate an illegal RAM address.
//
// Timing contract for the synchronous memories (sync_ram_u8 / sync_rom_*):
// every read has ONE cycle of latency - the address is presented, the memory
// samples it at the posedge, and the data is valid on the *next* cycle.  The
// FSM never assumes data is valid in the same cycle the address is issued.
//
// Verilog-2001 gotchas handled: the `*` operator result width equals the wider
// operand width (every product/address multiply is widened explicitly), and
// every signal connected to a module port is declared before the instance
// (named port connections create implicit nets otherwise).
// No vendor IP.
`timescale 1ns/1ps

module stem_conv_serial #(
    // Relative init paths, resolved from the Questa working directory (repo
    // root).  The Quartus smoke wrapper overrides them with paths that resolve
    // from the project directory.  No absolute paths.
    parameter WEIGHT_MEM_FILE = "fpga/baseline_cnn/params/weights/stem_weight.mem",
    parameter BIAS_MEM_FILE   = "fpga/baseline_cnn/params/biases/stem_bias.mem"
) (
    // ---- clock / reset ----
    input  wire              clk,
    input  wire              rst_n,

    // ---- input RAM write port (PC/UART path in the future) ----
    input  wire              input_we,
    input  wire [9:0]        input_waddr,
    input  wire signed [7:0] input_wdata,

    // ---- control ----
    input  wire              start,   // single-cycle pulse; ignored while busy
    output reg               busy,    // 1 while running (PROLOGUE..REQ)
    output reg               done,    // single-cycle pulse at end of run

    // ---- conv1_acc debug stream (one pulse per output element) ----
    output reg               acc_valid,
    output reg  [13:0]       acc_addr,    // CHW, strictly 0..12543
    output reg  signed [31:0] acc_value,   // INT32 accumulator (post-bias)

    // ---- stem_q debug stream (one pulse per output element) ----
    output reg               q_valid,
    output reg  [13:0]       q_addr,      // CHW, strictly 0..12543
    output reg         [7:0] q_value,     // UINT8 requant result

    // ---- output feature-map RAM read port (readback after done) ----
    input  wire [13:0]       output_raddr,
    output wire        [7:0] output_rdata
);
    // ================= frozen geometry =================
    localparam IN_DEPTH    = 784;       // 28*28
    localparam IN_ADDR_W   = 10;
    localparam WT_DEPTH    = 144;       // 16*1*3*3 (OIHW)
    localparam WT_ADDR_W   = 8;
    localparam BIAS_DEPTH  = 16;
    localparam BIAS_ADDR_W = 4;
    localparam OUT_DEPTH   = 12544;     // 16*28*28 (CHW)
    localparam OUT_ADDR_W  = 14;

    // Frozen requant constants (params/baseline_cnn_params.vh).
    localparam STEM_MULT  = 32'sd2056884242;  // 32'h7A999012
    localparam STEM_SHIFT = 6'd38;

    // INT32 saturation bounds (clamp, no wraparound).
    localparam signed [63:0] INT32_MAX =  64'sd2147483647;
    localparam signed [63:0] INT32_MIN = -64'sd2147483648;

    // ================= FSM states =================
    localparam S_IDLE     = 3'd0;
    localparam S_PROLOGUE = 3'd1;
    localparam S_ACC      = 3'd2;
    localparam S_ADD_BIAS = 3'd3;
    localparam S_REQ      = 3'd4;
    localparam S_DONE     = 3'd5;

    reg [2:0] state;
    // current output element (oc -> y -> x)
    reg [3:0] oc;
    reg [4:0] y;
    reg [4:0] x;
    // tap being accumulated in S_ACC (0..8); tap_valid_r = validity of the
    // data currently in in_rdata/wt_rdata (registered one cycle after issue).
    reg [3:0] tap;
    reg       tap_valid_r;
    reg signed [63:0] acc64;           // exact accumulator (S64)

    // ================= combinational addressing =================
    // Tap index being ISSUED this cycle: tap0 in PROLOGUE, tap+1 in ACC
    // (unless tap==8, the last tap - nothing more to issue).  Else 0.
    wire [3:0] issue_tap = (state == S_PROLOGUE) ? 4'd0 :
                           ((state == S_ACC) && (tap != 4'd8)) ? (tap + 4'd1) : 4'd0;

    // Full OIHW tap position: ky = issue_tap/3, kx = issue_tap%3  (Cin=1).
    wire [3:0] i_ky = issue_tap / 4'd3;
    wire [3:0] i_kx = issue_tap % 4'd3;

    // input sample coordinate with padding; iy/ix in {-1..28}.
    wire signed [6:0] iy = $signed({2'b00, y}) + $signed({3'b000, i_ky}) - 7'sd1;
    wire signed [6:0] ix = $signed({2'b00, x}) + $signed({3'b000, i_kx}) - 7'sd1;
    wire valid_tap = (iy >= 7'sd0) && (iy <= 7'sd27) && (ix >= 7'sd0) && (ix <= 7'sd27);

    // input RAM address: legal sample address when in-bounds, else 0
    // (out-of-bounds taps contribute 0 and never read an illegal address).
    wire [9:0] in_raddr = valid_tap ? ({5'd0, iy[4:0]} * 5'd28 + {5'd0, ix[4:0]}) : 10'd0;

    // weight ROM address: OIHW formula ((oc*1 + 0)*3 + ky)*3 + kx = oc*9 + tap.
    wire [7:0] wt_raddr = (state == S_PROLOGUE || state == S_ACC) ?
                          ({4'd0, oc} * 4'd9 + issue_tap) : 8'd0;

    wire [3:0] bias_raddr = oc;                  // 16 x S32

    // output address (CHW) and next coordinates (oc -> y -> x).
    wire [8:0]  row_h  = ({5'd0, oc} * 5'd28) + ({4'd0, y});
    wire [13:0] out_addr = ({5'd0, row_h} * 5'd28) + ({9'd0, x});
    wire [4:0]  x_next = (x == 5'd27) ? 5'd0 : (x + 5'd1);
    wire [4:0]  y_next = (x == 5'd27) ? ((y == 5'd27) ? 5'd0 : (y + 5'd1)) : y;
    wire [3:0]  oc_next = (x == 5'd27) ? ((y == 5'd27) ? (oc + 4'd1) : oc) : oc;
    wire last_output = (oc == 4'd15) && (y == 5'd27) && (x == 5'd27);

    // ================= memory output wires =================
    wire [7:0]        in_rdata;        // input RAM read data (S8 bit pattern)
    wire signed [7:0] wt_rdata;        // weight ROM read data (S8)
    wire signed [31:0] bias_rdata;     // bias ROM read data (S32)

    // ================= datapath =================
    // 16-bit sign-extended product (Verilog-2001: operands widened to the
    // result width or the product would truncate to 8 bits).
    wire signed [15:0] tap_product = $signed({{8{in_rdata[7]}}, in_rdata}) *
                                     $signed({{8{wt_rdata[7]}}, wt_rdata});

    // INT32 saturation AFTER bias is folded into acc64 (no wraparound).
    wire signed [31:0] acc32 = (acc64 > INT32_MAX) ? 32'sh7FFFFFFF :
                               (acc64 < INT32_MIN) ? 32'sh80000000 :
                               acc64[31:0];

    // Reuse the frozen, already-verified requantize_u8 (never reimplement).
    wire [7:0] q_w;
    requantize_u8 u_req (
        .acc        (acc32),
        .multiplier (STEM_MULT),
        .shift      (STEM_SHIFT),
        .q          (q_w)
    );

    // output RAM write (sampled at the REQ posedge; coords advance at the same
    // edge via nonblocking assignment, so the write targets the current output).
    wire out_we    = (state == S_REQ);
    wire [13:0] out_waddr = out_addr;
    wire [7:0]  out_wdata = q_w;

    // ================= memory instances =================
    // input RAM: write port from outside, read port from the tap logic.
    sync_ram_u8 #(.DEPTH(IN_DEPTH), .ADDR_W(IN_ADDR_W)) u_input_ram (
        .clk   (clk),
        .we    (input_we),
        .waddr (input_waddr),
        .wdata (input_wdata),
        .raddr (in_raddr),
        .rdata (in_rdata)
    );

    // weight ROM: read-only, one-cycle latency.
    sync_rom_s8 #(.DEPTH(WT_DEPTH), .ADDR_W(WT_ADDR_W), .FILE(WEIGHT_MEM_FILE))
        u_weight_rom (.clk(clk), .addr(wt_raddr), .rdata(wt_rdata));

    // bias ROM: read-only, one-cycle latency, addressed by oc.
    sync_rom_s32 #(.DEPTH(BIAS_DEPTH), .ADDR_W(BIAS_ADDR_W), .FILE(BIAS_MEM_FILE))
        u_bias_rom (.clk(clk), .addr(bias_raddr), .rdata(bias_rdata));

    // output RAM: write port from the engine, read port from outside.
    sync_ram_u8 #(.DEPTH(OUT_DEPTH), .ADDR_W(OUT_ADDR_W)) u_output_ram (
        .clk   (clk),
        .we    (out_we),
        .waddr (out_waddr),
        .wdata (out_wdata),
        .raddr (output_raddr),
        .rdata (output_rdata)
    );

    // ================= debug streams / control outputs =================
    always @* begin
        busy = (state == S_PROLOGUE) || (state == S_ACC) ||
               (state == S_ADD_BIAS) || (state == S_REQ);
    end
    always @* begin
        acc_valid = (state == S_REQ);
        acc_addr  = out_addr;
        acc_value = acc32;
        q_valid   = (state == S_REQ);
        q_addr    = out_addr;
        q_value   = q_w;
    end

    // ================= FSM =================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            oc     <= 4'd0;
            y      <= 5'd0;
            x      <= 5'd0;
            tap    <= 4'd0;
            tap_valid_r <= 1'b0;
            acc64  <= 64'sd0;
            done   <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        oc  <= 4'd0;
                        y   <= 5'd0;
                        x   <= 5'd0;
                        tap <= 4'd0;
                        acc64 <= 64'sd0;
                        state <= S_PROLOGUE;
                    end
                end
                S_PROLOGUE: begin
                    // clear the accumulator; this cycle's addresses (tap0) are
                    // sampled at this posedge, so the next cycle holds tap0 data.
                    acc64 <= 64'sd0;
                    tap_valid_r <= valid_tap;
                    state <= S_ACC;
                end
                S_ACC: begin
                    // accumulate the tap whose data arrived this cycle, and
                    // issue the next tap's read (software-pipelined).  tap==8
                    // is the last tap: no new issue, drop out of the loop.
                    acc64 <= acc64 + (tap_valid_r ? tap_product : 16'sd0);
                    if (tap == 4'd8) begin
                        state <= S_ADD_BIAS;
                    end else begin
                        tap <= tap + 4'd1;
                        tap_valid_r <= valid_tap;
                    end
                end
                S_ADD_BIAS: begin
                    // bias ROM sampled oc last cycle -> data valid now; add it
                    // once, after all MACs.
                    acc64 <= acc64 + $signed(bias_rdata);
                    state <= S_REQ;
                end
                S_REQ: begin
                    // combinational acc32 = saturate(acc64) and q = requant
                    // are valid this whole cycle; output RAM write is sampled
                    // at this posedge; advance to the next output element.
                    if (last_output) begin
                        state <= S_DONE;
                    end else begin
                        oc  <= oc_next;
                        y   <= y_next;
                        x   <= x_next;
                        tap <= 4'd0;
                        state <= S_PROLOGUE;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;      // single-cycle pulse; busy already low
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
