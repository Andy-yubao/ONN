// fc_argmax_serial.v - Serial fully-connected layer + signed argmax.
//
// Implements exactly (Int8Reference scheme A, docs/data_format.md §5.5):
//     gap_q    32 x UINT8   (GAP output, written through gap_we port)
//     fc_weight 10 x 32     OI, SINT8  (class o weights at addr o*32 + f)
//     fc_bias   10 x SINT32
//     fc_acc[o]   = sum_f gap_q[f] * fc_weight[o][f] + fc_bias[o]
//     prediction  = argmax over the 10 INT32 fc_acc, SIGNED comparison,
//                   ties resolved to the SMALLER class index (PyTorch
//                   first-max semantics):
//                       class 0 seeds `best` unconditionally;
//                       later classes update only when fc_acc > best;
//                       NEVER >=  (a tie keeps the earlier class).
//
// Numeric semantics:
//   - gap_q is UINT8 (0..255), weights SINT8 (-128..127): the tap product is
//     widened to 17 signed bits (max |255*127| = 32385 < 2^15).
//   - the exact accumulator is signed 64-bit (32 taps x <=255x127 => |sum|
//     <= 32*32385 ~= 1.0M, far below INT64); bias is added ONCE after all 32
//     MACs; INT32 saturation happens AFTER bias (clamp, never wraparound).
//   - NO requantisation/dequantisation on the FC path: prediction is taken
//     directly on the INT32 accumulators.
//
// Timing / protocol (frozen, mirrors the conv engine):
//   - gap_we/gap_waddr/gap_wdata write gap_mem (32 x UINT8 registers) at any
//     time; the controller writes all 32 values before asserting start.
//   - ONE `start` runs classes 0..9; `busy` stays 1 from start until the done
//     pulse (never a `busy=0 && done=0` window); `done` is a single-cycle pulse.
//   - one fc_valid/fc_class/fc_acc pulse per class, classes 0..9 in order.
//   - `prediction` is valid at done and HOLDEN until the next start; consecutive
//     runs without reset are supported (start re-arms everything).
//
// Cycle budget: per class PROLOGUE(1) + MAC(32) + ADD_BIAS(1) + FC_VALID(1) = 35;
//   10 classes -> 350 + 1 (S_DONE) = 351 cycles start -> done.
//
// Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module fc_argmax_serial #(
    // Relative init paths, resolved from the Questa working directory (repo
    // root).  The Quartus smoke wrapper overrides them with paths that resolve
    // from the project directory.  No absolute paths.
    parameter WT_MEM_FILE    = "fpga/baseline_cnn/params/weights/fc_weight.mem",
    parameter BIAS_MEM_FILE  = "fpga/baseline_cnn/params/biases/fc_bias.mem"
) (
    // ---- clock / reset ----
    input  wire              clk,
    input  wire              rst_n,

    // ---- GAP result write port (32 x UINT8 register array) ----
    input  wire              gap_we,     // qualified write (sampled at posedge)
    input  wire [4:0]        gap_waddr,  // channel index, strictly 0..31
    input  wire [7:0]        gap_wdata,  // UINT8 GAP result

    // ---- control ----
    input  wire              start,      // single-cycle pulse; ignored while busy
    output reg               busy,       // 1 from start until done (never drops first)
    output reg               done,       // single-cycle pulse at end of run

    // ---- fc debug stream (one pulse per class) ----
    output reg               fc_valid,   // one-cycle pulse per fc_acc
    output reg  [3:0]        fc_class,   // strictly 0..9
    output reg signed [31:0] fc_acc,     // INT32 accumulator (post-bias)

    // ---- prediction ----
    output reg  [3:0]        prediction  // signed argmax; valid at done, held
);
    // ================= frozen geometry =================
    localparam FC_OC = 10;             // classes
    localparam FC_IC = 32;             // gap features per class
    localparam OC_LAST = FC_OC - 1;    // 9
    localparam F_LAST  = FC_IC - 1;    // 31
    localparam WT_DEPTH = 320;         // 10 x 32 OI
    localparam WT_ADDR_W = 9;

    // INT32 saturation bounds (clamp, no wraparound).
    localparam signed [63:0] INT32_MAX =  64'sd2147483647;
    localparam signed [63:0] INT32_MIN = -64'sd2147483648;

    // ================= FSM states =================
    localparam S_IDLE      = 3'd0;
    localparam S_PROLOGUE  = 3'd1;
    localparam S_MAC       = 3'd2;
    localparam S_ADD_BIAS  = 3'd3;
    localparam S_FC_VALID  = 3'd4;
    localparam S_DONE      = 3'd5;

    reg [2:0] state;
    reg [3:0] oc;              // current class 0..9
    reg [4:0] f;               // current feature 0..31
    reg signed [63:0] acc64;   // exact accumulator (S64)
    reg signed [31:0] best;    // best-so-far fc_acc (argmax)

    // ================= gap register array (32 x UINT8) =================
    reg [7:0] gap_mem [0:31];
    always @(posedge clk) begin
        if (gap_we) gap_mem[gap_waddr] <= gap_wdata;
    end

    // ================= weight-ROM issue gating =================
    // Software-pipelined like the conv engine: PROLOGUE presents tap0's
    // address; each MAC cycle presents the NEXT tap's address (the ROM samples
    // it at the posedge, data valid the following cycle).  Outside PROLOGUE/MAC
    // (and on the last tap) the ROM is addressed at 0 so it is never read out
    // of the 320-cell array.
    wire [4:0] iss_f = (state == S_PROLOGUE) ? 5'd0 :
                       ((state == S_MAC) && (f != F_LAST)) ? (f + 5'd1) : 5'd0;
    wire issuing = (state == S_PROLOGUE) || ((state == S_MAC) && (f != F_LAST));
    wire [8:0]  wt_raddr = issuing ? {oc, iss_f} : 9'd0;

    // ================= weight / bias ROMs =================
    wire signed [7:0]  wt_rdata;
    wire signed [31:0] bias_rdata;
    // The bias ROM is addressed by the stable `oc`; the weight ROM read port is
    // driven by the software-pipelined issue address `wt_raddr` (NOT the raw
    // `{oc, f}`): the ROM has one-cycle latency, so the address for tap k+1 must
    // be presented one cycle before its data is accumulated.
    sync_rom_s8  #(.DEPTH(WT_DEPTH), .ADDR_W(WT_ADDR_W), .FILE(WT_MEM_FILE))
        u_wt_rom   (.clk(clk), .addr(wt_raddr), .rdata(wt_rdata));
    sync_rom_s32 #(.DEPTH(FC_OC), .ADDR_W(4), .FILE(BIAS_MEM_FILE))
        u_bias_rom (.clk(clk), .addr(oc), .rdata(bias_rdata));

    // ================= datapath =================
    // UINT8 gap x SINT8 weight, widened to 17 signed bits (never truncates).
    wire signed [16:0] tap_product = $signed({9'd0, gap_mem[f]}) *
                                     $signed({{9{wt_rdata[7]}}, wt_rdata});

    // INT32 saturation AFTER bias is folded into acc64 (no wraparound).
    wire signed [31:0] acc32 = (acc64 > INT32_MAX) ? 32'sh7FFFFFFF :
                               (acc64 < INT32_MIN) ? 32'sh80000000 :
                               acc64[31:0];

    wire last_f  = (f == F_LAST);
    wire last_oc = (oc == OC_LAST);

    // ================= combinational outputs =================
    always @* begin
        busy     = (state != S_IDLE);       // covers PROLOGUE..DONE
        fc_valid = (state == S_FC_VALID);
        fc_class = oc;
        fc_acc   = acc32;
    end

    // ================= FSM =================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            oc         <= 4'd0;
            f          <= 5'd0;
            acc64      <= 64'sd0;
            best       <= 32'sd0;
            prediction <= 4'd0;
            done       <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        oc    <= 4'd0;
                        f     <= 5'd0;
                        acc64 <= 64'sd0;
                        state <= S_PROLOGUE;
                    end
                end
                S_PROLOGUE: begin
                    // tap0's weight address is presented this cycle (wt_raddr =
                    // {oc,0}); clear the accumulator for this class.
                    acc64 <= 64'sd0;
                    state <= S_MAC;
                end
                S_MAC: begin
                    // accumulate the tap whose data arrived this cycle, and
                    // issue the next tap's read (software-pipelined).  The last
                    // tap drops out of the loop with no new issue.
                    acc64 <= acc64 + tap_product;
                    if (last_f) begin
                        state <= S_ADD_BIAS;
                    end else begin
                        f <= f + 5'd1;
                    end
                end
                S_ADD_BIAS: begin
                    // bias ROM sampled oc long ago -> data valid now; add once
                    // after all 32 MACs.
                    acc64 <= acc64 + $signed(bias_rdata);
                    state <= S_FC_VALID;
                end
                S_FC_VALID: begin
                    // fc_acc = saturate(acc64) is valid this whole cycle; emit
                    // the fc_valid pulse and run the signed argmax with the
                    // frozen tie rule (class 0 seeds unconditionally; later
                    // classes only on strict >, so an equal score keeps the
                    // smaller class index).
                    if (oc == 4'd0) begin
                        best       <= acc32;
                        prediction <= 4'd0;
                    end else if (acc32 > best) begin
                        best       <= acc32;
                        prediction <= oc;
                    end
                    if (last_oc) begin
                        state <= S_DONE;
                    end else begin
                        oc    <= oc + 4'd1;
                        f     <= 5'd0;
                        acc64 <= 64'sd0;
                        state <= S_PROLOGUE;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;      // single-cycle pulse; prediction held
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
