// gap_stream_u8.v - Streaming global-average-pool over conv3_q.
//
// Consumes the conv3_q stream directly (32x7x7, CHW, channel -> y -> x, 49
// UINT8 values per channel) and emits one GAP result per channel:
//     sum[c]   = sum over the channel's 49 UINT8 inputs
//     gap_q[c] = round_half_away_from_zero(sum[c] / 49)
//              = floor((sum[c] + 24) / 49), saturated to UINT8 [0,255]
// computed by the frozen, already-verified gap_div49.v (never reimplemented).
//
// Streaming contract (mirrors maxpool2x2_stream):
//   - ONE `start` pulse launches the run; `busy` stays 1 until the done pulse
//     (RUN..DONE), never a `busy=0 && done=0` window before done.
//   - input count / accumulator advance ONLY on (busy && in_valid); arbitrary-
//     length gaps in the stream are tolerated and never advance state.
//   - every 49 consumed inputs produce one `out_valid` pulse with out_channel
//     strictly 0..31; 32 outputs in total (channel order).
//   - `done` is a single-cycle pulse AFTER the last output's value has been
//     presented (the last out_valid cycle precedes the done cycle by one).
//   - start and the first in_valid are never on the same sampling edge: in the
//     integration the retimed conv3 engine's first q_valid is 293 cycles after start.
//
// The accumulator is a 14-bit unsigned register (49*255 = 12495 < 2^14).
// No division / modulo anywhere: gap_div49.v is a constant-division multiply
// (n*2675 >> 17), NOT a divider.
//
// Cycle budget (no gaps): 1568 inputs consumed, done 2 cycles after the 1568th
// input (1 cycle to the last out_valid, 1 more for the S_DONE done pulse).
//
// Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module gap_stream_u8 (
    input  wire       clk,
    input  wire       rst_n,

    input  wire       start,     // single-cycle pulse; ignored while busy
    input  wire       in_valid,  // qualified conv3_q input (sampled at posedge)
    input  wire [7:0] in_q,      // UINT8 conv3_q value

    output reg        busy,      // 1 from start until done (never drops first)
    output reg        out_valid, // one-cycle pulse per GAP result (32 total)
    output reg  [4:0] out_channel, // channel index, strictly 0..31
    output reg  [7:0] out_q,     // UINT8 GAP result (round-half-away-from-zero)
    output reg        done       // single-cycle pulse after the last output
);
    // ================= frozen geometry =================
    localparam N_CH    = 32;     // conv3 output channels
    localparam CH_MAX  = N_CH - 1;
    // 49 = 7x7 inputs per channel.  Counters count CONSUMED inputs, so the
    // "last of 49" is cnt == 48.
    localparam CNT_LAST = 6'd48;

    // ================= FSM states =================
    localparam S_IDLE = 2'd0;
    localparam S_RUN  = 2'd1;
    localparam S_DONE = 2'd2;

    reg [1:0] state;

    // current channel / per-channel item counter / accumulator
    reg [4:0]  chn;              // 0..31
    reg [5:0]  cnt;              // 0..48 consumed inputs in the current channel
    reg [13:0] sum;              // 14-bit unsigned per-channel sum (<= 12495)

    // registered output capture (presented one cycle after the producing input)
    reg               out_valid_r;
    reg [4:0]         out_channel_r;
    reg [7:0]         out_q_r;

    // ================= combinational helpers =================
    // next sum including the current qualified input (combinational add)
    wire [13:0] sum_next = sum + {6'd0, in_q};
    // the 49th qualified input of the current channel produces the output
    wire produce = (state == S_RUN) && in_valid && (cnt == CNT_LAST);
    wire last_ch = (chn == 5'd31);

    // reuse the frozen integer GAP division (pure combinational)
    wire [7:0] gap_q_w;
    gap_div49 u_gap (
        .sum (sum_next),
        .q   (gap_q_w)
    );

    // ================= outputs =================
    // busy covers S_DONE: stays 1 until the done pulse, never drops first.
    always @* begin
        busy        = (state == S_RUN) || (state == S_DONE);
        out_valid   = out_valid_r;
        out_channel = out_channel_r;
        out_q       = out_q_r;
    end

    // ================= FSM / output capture =================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            chn            <= 5'd0;
            cnt            <= 6'd0;
            sum            <= 14'd0;
            out_valid_r    <= 1'b0;
            out_channel_r  <= 5'd0;
            out_q_r        <= 8'd0;
            done           <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    out_valid_r <= 1'b0;
                    if (start) begin
                        chn   <= 5'd0;
                        cnt   <= 6'd0;
                        sum   <= 14'd0;
                        state <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (in_valid) begin
                        // consume the input: accumulate, advance the count,
                        // and on the 49th item emit the GAP result.
                        sum <= (produce) ? 14'd0 : sum_next;
                        cnt <= (produce) ? 6'd0  : (cnt + 6'd1);
                        out_valid_r   <= produce;
                        out_q_r       <= gap_q_w;       // gap_div49(sum_next)
                        out_channel_r <= chn;
                        if (produce) begin
                            if (last_ch) begin
                                state <= S_DONE;   // all 32 channels emitted
                            end else begin
                                chn <= chn + 5'd1;
                            end
                        end
                    end else begin
                        // gap: no advance, no capture
                        out_valid_r <= 1'b0;
                    end
                end
                S_DONE: begin
                    // single-cycle done pulse, strictly AFTER the last output
                    done        <= 1'b1;
                    out_valid_r <= 1'b0;
                    state       <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
