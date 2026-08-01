// maxpool2x2_stream.v - Streaming 2x2 stride-2 max-pool primitive.
//
// Consumes the stem engine's q_valid stream directly:
//     stem_q  16x28x28 (UINT8, CHW, oc -> y -> x order)
//     -> 2x2 MaxPool, stride 2
//     -> pool1_q 16x14x14 (UINT8, CHW, oc -> pool_y -> pool_x order)
//
// One run accepts EXACTLY 12544 inputs in fixed oc -> y -> x order.  The
// internal oc/y/x counters advance ONLY on (busy && in_valid); arbitrary-length
// gaps in the input stream are tolerated and never advance any state.  Outputs
// arrive one per completed 2x2 window, in oc -> pool_y -> pool_x order, with
// out_addr strictly 0..3135 (one pooled result per increment).
//
// Streaming structure: only the previous even row is kept (row_buffer[0:27],
// 28x8 = 224 bits).  Odd rows build the 2x2 window from row_buffer plus a
// one-cell odd-row latch (current_left):
//     top    = row_buffer[x-1], row_buffer[x]   (previous even row)
//     bottom = current_left, in_q               (current odd row)
//     out_q  = max(top, bottom)                  (unsigned UINT8 compare tree)
//
// No division / modulo / general multiply anywhere: oc/y/x and out_addr are
// plain counters.  UINT8 compares are unsigned - never use signed compares on
// UINT8 feature maps.  row_buffer reads are guarded so a window is only formed
// at odd x (x-1 >= 0), keeping the data path free of X even on the first row.
//
// This is the standalone, integration-ready primitive; it does NOT read/write
// any feature-map RAM (the stem output RAM stays for phase-1 verification and
// is removed only when the engine streams directly into this module).
//
// Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module maxpool2x2_stream (
    input  wire       clk,
    input  wire       rst_n,

    input  wire       start,     // single-cycle pulse; ignored while busy
    input  wire       in_valid,  // qualified input (sampled at posedge)
    input  wire [7:0] in_q,      // UINT8 feature-map value

    output reg        busy,      // 1 while consuming the input stream
    output reg        out_valid, // one-cycle pulse per pooled result
    output reg [11:0] out_addr,  // CHW pool address, strictly 0..3135
    output reg [7:0]  out_q,     // UINT8 pooled value
    output reg        done       // single-cycle pulse after the last output
);
    // ================= frozen geometry =================
    localparam IN_H       = 28;
    localparam IN_W       = 28;
    localparam POOL_H     = 14;   // IN_H/2
    localparam POOL_W     = 14;   // IN_W/2
    localparam N_INPUTS   = 12544; // 16*28*28
    localparam N_OUTPUTS  = 3136;  // 16*14*14
    localparam OUT_ADDR_W = 12;

    // ================= FSM states =================
    localparam S_IDLE = 2'd0;
    localparam S_RUN  = 2'd1;
    localparam S_DONE = 2'd2;

    reg [1:0] state;

    // current input coordinates (advance only on busy && in_valid)
    reg [3:0]  oc;            // 0..15
    reg [4:0]  y;             // 0..27
    reg [4:0]  x;             // 0..27
    reg [OUT_ADDR_W-1:0] out_cnt;   // pooled results emitted so far (next addr)

    // streaming window storage
    reg [7:0] row_buffer [0:27];    // previous even row (28 x 8 = 224 bits)
    reg [7:0] current_left;         // odd row, previous (even) x value

    // registered outputs (captured at the produce edge)
    reg               out_valid_r;
    reg [OUT_ADDR_W-1:0] out_addr_r;
    reg [7:0]         out_q_r;

    // ================= combinational window =================
    // Guarded reads: row_buffer only holds a *completed previous even row* and
    // is only read while forming a window (odd row, odd x), so x-1 >= 0 is
    // guaranteed.  Forcing 0 at even rows / even x keeps max_all deterministic
    // (no X) even on the first row, when row_buffer is not yet written.
    wire [7:0] row_l = (y[0] && x[0]) ? row_buffer[x - 5'd1] : 8'd0;
    wire [7:0] row_r = (y[0] && x[0]) ? row_buffer[x]       : 8'd0;
    wire [7:0] max_top    = (row_l > row_r)          ? row_l : row_r;
    wire [7:0] max_bottom = (current_left > in_q)    ? current_left : in_q;
    wire [7:0] max_all    = (max_top > max_bottom)   ? max_top : max_bottom;

    // this cycle produces a pooled result: odd row, odd x, qualified input
    wire produce = (state == S_RUN) && in_valid && y[0] && x[0];

    // last input of the run
    wire last_input = (oc == 4'd15) && (y == 5'd27) && (x == 5'd27);

    // ================= storage writes =================
    // even row: overwrite row_buffer[x].  A new channel's even row writes all
    // 28 positions before its odd row reads any, so no clear is needed on
    // channel switch.  odd row, even x: latch the odd row's left value.
    // current_left is reset here (same block as its writes) so the window
    // datapath is free of X before the first odd-row write.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_left <= 8'd0;
        end else if (state == S_RUN && in_valid) begin
            if (!y[0])      row_buffer[x] <= in_q;
            else if (!x[0]) current_left <= in_q;
        end
    end

    // ================= outputs (busy is combinational; the rest registered) =================
    always @* begin
        busy      = (state == S_RUN);
        out_valid = out_valid_r;
        out_addr  = out_addr_r;
        out_q     = out_q_r;
    end

    // ================= FSM / output capture =================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            oc         <= 4'd0;
            y          <= 5'd0;
            x          <= 5'd0;
            out_cnt    <= {OUT_ADDR_W{1'b0}};
            out_valid_r <= 1'b0;
            out_addr_r <= {OUT_ADDR_W{1'b0}};
            out_q_r    <= 8'd0;
            done       <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    out_valid_r <= 1'b0;
                    if (start) begin
                        oc      <= 4'd0;
                        y       <= 5'd0;
                        x       <= 5'd0;
                        out_cnt <= {OUT_ADDR_W{1'b0}};
                        state   <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (in_valid) begin
                        // capture this cycle's window output (produce or not);
                        // out_addr_r holds the pre-increment count so the
                        // valid cycle advertises the current pooled index.
                        out_valid_r <= produce;
                        out_q_r     <= max_all;
                        out_addr_r  <= out_cnt;
                        if (produce) out_cnt <= out_cnt + 12'd1;

                        if (last_input) begin
                            state <= S_DONE;
                        end else begin
                            // advance input coordinates (oc -> y -> x)
                            x <= (x == 5'd27) ? 5'd0 : (x + 5'd1);
                            y <= (x == 5'd27) ? ((y == 5'd27) ? 5'd0 : (y + 5'd1)) : y;
                            oc <= (x == 5'd27 && y == 5'd27) ? (oc + 4'd1) : oc;
                        end
                    end else begin
                        // gap: no advance, no capture
                        out_valid_r <= 1'b0;
                    end
                end
                S_DONE: begin
                    // single-cycle done pulse, strictly AFTER the last output
                    // (the last out_valid_r was captured on entry to S_DONE).
                    done        <= 1'b1;
                    out_valid_r <= 1'b0;
                    state       <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
