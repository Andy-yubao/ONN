`timescale 1ns/1ps

// USB-UART transport around the frozen sparse SNN. No model arithmetic lives here.
module onn_basys3_top #(
    parameter integer CLKS_PER_BIT = 217,
    parameter integer DEBUG_BUILD = 0,
    parameter integer USE_MMCM = 1,
    parameter integer RX_TIMEOUT_CLKS = 2500000,
    parameter CONV1_WEIGHT_FILE = "model/snn/export/conv1_weight.mem",
    parameter CONV2_WEIGHT_FILE = "model/snn/export/conv2_weight.mem",
    parameter READOUT_WEIGHT_FILE = "model/snn/export/readout_weight.mem"
) (
    input wire clk_100m,
    input wire btn_c,
    input wire rs_rx,
    output wire rs_tx,
    output wire [3:0] led
);
    wire clk_logic;
    wire clock_locked;
    generate if (USE_MMCM) begin : clock_25m
        wire clock_unbuffered, feedback_unbuffered, feedback_buffered;
        MMCME2_BASE #(
            .CLKIN1_PERIOD(10.000), .DIVCLK_DIVIDE(1),
            .CLKFBOUT_MULT_F(8.000), .CLKOUT0_DIVIDE_F(32.000)
        ) mmcm (
            .CLKIN1(clk_100m), .CLKFBIN(feedback_buffered),
            .CLKFBOUT(feedback_unbuffered), .CLKOUT0(clock_unbuffered),
            .LOCKED(clock_locked), .PWRDWN(1'b0), .RST(btn_c)
        );
        BUFG feedback_buf (.I(feedback_unbuffered), .O(feedback_buffered));
        BUFG logic_buf (.I(clock_unbuffered), .O(clk_logic));
    end else begin : clock_sim
        assign clk_logic = clk_100m;
        assign clock_locked = 1'b1;
    end endgenerate

    reg [1:0] reset_release = 0;
    always @(posedge clk_logic or posedge btn_c or negedge clock_locked) begin
        if (btn_c || !clock_locked) reset_release <= 0;
        else reset_release <= {reset_release[0], 1'b1};
    end
    wire rst_n = reset_release[1];

    wire [7:0] rx_byte;
    wire rx_valid;
    reg [7:0] tx_byte = 0;
    reg tx_valid = 0;
    wire tx_ready;
    onn_uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) uart_rx (
        .clk(clk_logic), .rst_n(rst_n), .rx(rs_rx), .data(rx_byte), .valid(rx_valid));
    onn_uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) uart_tx (
        .clk(clk_logic), .rst_n(rst_n), .data(tx_byte), .valid(tx_valid),
        .ready(tx_ready), .tx(rs_tx));

    // Receive a fixed 36-byte request payload: four (step index, 64-bit word) pairs.
    reg [2:0] parse_state = 0;
    reg [7:0] rx_sequence = 0, rx_length = 0, rx_xor = 0, payload_index = 0;
    reg [$clog2(RX_TIMEOUT_CLKS + 1)-1:0] rx_idle_cycles = 0;
    reg bad_payload = 0;
    reg [63:0] receive_words [0:3];
    reg packet_valid = 0, packet_error = 0;
    reg [7:0] packet_error_code = 0;
    integer step_slot, byte_slot;
    always_comb begin
        step_slot = payload_index / 9;
        byte_slot = payload_index % 9;
    end
    always_ff @(posedge clk_logic) begin
        if (rst_n && rx_valid && parse_state == 6 && rx_length == 36 &&
            step_slot < 4 && byte_slot != 0)
            receive_words[step_slot][8 * (byte_slot - 1) +: 8] <= rx_byte;
    end
    always @(posedge clk_logic or negedge rst_n) begin
        if (!rst_n) begin
            parse_state <= 0; rx_sequence <= 0; rx_length <= 0;
            rx_xor <= 0; payload_index <= 0; bad_payload <= 0;
            rx_idle_cycles <= 0;
            packet_valid <= 0; packet_error <= 0; packet_error_code <= 0;
        end else begin
            packet_valid <= 0;
            packet_error <= 0;
            if (rx_valid || parse_state == 0) rx_idle_cycles <= 0;
            else if (rx_idle_cycles == RX_TIMEOUT_CLKS - 1) begin
                rx_idle_cycles <= 0;
                parse_state <= 0;
                if (parse_state >= 4) begin
                    packet_error <= 1;
                    packet_error_code <= 2;
                end
            end else rx_idle_cycles <= rx_idle_cycles + 1'b1;
            if (rx_valid) begin
                case (parse_state)
                    0: if (rx_byte == 8'hA5) parse_state <= 1;
                    1: if (rx_byte == 8'h5A) parse_state <= 2;
                       else if (rx_byte != 8'hA5) parse_state <= 0;
                    2: if (rx_byte == 8'h01) begin
                           rx_xor <= rx_byte; parse_state <= 3;
                       end else parse_state <= 0;
                    3: begin rx_sequence <= rx_byte; rx_xor <= rx_xor ^ rx_byte; parse_state <= 4; end
                    4: begin
                           bad_payload <= (rx_byte != 8'h01);
                           rx_xor <= rx_xor ^ rx_byte; parse_state <= 5;
                       end
                    5: begin
                           rx_length <= rx_byte;
                           bad_payload <= bad_payload || (rx_byte != 8'd36);
                           rx_xor <= rx_xor ^ rx_byte;
                           payload_index <= 0;
                           if (rx_byte != 8'd36) begin
                               packet_error <= 1;
                               packet_error_code <= 2;
                               parse_state <= 0;
                           end else parse_state <= 6;
                       end
                    6: begin
                           rx_xor <= rx_xor ^ rx_byte;
                           if (rx_length == 36 && step_slot < 4) begin
                               if (byte_slot == 0) begin
                                   if (rx_byte != step_slot) bad_payload <= 1;
                               end
                           end
                           if (payload_index == rx_length - 1) parse_state <= 7;
                           else payload_index <= payload_index + 1;
                       end
                    7: begin
                           if ((rx_xor ^ rx_byte) != 0) begin
                               packet_error <= 1; packet_error_code <= 1;
                           end else if (bad_payload) begin
                               packet_error <= 1; packet_error_code <= 2;
                           end else packet_valid <= 1;
                           parse_state <= 0;
                       end
                    default: parse_state <= 0;
                endcase
            end
        end
    end

    reg [63:0] active_words [0:3];
    reg [2:0] state = 0;
    localparam S_IDLE=0, S_LAUNCH=1, S_WAIT=2, S_SEND=3,
               S_TX_START=4, S_TX_ACCEPT=5, S_TX_WAIT=6;
    reg [1:0] step_index = 0;
    reg [7:0] response_sequence = 0, response_status = 0;
    reg [6:0] tx_index = 0, tx_total = 0;
    reg [7:0] tx_xor = 0;
    reg [63:0] core_spike = 0;
    reg core_valid = 0, core_frame_start = 0;
    wire core_ready, core_step_done, core_frame_done;
    wire [3:0] core_class;
    wire readout_valid, l1_valid, l1_spike, l2_valid, l2_spike;
    wire [3:0] readout_class;
    wire signed [20:0] weighted_score;
    wire [31:0] frame_additions, frame_cycles;
    reg signed [20:0] final_scores [0:9];
    reg [15:0] l1_counts [0:3], l2_counts [0:3];
    integer i;

    onn_snn_core #(
        .CONV1_WEIGHT_FILE(CONV1_WEIGHT_FILE),
        .CONV2_WEIGHT_FILE(CONV2_WEIGHT_FILE),
        .READOUT_WEIGHT_FILE(READOUT_WEIGHT_FILE)
    ) core (
        .clk(clk_logic), .rst_n(rst_n), .spike_in(core_spike),
        .step_valid(core_valid), .frame_start(core_frame_start),
        .ready(core_ready), .step_done(core_step_done), .frame_done(core_frame_done),
        .class_out(core_class), .l1_valid(l1_valid), .l1_spike(l1_spike),
        .l2_valid(l2_valid), .l2_spike(l2_spike),
        .readout_valid(readout_valid), .readout_class(readout_class),
        .weighted_score(weighted_score),
        .frame_synaptic_additions(frame_additions), .frame_compute_cycles(frame_cycles)
    );

    function automatic [7:0] response_byte(input integer index);
        reg signed [31:0] signed_score;
        reg [31:0] unsigned_value;
        integer entry;
        begin
            response_byte = 0;
            if (index == 0) response_byte = 8'hA5;
            else if (index == 1) response_byte = 8'h5A;
            else if (index == 2) response_byte = 8'h01;
            else if (index == 3) response_byte = response_sequence;
            else if (index == 4) response_byte = (DEBUG_BUILD && response_status == 0) ? 8'h82 : 8'h81;
            else if (index == 5) response_byte = (DEBUG_BUILD && response_status == 0) ? 8'd66 : 8'd50;
            else if (index == 6) response_byte = response_status;
            else if (index == 7) response_byte = response_status == 0 ? {4'b0, core_class} : 0;
            else if (index >= 8 && index < 48) begin
                entry = (index - 8) / 4;
                signed_score = response_status == 0 ? $signed(final_scores[entry]) : 0;
                response_byte = signed_score >> (8 * ((index - 8) % 4));
            end else if (index >= 48 && index < 56) begin
                unsigned_value = response_status == 0 ?
                    (index < 52 ? frame_additions : frame_cycles) : 0;
                response_byte = unsigned_value >> (8 * ((index - 48) % 4));
            end else if (index >= 56 && index < 72 && DEBUG_BUILD) begin
                entry = (index - 56) / 4;
                unsigned_value = (index - 56) % 4 < 2 ? l1_counts[entry] : l2_counts[entry];
                response_byte = unsigned_value >> (8 * ((index - 56) % 2));
            end else if (index == tx_total - 1) response_byte = tx_xor;
        end
    endfunction

    always @(posedge clk_logic or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; step_index <= 0; response_sequence <= 0;
            response_status <= 0; tx_index <= 0; tx_total <= 0;
            tx_xor <= 0; tx_byte <= 0; tx_valid <= 0;
            core_spike <= 0; core_valid <= 0; core_frame_start <= 0;
            for (i = 0; i < 4; i = i + 1) begin
                active_words[i] <= 0; l1_counts[i] <= 0; l2_counts[i] <= 0;
            end
            for (i = 0; i < 10; i = i + 1) final_scores[i] <= 0;
        end else begin
            tx_valid <= 0;
            core_valid <= 0;
            core_frame_start <= 0;
            if (readout_valid) final_scores[readout_class] <= weighted_score;
            if (state == S_WAIT) begin
                if (l1_valid && l1_spike) l1_counts[step_index] <= l1_counts[step_index] + 1;
                if (l2_valid && l2_spike) l2_counts[step_index] <= l2_counts[step_index] + 1;
            end
            case (state)
                S_IDLE: begin
                    if (packet_valid) begin
                        for (i = 0; i < 4; i = i + 1) begin
                            active_words[i] <= receive_words[i];
                            l1_counts[i] <= 0; l2_counts[i] <= 0;
                        end
                        for (i = 0; i < 10; i = i + 1) final_scores[i] <= 0;
                        response_sequence <= rx_sequence;
                        response_status <= 0;
                        step_index <= 0;
                        state <= S_LAUNCH;
                    end else if (packet_error) begin
                        response_sequence <= rx_sequence;
                        response_status <= packet_error_code;
                        state <= S_SEND;
                    end
                end
                S_LAUNCH: if (core_ready) begin
                    core_spike <= active_words[step_index];
                    core_frame_start <= step_index == 0;
                    core_valid <= 1;
                    state <= S_WAIT;
                end
                S_WAIT: if (core_step_done) begin
                    if (step_index == 3) state <= S_SEND;
                    else begin step_index <= step_index + 1; state <= S_LAUNCH; end
                end
                S_SEND: begin
                    tx_total <= (DEBUG_BUILD && response_status == 0) ? 73 : 57;
                    tx_index <= 0;
                    tx_xor <= 0;
                    state <= S_TX_START;
                end
                S_TX_START: if (tx_ready) begin
                    tx_byte <= response_byte(tx_index);
                    tx_valid <= 1;
                    if (tx_index >= 2 && tx_index < tx_total - 1)
                        tx_xor <= tx_xor ^ response_byte(tx_index);
                    state <= S_TX_ACCEPT;
                end
                S_TX_ACCEPT: if (!tx_ready) state <= S_TX_WAIT;
                S_TX_WAIT: if (tx_ready) begin
                    if (tx_index == tx_total - 1) state <= S_IDLE;
                    else begin tx_index <= tx_index + 1; state <= S_TX_START; end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
    assign led = {state != S_IDLE, response_status != 0, core_frame_done, rst_n};
endmodule
