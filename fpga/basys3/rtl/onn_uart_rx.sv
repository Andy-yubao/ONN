`timescale 1ns/1ps

module onn_uart_rx #(parameter integer CLKS_PER_BIT = 868) (
    input wire clk, input wire rst_n, input wire rx,
    output reg [7:0] data = 0, output reg valid = 0
);
    reg sync1 = 1, sync2 = 1;
    reg [1:0] state = 0;
    integer tick = 0;
    reg [2:0] bit_index = 0;
    reg [7:0] shift = 0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sync1 <= 1; sync2 <= 1; state <= 0; tick <= 0;
            bit_index <= 0; shift <= 0; data <= 0; valid <= 0;
        end else begin
            sync1 <= rx;
            sync2 <= sync1;
            valid <= 0;
            case (state)
                0: if (!sync2) begin tick <= CLKS_PER_BIT / 2; state <= 1; end
                1: if (tick == 0) begin
                       if (!sync2) begin tick <= CLKS_PER_BIT - 1; bit_index <= 0; state <= 2; end
                       else state <= 0;
                   end else tick <= tick - 1;
                2: if (tick == 0) begin
                       shift[bit_index] <= sync2;
                       tick <= CLKS_PER_BIT - 1;
                       if (bit_index == 7) state <= 3;
                       else bit_index <= bit_index + 1;
                   end else tick <= tick - 1;
                3: if (tick == 0) begin
                       if (sync2) begin data <= shift; valid <= 1; end
                       state <= 0;
                   end else tick <= tick - 1;
                default: state <= 0;
            endcase
        end
    end
endmodule
