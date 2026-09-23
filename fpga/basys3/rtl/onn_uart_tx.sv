`timescale 1ns/1ps

module onn_uart_tx #(parameter integer CLKS_PER_BIT = 868) (
    input wire clk, input wire rst_n, input wire [7:0] data,
    input wire valid, output wire ready, output reg tx = 1
);
    reg [1:0] state = 0;
    integer tick = 0;
    reg [2:0] bit_index = 0;
    reg [7:0] shift = 0;
    assign ready = (state == 0);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= 0; tick <= 0; bit_index <= 0; shift <= 0; tx <= 1;
        end else begin
            case (state)
                0: if (valid) begin
                       shift <= data; tx <= 0; tick <= CLKS_PER_BIT - 1; state <= 1;
                   end
                1: if (tick == 0) begin
                       tx <= shift[bit_index]; tick <= CLKS_PER_BIT - 1; state <= 2;
                   end else tick <= tick - 1;
                2: if (tick == 0) begin
                       if (bit_index == 7) begin
                           tx <= 1; tick <= CLKS_PER_BIT - 1; state <= 3;
                       end else begin
                           bit_index <= bit_index + 1;
                           tx <= shift[bit_index + 1];
                           tick <= CLKS_PER_BIT - 1;
                       end
                   end else tick <= tick - 1;
                3: if (tick == 0) begin bit_index <= 0; state <= 0; end
                   else tick <= tick - 1;
                default: state <= 0;
            endcase
        end
    end
endmodule
