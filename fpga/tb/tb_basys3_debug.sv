`timescale 1ns/1ps

module tb_basys3_debug;
    localparam integer CLKS_PER_BIT = 8;
    reg clk = 0, btn_c = 1, rs_rx = 1;
    wire rs_tx;
    wire [3:0] led;
    wire [7:0] tx_data;
    wire tx_data_valid;
    reg [7:0] request_bytes [0:300];
    reg [7:0] expected_bytes [0:72];
    integer received = 0, timeout, i, bit_index;

    always #5 clk = ~clk;
    onn_basys3_top #(
        .CLKS_PER_BIT(CLKS_PER_BIT), .DEBUG_BUILD(1), .USE_MMCM(0),
        .CONV1_WEIGHT_FILE("../../model/snn/export/conv1_weight.mem"),
        .CONV2_WEIGHT_FILE("../../model/snn/export/conv2_weight.mem"),
        .READOUT_WEIGHT_FILE("../../model/snn/export/readout_weight.mem")
    ) dut (.clk_100m(clk), .btn_c(btn_c), .rs_rx(rs_rx), .rs_tx(rs_tx), .led(led));
    onn_uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) monitor (
        .clk(clk), .rst_n(!btn_c), .rx(rs_tx), .data(tx_data), .valid(tx_data_valid));
    always @(posedge clk) if (tx_data_valid) begin
        if (received >= 73 || tx_data !== expected_bytes[received])
            $fatal(1, "debug UART mismatch byte %0d got %02X expected %02X",
                   received, tx_data, expected_bytes[received]);
        received = received + 1;
    end

    task automatic send_byte(input [7:0] value);
        begin
            @(negedge clk); rs_rx = 0;
            repeat (CLKS_PER_BIT) @(negedge clk);
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                rs_rx = value[bit_index];
                repeat (CLKS_PER_BIT) @(negedge clk);
            end
            rs_rx = 1;
            repeat (CLKS_PER_BIT) @(negedge clk);
        end
    endtask

    initial begin
        $readmemh("../vectors/uart_requests.mem", request_bytes);
        $readmemh("../vectors/uart_debug_response.mem", expected_bytes);
        repeat (5) @(negedge clk);
        btn_c = 0;
        repeat (5) @(negedge clk);
        for (i = 86; i < 129; i = i + 1) send_byte(request_bytes[i]);
        timeout = 0;
        while (received < 73) begin
            @(negedge clk);
            timeout = timeout + 1;
            if (timeout > 300000) $fatal(1, "debug UART timeout");
        end
        $display("PASS tb_basys3_debug: 73-byte MNIST debug result matched, including L1/L2 counts");
        $finish;
    end
endmodule
