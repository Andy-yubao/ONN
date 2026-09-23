`timescale 1ns/1ps

module tb_basys3_uart;
    localparam integer CLKS_PER_BIT = 8;
    reg clk = 0, btn_c = 1, rs_rx = 1;
    wire rs_tx;
    wire [3:0] led;
    wire [7:0] tx_data;
    wire tx_data_valid;
    reg [7:0] request_bytes [0:300];
    reg [7:0] expected_bytes [0:398];
    integer received = 0;
    integer timeout;
    integer i, bit_index;

    always #5 clk = ~clk;
    onn_basys3_top #(
        .CLKS_PER_BIT(CLKS_PER_BIT), .USE_MMCM(0), .RX_TIMEOUT_CLKS(500),
        .CONV1_WEIGHT_FILE("../../model/snn/export/conv1_weight.mem"),
        .CONV2_WEIGHT_FILE("../../model/snn/export/conv2_weight.mem"),
        .READOUT_WEIGHT_FILE("../../model/snn/export/readout_weight.mem")
    ) dut (
        .clk_100m(clk), .btn_c(btn_c), .rs_rx(rs_rx), .rs_tx(rs_tx), .led(led));
    onn_uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) monitor (
        .clk(clk), .rst_n(!btn_c), .rx(rs_tx), .data(tx_data), .valid(tx_data_valid));

    always @(posedge clk) begin
        if (tx_data_valid) begin
            if (received >= 399 || tx_data !== expected_bytes[received])
                $fatal(1, "UART response byte %0d got %02X expected %02X",
                    received, tx_data, expected_bytes[received]);
            received = received + 1;
        end
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

    task automatic send_request(input integer number);
        integer offset;
        begin
            for (offset = 0; offset < 43; offset = offset + 1)
                send_byte(request_bytes[number * 43 + offset]);
        end
    endtask

    task automatic wait_responses(input integer target);
        begin
            timeout = 0;
            while (received < target) begin
                @(negedge clk);
                timeout = timeout + 1;
                if (timeout > 300000) $fatal(1, "UART response timeout target %0d", target);
            end
        end
    endtask

    initial begin
        $readmemh("../vectors/uart_requests.mem", request_bytes);
        $readmemh("../vectors/uart_responses.mem", expected_bytes);
        repeat (5) @(negedge clk);
        btn_c = 0;
        repeat (5) @(negedge clk);
        send_request(0);  // all zero
        send_request(1);  // busy packet is dropped, never alters the accepted frame
        wait_responses(57);
        repeat (6000) @(negedge clk);
        if (received != 57) $fatal(1, "busy request generated an unexpected response");
        send_request(2);  // real MNIST
        wait_responses(114);
        send_request(3);  // checksum error
        wait_responses(171);
        send_request(4);  // invalid timestep index, valid checksum
        wait_responses(228);
        btn_c = 1;
        repeat (5) @(negedge clk);
        btn_c = 0;
        repeat (5) @(negedge clk);
        send_request(5);  // real MNIST after reset
        wait_responses(285);
        send_request(6);  // consecutive frame with no reset
        wait_responses(342);
        for (i = 0; i < 10; i = i + 1) send_byte(request_bytes[i]);
        wait_responses(399);  // partial request times out with status 2
        $display("PASS tb_basys3_uart: 7 full and 1 partial RX packets, 7 exact TX packets");
        $finish;
    end
endmodule
