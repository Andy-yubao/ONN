// tb_conv3_gap.v - Real conv3 + streaming GAP integration regression.
`timescale 1ns/1ps

module tb_conv3_gap;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst_n = 1'b0;
    reg start = 1'b0;
    wire conv_busy, conv_done;
    wire [11:0] fm_raddr;
    wire [7:0] fm_rdata;
    wire acc_valid, q_valid;
    wire [12:0] acc_addr, q_addr;
    wire signed [31:0] acc_value;
    wire [7:0] q_value;
    wire gap_busy, gap_done, gap_valid;
    wire [4:0] gap_channel;
    wire [7:0] gap_value;

    reg fm_we = 1'b0;
    reg [11:0] fm_waddr = 12'd0;
    reg [7:0] fm_wdata = 8'd0;
    sync_ram_u8 #(.DEPTH(1568), .ADDR_W(12)) u_fm_ram (
        .clk(clk), .we(fm_we), .waddr(fm_waddr), .wdata(fm_wdata),
        .raddr(fm_raddr), .rdata(fm_rdata)
    );

    conv_u8_serial u_conv (
        .clk(clk), .rst_n(rst_n), .start(start), .layer_sel(1'b1),
        .busy(conv_busy), .done(conv_done), .fm_raddr(fm_raddr),
        .fm_rdata(fm_rdata), .acc_valid(acc_valid), .acc_addr(acc_addr),
        .acc_value(acc_value), .q_valid(q_valid), .q_addr(q_addr),
        .q_value(q_value)
    );

    gap_stream_u8 u_gap (
        .clk(clk), .rst_n(rst_n), .start(start), .in_valid(q_valid),
        .in_q(q_value), .busy(gap_busy), .out_valid(gap_valid),
        .out_channel(gap_channel), .out_q(gap_value), .done(gap_done)
    );

    reg [7:0] pool2_q [0:1567];
    reg [31:0] conv3_acc [0:1567];
    reg [7:0] conv3_q [0:1567];
    reg [7:0] gap_q [0:31];

    integer cyc, start_cyc, first_q_cyc, last_q_cyc, last_gap_cyc, done_cyc;
    integer qcnt, gcnt, conv_done_count, gap_done_count;
    integer q_bad, q_addr_bad, gap_bad, gap_addr_bad, busy_bad;
    integer xz_bad, illegal_bad, rom_bad, post_bias_bad, emit_valid_bad;
    integer i;

    always @(posedge clk) begin
        #1;
        cyc = cyc + 1;
        if (conv_done) begin conv_done_count = conv_done_count + 1; done_cyc = cyc; end
        if (gap_done) gap_done_count = gap_done_count + 1;

        if (conv_busy && fm_raddr > 12'd1567) illegal_bad = illegal_bad + 1;
        if (conv_busy && u_conv.u_wt2_rom.addr !== 13'd0) rom_bad = rom_bad + 1;
        if (conv_busy && u_conv.u_wt3_rom.addr > 14'd9215) rom_bad = rom_bad + 1;
        if (q_valid && !gap_busy) busy_bad = busy_bad + 1;
        if ((u_conv.state == 4'd7) !== u_conv.req_out_valid) emit_valid_bad = emit_valid_bad + 1;
        if (q_valid !== ((u_conv.state == 4'd7) && u_conv.req_out_valid)) emit_valid_bad = emit_valid_bad + 1;
        if (u_conv.state == 4'd4 && u_conv.token_addr < 1568 &&
            u_conv.token_acc64 !== $signed(conv3_acc[u_conv.token_addr]))
            post_bias_bad = post_bias_bad + 1;

        if (q_valid) begin
            if (qcnt == 0) first_q_cyc = cyc;
            last_q_cyc = cyc;
            if (qcnt >= 1568 || acc_value !== $signed(conv3_acc[qcnt]) ||
                q_value !== conv3_q[qcnt]) q_bad = q_bad + 1;
            if (acc_addr !== qcnt[12:0] || q_addr !== qcnt[12:0]) q_addr_bad = q_addr_bad + 1;
            if (^acc_value === 1'bx || ^q_value === 1'bx) xz_bad = xz_bad + 1;
            qcnt = qcnt + 1;
        end
        if (gap_valid) begin
            last_gap_cyc = cyc;
            if (gcnt >= 32 || gap_value !== gap_q[gcnt]) gap_bad = gap_bad + 1;
            if (gap_channel !== gcnt[4:0]) gap_addr_bad = gap_addr_bad + 1;
            if (^gap_value === 1'bx || ^gap_channel === 1'bx) xz_bad = xz_bad + 1;
            gcnt = gcnt + 1;
        end
        if (conv_done !== gap_done) busy_bad = busy_bad + 1;
    end

    initial begin : watchdog
        repeat (1000000) @(posedge clk);
        $fatal(1, "C3GAP: TIMEOUT");
    end

    initial begin
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/pool2_q.mem", pool2_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_acc.mem", conv3_acc);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/conv3_q.mem", conv3_q);
        $readmemh("fpga/baseline_cnn/sim/vectors/golden_trace/gap_q.mem", gap_q);
        if (pool2_q[0] !== 8'h1F || conv3_acc[0] !== 32'hFFFFF7E6 ||
            conv3_q[4] !== 8'h09 || gap_q[31] !== 8'h07)
            $fatal(1, "C3GAP: golden vectors not loaded");

        cyc = 0; qcnt = 0; gcnt = 0; conv_done_count = 0; gap_done_count = 0;
        q_bad = 0; q_addr_bad = 0; gap_bad = 0; gap_addr_bad = 0; busy_bad = 0;
        xz_bad = 0; illegal_bad = 0; rom_bad = 0; post_bias_bad = 0; emit_valid_bad = 0;
        start_cyc = 0; first_q_cyc = 0; last_q_cyc = 0; last_gap_cyc = 0; done_cyc = 0;

        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);
        for (i = 0; i < 1568; i = i + 1) begin
            fm_we = 1'b1; fm_waddr = i[11:0]; fm_wdata = pool2_q[i];
            @(posedge clk);
        end
        fm_we = 1'b0;
        @(posedge clk);

        start = 1'b1;
        @(posedge clk);
        start_cyc = cyc + 1;
        start = 1'b0;
        while (!gap_done) @(posedge clk);
        #1;

        $display("C3GAP: q=%0d/1568 gap=%0d/32 first_q=%0d start->done=%0d last_q=%0d last_gap=%0d done=%0d",
                 qcnt, gcnt, first_q_cyc-start_cyc, done_cyc-start_cyc,
                 last_q_cyc, last_gap_cyc, done_cyc);
        $display("C3GAP: q_bad=%0d q_addr=%0d gap_bad=%0d gap_addr=%0d busy=%0d xz=%0d illegal=%0d rom=%0d post_bias=%0d emit_valid=%0d",
                 q_bad, q_addr_bad, gap_bad, gap_addr_bad, busy_bad, xz_bad,
                 illegal_bad, rom_bad, post_bias_bad, emit_valid_bad);
        if (qcnt != 1568 || gcnt != 32 || conv_done_count != 1 || gap_done_count != 1 ||
            q_bad != 0 || q_addr_bad != 0 || gap_bad != 0 || gap_addr_bad != 0 ||
            busy_bad != 0 || xz_bad != 0 || illegal_bad != 0 || rom_bad != 0 ||
            post_bias_bad != 0 || emit_valid_bad != 0 ||
            first_q_cyc-start_cyc != 293 || done_cyc-start_cyc != 460993 ||
            last_gap_cyc != last_q_cyc + 1 || done_cyc != last_gap_cyc + 1)
            $fatal(1, "C3GAP: FAILED");

        $display("C3GAP_ALL_PASS");
        $finish;
    end
endmodule
