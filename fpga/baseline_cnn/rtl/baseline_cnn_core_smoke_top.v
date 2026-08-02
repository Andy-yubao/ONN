// baseline_cnn_core_smoke_top.v - Quartus smoke wrapper for the complete core.
//
// TOP_LEVEL_ENTITY of the baseline_cnn_core_smoke project.  Compiles the full
// compute core (stem + pool1 + shared conv2/conv3 engine + pool2 + GAP + FC +
// argmax + every parameter ROM) on the frozen target EP4CE10F17C8 to prove it
// fits and is clean (no latches / truncation / signed surprises, no division/
// modulo inferred, no complete stem_q/conv2_q/conv3_q RAM).
//
// Every port - including the smoke-only virtual clock `clk` - is assigned
// VIRTUAL_PIN in the QSF, so NO physical pin is consumed (AC620 pinout is not
// frozen).  All top-level ports are registered one stage so the Fitter never
// sees a register-less virtual pin (Error 171016).
//
// The large debug stream of the core is COMPRESSED at this boundary to keep the
// virtual-pin count low, per the frozen smoke policy:
//     dbg_stage     - controller stage
//     dbg_cnt       - stem_q pulses mod 256 (completion counter)
//     dbg_xor       - xor checksum of every conv_q value
//     dbg_last_addr - last conv_q address
//     dbg_last_val  - last conv_q value
//     prediction / done / busy
//
// `clk` / `rst_n` are TEST-ONLY virtual signals, not board clock/reset.  This
// wrapper is NOT the final board-level top.
`timescale 1ns/1ps

module baseline_cnn_core_smoke_top #(
    // Quartus resolves $readmemh relative paths against the project dir; the
    // project lives in fpga/baseline_cnn/quartus so "../params/..." is correct.
    parameter STEM_WT_MEM_FILE   = "../params/weights/stem_weight.mem",
    parameter STEM_BIAS_MEM_FILE = "../params/biases/stem_bias.mem",
    parameter CONV2_WT_MEM_FILE  = "../params/weights/conv2_weight.mem",
    parameter CONV2_BIAS_MEM_FILE= "../params/biases/conv2_bias.mem",
    parameter CONV3_WT_MEM_FILE  = "../params/weights/conv3_weight.mem",
    parameter CONV3_BIAS_MEM_FILE= "../params/biases/conv3_bias.mem",
    parameter FC_WT_MEM_FILE     = "../params/weights/fc_weight.mem",
    parameter FC_BIAS_MEM_FILE   = "../params/biases/fc_bias.mem"
) (
    input  wire        clk,          // virtual smoke clock, no physical pin
    input  wire        rst_n,        // virtual reset, no physical pin

    // control (registered)
    input  wire        start,
    output wire        busy,
    output wire        done,

    // input_q RAM write port (registered)
    input  wire        input_we,
    input  wire [9:0]  input_waddr,
    input  wire signed [7:0] input_wdata,

    // compressed debug observables (registered)
    output wire [3:0]  prediction,
    output wire [3:0]  dbg_stage,
    output wire [7:0]  dbg_cnt,       // stem_q pulses mod 256
    output wire [7:0]  dbg_xor,       // xor of conv_q values
    output wire [12:0] dbg_last_addr, // last conv_q address
    output wire [7:0]  dbg_last_val   // last conv_q value
);
    // ---- registered inputs ----
    reg        rst_n_r;
    reg        start_r;
    reg        input_we_r;
    reg [9:0]  input_waddr_r;
    reg signed [7:0] input_wdata_r;

    // ---- core connections ----
    wire busy_w, done_w;
    wire [3:0] prediction_w;
    wire [3:0] dbg_stage_w;
    wire stem_q_valid_w;
    wire conv_q_valid_w;
    wire [12:0] conv_q_addr_w;
    wire [7:0]  conv_q_value_w;

    baseline_cnn_core #(
        .STEM_WT_MEM_FILE   (STEM_WT_MEM_FILE),
        .STEM_BIAS_MEM_FILE (STEM_BIAS_MEM_FILE),
        .CONV2_WT_MEM_FILE  (CONV2_WT_MEM_FILE),
        .CONV2_BIAS_MEM_FILE(CONV2_BIAS_MEM_FILE),
        .CONV3_WT_MEM_FILE  (CONV3_WT_MEM_FILE),
        .CONV3_BIAS_MEM_FILE(CONV3_BIAS_MEM_FILE),
        .FC_WT_MEM_FILE     (FC_WT_MEM_FILE),
        .FC_BIAS_MEM_FILE   (FC_BIAS_MEM_FILE)
    ) u_core (
        .clk            (clk),
        .rst_n          (rst_n_r),
        .input_we       (input_we_r),
        .input_waddr    (input_waddr_r),
        .input_wdata    (input_wdata_r),
        .start          (start_r),
        .busy           (busy_w),
        .done           (done_w),
        .prediction     (prediction_w),
        .dbg_stage      (dbg_stage_w),
        .stem_acc_valid (),
        .stem_acc_addr  (),
        .stem_acc_value (),
        .stem_q_valid   (stem_q_valid_w),
        .stem_q_addr    (),
        .stem_q_value   (),
        .pool1_valid    (),
        .pool1_addr     (),
        .pool1_value    (),
        .conv_layer     (),
        .conv_acc_valid (),
        .conv_acc_addr  (),
        .conv_acc_value (),
        .conv_q_valid   (conv_q_valid_w),
        .conv_q_addr    (conv_q_addr_w),
        .conv_q_value   (conv_q_value_w),
        .pool2_valid    (),
        .pool2_addr     (),
        .pool2_value    (),
        .gap_valid      (),
        .gap_channel    (),
        .gap_value      (),
        .fc_valid       (),
        .fc_class       (),
        .fc_acc         ()
    );

    // ---- compressed observables (registered) ----
    reg [7:0]  dbg_cnt_r;
    reg [7:0]  dbg_xor_r;
    reg [12:0] dbg_last_addr_r;
    reg [7:0]  dbg_last_val_r;
    always @(posedge clk or negedge rst_n_r) begin
        if (!rst_n_r) begin
            dbg_cnt_r       <= 8'd0;
            dbg_xor_r       <= 8'd0;
            dbg_last_addr_r <= 13'd0;
            dbg_last_val_r  <= 8'd0;
        end else begin
            if (stem_q_valid_w) dbg_cnt_r <= dbg_cnt_r + 8'd1;
            if (conv_q_valid_w) begin
                dbg_xor_r       <= dbg_xor_r ^ conv_q_value_w;
                dbg_last_addr_r <= conv_q_addr_w;
                dbg_last_val_r  <= conv_q_value_w;
            end
        end
    end

    // ---- register boundary for every top-level port ----
    always @(posedge clk) begin
        rst_n_r      <= rst_n;
        start_r      <= start;
        input_we_r   <= input_we;
        input_waddr_r<= input_waddr;
        input_wdata_r<= input_wdata;
    end

    reg busy_r, done_r;
    reg [3:0] prediction_r;
    reg [3:0] dbg_stage_r;
    always @(posedge clk) begin
        busy_r       <= busy_w;
        done_r       <= done_w;
        prediction_r <= prediction_w;
        dbg_stage_r  <= dbg_stage_w;
    end

    assign busy          = busy_r;
    assign done          = done_r;
    assign prediction    = prediction_r;
    assign dbg_stage     = dbg_stage_r;
    assign dbg_cnt       = dbg_cnt_r;
    assign dbg_xor       = dbg_xor_r;
    assign dbg_last_addr = dbg_last_addr_r;
    assign dbg_last_val  = dbg_last_val_r;
endmodule
