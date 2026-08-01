// sync_rom_s32.v - Synthesizable synchronous ROM template (signed 32-bit data).
//
// Same contract as sync_rom_s8.v but for 32-bit words (used for the stem bias).
// Contents loaded once at time 0 with $readmemh from the `FILE` parameter
// (relative path; no absolute paths).  Synchronous read, one-cycle latency:
// `rdata` is valid one cycle after `addr` is presented.
//
// For synthesis Quartus uses the read loop + $readmemh initialisation to infer
// an M9K ROM.  If M9K is not inferred, record the reason in
// docs/rtl_microarchitecture.md instead of switching to vendor memory IP.
// Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module sync_rom_s32 #(
    parameter DEPTH  = 16,           // number of cells
    parameter ADDR_W = 4,            // address width
    parameter FILE   = "stem_bias.mem"
) (
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,   // read address (sampled at posedge)
    output reg signed [31:0] rdata   // read data (valid one cycle after addr)
);
    reg signed [31:0] mem [0:DEPTH-1];

    initial $readmemh(FILE, mem);

    always @(posedge clk) begin
        rdata <= mem[addr];          // synchronous read, one-cycle latency
    end
endmodule
