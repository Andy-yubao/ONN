// sync_rom_s8.v - Synthesizable synchronous ROM template (signed 8-bit data).
//
// Contents are loaded once at time 0 with $readmemh from the file named by the
// `FILE` parameter.  The path is relative: simulation resolves it against the
// Questa working directory (repo root), synthesis against the Quartus working
// directory (project dir); the instantiating module picks the right default
// per environment.  No absolute paths are used anywhere.
//
// The read is synchronous (one-cycle latency): `rdata` is valid one cycle
// after `addr` is presented.  There is no reset and no write port (ROM).
//
// For synthesis the read loop and the $readmemh initialisation are what Quartus
// uses to infer an M9K ROM (or MLAB / logic cells on Cyclone IV E depending on
// the access pattern).  If M9K is not inferred, record the reason in
// docs/rtl_microarchitecture.md instead of switching to vendor memory IP.
// Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module sync_rom_s8 #(
    parameter DEPTH  = 144,          // number of cells
    parameter ADDR_W = 8,            // address width
    parameter FILE   = "stem_weight.mem"
) (
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,   // read address (sampled at posedge)
    output reg signed  [7:0] rdata   // read data (valid one cycle after addr)
);
    reg signed [7:0] mem [0:DEPTH-1];

    initial $readmemh(FILE, mem);

    always @(posedge clk) begin
        rdata <= mem[addr];          // synchronous read, one-cycle latency
    end
endmodule
