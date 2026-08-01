// sync_ram_u8.v - Synthesizable synchronous RAM template (8-bit data).
//
// One write port + one read port.  The READ is synchronous: `rdata` updates
// on the same posedge that `raddr` is sampled, so data is available one cycle
// after the address is presented (one-cycle read latency).  A consumer must
// NOT assume `rdata` is valid in the same cycle it asserts `raddr`.
//
// There is no reset on the memory contents (FPGA RAMs do not have a global
// reset), so uninitialized cells read X in simulation until written.  For the
// input RAM the write port is driven from outside (PC/UART path in the future);
// for the output RAM the write port is driven by the convolution engine.
//
// Quartus is expected to infer M9K blocks from this template (single-port
// write, single-port read, synchronous read).  If it instead maps to logic
// cells, record the reason in docs/rtl_microarchitecture.md rather than
// swapping to vendor memory IP.
// Verilog-2001, no vendor IP.
`timescale 1ns/1ps

module sync_ram_u8 #(
    parameter DEPTH  = 784,          // number of cells
    parameter ADDR_W = 10            // address width (>= ceil(log2(DEPTH)))
) (
    input  wire              clk,
    input  wire              we,     // write enable (1 = write at posedge)
    input  wire [ADDR_W-1:0] waddr,  // write address
    input  wire        [7:0] wdata,  // write data
    input  wire [ADDR_W-1:0] raddr,  // read address (sampled at posedge)
    output reg         [7:0] rdata   // read data (valid one cycle after raddr)
);
    reg [7:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[raddr];         // synchronous read, one-cycle latency
    end
endmodule
