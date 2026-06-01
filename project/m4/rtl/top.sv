// ============================================================
// top.sv
// Ternary KWS Accelerator — Top-Level Integration
// ECE 410/510 — Spring 2026
// ============================================================
//
// TOP MODULE: kws_top
//
// PURPOSE:
//   Integrates compute_core (ternary MAC array) and spi_slave
//   (SPI Mode 0 interface) into a complete accelerator chiplet.
//   An SPI master (MCU) controls inference; weights are loaded
//   through a parallel port during device initialisation.
//
// ARCHITECTURE:
//
//   MCU (SPI master)
//        │  sclk / cs_n / mosi / miso
//        ▼
//   ┌─────────────┐   bus_wr / bus_addr / bus_wdata
//   │  spi_slave  │──────────────────────────────────► glue logic
//   │  (interface)│◄─── ext_wr / ext_addr / ext_wdata ─ (result/status)
//   └─────────────┘
//                                    start ▼      valid ▲
//   ┌─────────────────────────────────────────────────────┐
//   │               compute_core                          │
//   │  w_wr_en / w_row / w_col / w_din ◄── weight port   │
//   │  x_flat ◄─── activation port                       │
//   │  y_flat / valid ───► argmax ───► spi reg[2]        │
//   └─────────────────────────────────────────────────────┘
//
// SPI REGISTER MAP (16-bit transactions: {R/W#, ADDR[6:0], DATA[7:0]}):
//   Addr 0x01  W   Control   — bit[0] = 1 → start inference
//   Addr 0x02  R   Result    — argmax label (0 … N_OUT-1), written on valid
//   Addr 0x03  R   Status    — bit[0] = 1 → inference done
//
// PARAMETERS:
//   N_IN   : input vector length  (default 8; production 1960)
//   N_OUT  : output vector length (default 4; production  512)
//   X_W    : activation width     (16 bits, INT16)
//   ACC_W  : accumulator width    (32 bits, INT32)
//
// PORTS:
//   clk       in  1            system clock
//   rst_n     in  1            synchronous active-low reset
//   sclk      in  1            SPI clock (Mode 0, idle low)
//   cs_n      in  1            SPI chip-select (active low)
//   mosi      in  1            SPI master→slave data
//   miso      out 1            SPI slave→master data
//   w_wr_en   in  1            weight write enable
//   w_row     in  clog2(N_OUT) weight row address (output neuron)
//   w_col     in  clog2(N_IN)  weight column address (input feature)
//   w_din     in  2            ternary weight: 2'b00=0, 01=+1, 11=-1
//   x_flat    in  X_W*N_IN    packed activation vector (elem 0 at LSB)
//
// LATENCY: N_IN + 2 cycles from SPI start command to valid pulse.
// ============================================================

`timescale 1ns/1ps

module kws_top #(
    parameter int N_IN  = 8,
    parameter int N_OUT = 4,
    parameter int X_W   = 16,
    parameter int ACC_W = 32
) (
    input  logic clk,
    input  logic rst_n,

    // SPI interface (control path — from MCU host)
    input  logic sclk,
    input  logic cs_n,
    input  logic mosi,
    output logic miso,

    // Weight load interface (parallel — e.g. from MCU GPIO or DMA at boot)
    input  logic                       w_wr_en,
    input  logic [$clog2(N_OUT)-1:0]  w_row,
    input  logic [$clog2(N_IN) -1:0]  w_col,
    input  logic [1:0]                 w_din,

    // Activation input (parallel — loaded alongside SPI start command)
    input  logic [X_W*N_IN-1:0]       x_flat
);

    // ----------------------------------------------------------
    // Internal wires: SPI slave ↔ glue
    // ----------------------------------------------------------
    logic [1:0] bus_addr;
    logic [7:0] bus_wdata;
    logic       bus_wr;
    logic [7:0] bus_rdata;

    logic       ext_wr;
    logic [1:0] ext_addr;
    logic [7:0] ext_wdata;

    // ----------------------------------------------------------
    // SPI Slave instance
    // ----------------------------------------------------------
    spi_slave #(.N_REGS(4), .AW(2)) u_spi (
        .clk       (clk),
        .rst_n     (rst_n),
        .sclk      (sclk),
        .cs_n      (cs_n),
        .mosi      (mosi),
        .miso      (miso),
        .bus_addr  (bus_addr),
        .bus_wdata (bus_wdata),
        .bus_wr    (bus_wr),
        .bus_rdata (bus_rdata),
        .ext_wr    (ext_wr),
        .ext_addr  (ext_addr),
        .ext_wdata (ext_wdata)
    );

    // ----------------------------------------------------------
    // Compute core instance
    // ----------------------------------------------------------
    logic [ACC_W*N_OUT-1:0] y_flat;
    logic                   valid;
    logic                   start;

    compute_core #(
        .N_IN  (N_IN),
        .N_OUT (N_OUT),
        .X_W   (X_W),
        .ACC_W (ACC_W)
    ) u_core (
        .clk     (clk),
        .rst_n   (rst_n),
        .w_wr_en (w_wr_en),
        .w_row   (w_row),
        .w_col   (w_col),
        .w_din   (w_din),
        .x_flat  (x_flat),
        .start   (start),
        .y_flat  (y_flat),
        .valid   (valid)
    );

    // ----------------------------------------------------------
    // Glue: SPI write reg[1] bit[0] → start pulse
    // ----------------------------------------------------------
    assign start = bus_wr && (bus_addr == 2'b01) && bus_wdata[0];

    // ----------------------------------------------------------
    // Argmax: combinational tree over N_OUT accumulators
    // ----------------------------------------------------------
    logic [$clog2(N_OUT)-1:0] argmax_idx;

    // Argmax: use a generate-friendly approach
    logic [$clog2(N_OUT)-1:0] cmp_idx [0:N_OUT];
    logic signed [ACC_W-1:0]  cmp_val [0:N_OUT];

    assign cmp_idx[0] = '0;
    assign cmp_val[0] = $signed(y_flat[ACC_W-1:0]);  // y[0]

    genvar gk;
    generate
        for (gk = 1; gk < N_OUT; gk++) begin : g_argmax
            assign cmp_val[gk] = $signed(y_flat[gk*ACC_W +: ACC_W]);
            assign cmp_idx[gk] = ($signed(y_flat[gk*ACC_W +: ACC_W]) > cmp_val[gk-1])
                                  ? $clog2(N_OUT)'(gk) : cmp_idx[gk-1];
        end
    endgenerate

    assign argmax_idx = cmp_idx[N_OUT-1];

    // ----------------------------------------------------------
    // Done flag register (set on valid, cleared on next start)
    // ----------------------------------------------------------
    logic done_reg;
    always_ff @(posedge clk) begin
        if (!rst_n)
            done_reg <= 1'b0;
        else if (start)
            done_reg <= 1'b0;
        else if (valid)
            done_reg <= 1'b1;
    end

    // ----------------------------------------------------------
    // Write-back: on valid, push result (reg[2]) and status (reg[3])
    // into the SPI register file via ext_wr port.
    // Two-cycle sequence: cycle 0 → write reg[2], cycle 1 → write reg[3]
    // ----------------------------------------------------------
    logic valid_r;
    always_ff @(posedge clk) begin
        if (!rst_n) valid_r <= 1'b0;
        else         valid_r <= valid;
    end

    always_comb begin
        ext_wr    = 1'b0;
        ext_addr  = 2'b00;
        ext_wdata = 8'h00;

        if (valid) begin
            // Cycle of valid: write argmax to reg[2]
            ext_wr    = 1'b1;
            ext_addr  = 2'b10;
            ext_wdata = 8'(argmax_idx);
        end else if (valid_r) begin
            // Cycle after valid: write done flag to reg[3]
            ext_wr    = 1'b1;
            ext_addr  = 2'b11;
            ext_wdata = 8'h01;
        end
    end

endmodule
