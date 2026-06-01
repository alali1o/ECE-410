// ============================================================
// interface.sv  (M4 revision — fixes bus_rdata direction)
// SPI Slave Interface Module
// ECE 410/510 — Spring 2026
// ============================================================
//
// CHANGES FROM M2:
//   1. bus_rdata changed from input to output (it was incorrectly
//      declared as input while being driven by internal logic).
//   2. Added ext_wr / ext_addr / ext_wdata ports so the top
//      module can write results and status back into the register
//      file (reg[2] = argmax result, reg[3] = done status).
//
// *** NAMING NOTE ***
// 'interface' is a reserved keyword in SystemVerilog.
// The top module in this file is therefore named spi_slave.
//
// TOP MODULE: spi_slave
//
// PURPOSE:
//   SPI slave implementing Mode 0 (CPOL=0, CPHA=0).
//   Provides a 4-entry × 8-bit register file accessible
//   from an SPI master. In the full chiplet this register
//   file bridges the MCU host to the compute_core:
//     reg[0] — feature byte staging  (reserved / future use)
//     reg[1] — control  (bit 0 = start compute)
//     reg[2] — result byte (argmax label, written by top)
//     reg[3] — status   (bit 0 = done, written by top)
//
// TRANSACTION FORMAT (16 bits per CS assertion):
//   Byte 0 (first 8 SCLKs) : { R/W#[7], ADDR[6:0] }
//     R/W# = 0 → write;  R/W# = 1 → read
//   Byte 1 (next  8 SCLKs) : data (write) / don't-care (read)
//   During byte 1 of a READ, MISO shifts out reg[ADDR] MSB-first.
//
// REGISTER MAP:
//   Addr 0x00  RW  reserved      — feature staging (unused in M4)
//   Addr 0x01  RW  control       — bit[0]: start inference pulse
//   Addr 0x02  RO  result        — inference argmax label
//   Addr 0x03  RO  status        — bit[0]: inference done flag
//
// PORTS:
//   clk        in  1   system clock (must be ≥ 4× SCLK)
//   rst_n      in  1   synchronous active-low reset
//   sclk       in  1   SPI clock from master (CPOL=0: idle low)
//   cs_n       in  1   chip select, active low
//   mosi       in  1   master→slave data, sampled on SCLK rising
//   miso       out 1   slave→master data, updated on SCLK falling
//   bus_addr   out 2   register address of last completed transaction
//   bus_wdata  out 8   write data of last completed transaction
//   bus_wr     out 1   one-cycle write strobe (system clock domain)
//   bus_rdata  out 8   current value of reg[bus_addr] (M4: now output)
//   ext_wr     in  1   external write enable (top → reg file)
//   ext_addr   in  2   external write address
//   ext_wdata  in  8   external write data
//
// CLOCK DOMAIN: single system clock (clk).
//   SCLK and CS_N are synchronised via 2-FF synchronisers.
//   Requires system clock ≥ 4× SCLK.
// RESET: synchronous, active-low (rst_n).
//
// SPI MODE: 0 (CPOL=0 — clock idles low;
//               CPHA=0 — data sampled on rising SCLK edge)
// ============================================================

`timescale 1ns/1ps

module spi_slave #(
    parameter int N_REGS = 4,   // number of 8-bit registers
    parameter int AW     = 2    // address bits = log2(N_REGS)
) (
    input  logic        clk,
    input  logic        rst_n,

    // SPI pins
    input  logic        sclk,
    input  logic        cs_n,
    input  logic        mosi,
    output logic        miso,

    // Internal register bus (to top-level glue)
    output logic [AW-1:0] bus_addr,
    output logic [7:0]    bus_wdata,
    output logic          bus_wr,
    output logic [7:0]    bus_rdata,   // M4 fix: was 'input'

    // External write-back (from top: write result/status into reg file)
    input  logic          ext_wr,
    input  logic [AW-1:0] ext_addr,
    input  logic [7:0]    ext_wdata
);

    // ----------------------------------------------------------
    // 2-FF synchronisers for sclk, cs_n, mosi
    // ----------------------------------------------------------
    logic sclk_s1, sclk_s2, sclk_s3;
    logic csn_s1,  csn_s2,  csn_s3;
    logic mosi_s1, mosi_s2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            {sclk_s1, sclk_s2, sclk_s3} <= 3'b0;
            {csn_s1,  csn_s2,  csn_s3}  <= 3'b111;
            {mosi_s1, mosi_s2}            <= 2'b0;
        end else begin
            sclk_s1 <= sclk;  sclk_s2 <= sclk_s1;  sclk_s3 <= sclk_s2;
            csn_s1  <= cs_n;  csn_s2  <= csn_s1;   csn_s3  <= csn_s2;
            mosi_s1 <= mosi;  mosi_s2 <= mosi_s1;
        end
    end

    // Edge detects (one sys-clock pulse each)
    wire sclk_rise = ( sclk_s2 & ~sclk_s3);
    wire sclk_fall = (~sclk_s2 &  sclk_s3);
    wire csn_fall  = (~csn_s2  &  csn_s3);
    wire csn_rise  = ( csn_s2  & ~csn_s3);

    // ----------------------------------------------------------
    // Register file
    // ----------------------------------------------------------
    logic [7:0] reg_file [0:N_REGS-1];

    // ----------------------------------------------------------
    // Shift register & bit counter
    // ----------------------------------------------------------
    logic [15:0] rx_shift;
    logic [7:0]  tx_shift;
    logic [4:0]  bit_cnt;
    logic        rw_bit;
    logic [AW-1:0] addr_latch;

    // ----------------------------------------------------------
    // Main SPI receive / transmit + external write-back logic
    // ----------------------------------------------------------
    integer i;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rx_shift   <= 16'h0;
            tx_shift   <= 8'h0;
            bit_cnt    <= 5'h0;
            rw_bit     <= 1'b0;
            addr_latch <= '0;
            miso       <= 1'b0;
            bus_wr     <= 1'b0;
            bus_addr   <= '0;
            bus_wdata  <= 8'h0;
            for (i = 0; i < N_REGS; i = i+1)
                reg_file[i] <= 8'h0;
        end else begin
            bus_wr <= 1'b0;

            // --- External write-back (top → result/status regs) ---
            if (ext_wr)
                reg_file[ext_addr] <= ext_wdata;

            // --- Transaction start ---
            if (csn_fall) begin
                bit_cnt  <= 5'h0;
                rx_shift <= 16'h0;
            end

            // --- Rising SCLK: sample MOSI ---
            if (sclk_rise && !csn_s2) begin
                rx_shift <= {rx_shift[14:0], mosi_s2};
                bit_cnt  <= bit_cnt + 1'b1;

                if (bit_cnt == 5'd7) begin
                    rw_bit     <= rx_shift[6];
                    addr_latch <= {rx_shift[AW-2:0], mosi_s2};
                    tx_shift   <= reg_file[{rx_shift[AW-2:0], mosi_s2}];
                end
            end

            // --- Falling SCLK: update MISO ---
            if (sclk_fall && !csn_s2) begin
                if (bit_cnt >= 5'd8) begin
                    miso     <= tx_shift[7];
                    tx_shift <= {tx_shift[6:0], 1'b0};
                end else begin
                    miso <= 1'b0;
                end
            end

            // --- Transaction end: commit write ---
            if (csn_rise) begin
                if (!rw_bit) begin
                    reg_file[addr_latch] <= rx_shift[7:0];
                    bus_addr             <= addr_latch;
                    bus_wdata            <= rx_shift[7:0];
                    bus_wr               <= 1'b1;
                end
            end
        end
    end

    // bus_rdata reflects current register value
    assign bus_rdata = reg_file[bus_addr];

endmodule
