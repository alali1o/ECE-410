// ============================================================
// spi_slave_ext.sv
// SPI Slave — Extended for M3 Integration
// ECE 410/510 — Spring 2026
// ============================================================
//
// This is the M2 spi_slave with one addition:
//   cc_wr / cc_addr / cc_wdata — a backdoor write port that
//   lets the compute core update reg[2] (result) and reg[3]
//   (status) after an inference completes.
//
// Glue logic rationale: the M2 spi_slave's internal register
// file could only be written by the SPI master. M3 requires
// the compute core to push results back through the interface;
// this port adds that capability without changing any existing
// SPI protocol logic.
//
// All other behavior is identical to M2 spi_slave.
// Module name: spi_slave_ext  (avoids reserved keyword 'interface')
// ============================================================

`timescale 1ns/1ps

module spi_slave_ext #(
    parameter int N_REGS = 4,
    parameter int AW     = 2
) (
    input  logic        clk,
    input  logic        rst_n,

    // SPI pins
    input  logic        sclk,
    input  logic        cs_n,
    input  logic        mosi,
    output logic        miso,

    // Internal register bus (write-notify — same as M2)
    output logic [AW-1:0] bus_addr,
    output logic [7:0]    bus_wdata,
    output logic          bus_wr,

    // Backdoor write port: compute core updates reg[2]/reg[3]
    input  logic          cc_wr,
    input  logic [AW-1:0] cc_addr,
    input  logic [7:0]    cc_wdata
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
    // Main SPI + backdoor write logic
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

            // Backdoor write from compute core (highest priority)
            if (cc_wr)
                reg_file[cc_addr] <= cc_wdata;

            // Transaction start
            if (csn_fall) begin
                bit_cnt  <= 5'h0;
                rx_shift <= 16'h0;
            end

            // Rising SCLK: sample MOSI
            if (sclk_rise && !csn_s2) begin
                rx_shift <= {rx_shift[14:0], mosi_s2};
                bit_cnt  <= bit_cnt + 1'b1;
                if (bit_cnt == 5'd7) begin
                    rw_bit     <= rx_shift[6];
                    addr_latch <= {rx_shift[AW-2:0], mosi_s2};
                    tx_shift   <= reg_file[{rx_shift[AW-2:0], mosi_s2}];
                end
            end

            // Falling SCLK: update MISO
            if (sclk_fall && !csn_s2) begin
                if (bit_cnt >= 5'd8) begin
                    miso     <= tx_shift[7];
                    tx_shift <= {tx_shift[6:0], 1'b0};
                end else begin
                    miso <= 1'b0;
                end
            end

            // Transaction end: commit SPI write
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

endmodule
