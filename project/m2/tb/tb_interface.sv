// ============================================================
// tb_interface.sv — Testbench for spi_slave (interface.sv)
// ECE 410/510 — Spring 2026
// Simulator: Icarus Verilog 13.0
//   iverilog -g2012 -o sim_if tb/tb_interface.sv rtl/interface.sv
//   vvp sim_if
// ============================================================
//
// TESTS:
//   Test 1 — SPI Write:
//     Write 0xA5 to register address 0x01 (control register).
//     Byte 0: 0x01  (R/W#=0, addr=0x01)
//     Byte 1: 0xA5  (data)
//     Assert reg[0x01] == 0xA5 and bus_wr strobe fires.
//
//   Test 2 — SPI Read:
//     Read back register address 0x01.
//     Byte 0: 0x81  (R/W#=1, addr=0x01)
//     Byte 1: don't-care on MOSI; capture MISO[7:0].
//     Assert captured MISO byte == 0xA5.
//
// SPI MODE 0: CPOL=0 (clock idles low), CPHA=0 (sample rising)
// SCLK frequency = sys_clk / 8  (well within 4× sync requirement)
// ============================================================

`timescale 1ns/1ps

module tb_interface;

    // ----------------------------------------------------------
    // DUT signals
    // ----------------------------------------------------------
    logic       clk, rst_n;
    logic       sclk, cs_n, mosi;
    logic       miso;
    logic [1:0] bus_addr;
    logic [7:0] bus_wdata;
    logic       bus_wr;
    logic [7:0] bus_rdata;

    // ----------------------------------------------------------
    // DUT instantiation  (module is spi_slave, file is interface.sv)
    // ----------------------------------------------------------
    spi_slave #(
        .N_REGS(4),
        .AW    (2)
    ) dut (
        .clk      (clk      ),
        .rst_n    (rst_n    ),
        .sclk     (sclk     ),
        .cs_n     (cs_n     ),
        .mosi     (mosi     ),
        .miso     (miso     ),
        .bus_addr (bus_addr ),
        .bus_wdata(bus_wdata),
        .bus_wr   (bus_wr   ),
        .bus_rdata(bus_rdata)
    );

    // ----------------------------------------------------------
    // System clock: 10 ns period (100 MHz)
    // ----------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // ----------------------------------------------------------
    // VCD waveform dump (appended to waveform.vcd if run second;
    // tb_compute_core creates the file — this TB adds interface signals)
    // ----------------------------------------------------------
    initial begin
        $dumpfile("sim/waveform_if.vcd");
        $dumpvars(0, tb_interface);
    end

    // ----------------------------------------------------------
    // SPI master task: send 16-bit word (addr_byte + data_byte)
    // SCLK period = 8 sys-clock cycles → 80 ns
    // ----------------------------------------------------------
    task spi_transaction(
        input  logic [7:0] addr_byte,
        input  logic [7:0] data_byte,
        output logic [7:0] rx_byte
    );
        integer b;
        logic [15:0] tx_word;
        tx_word = {addr_byte, data_byte};
        rx_byte = 8'h00;

        // Assert CS_N
        @(negedge clk);
        cs_n = 1'b0;
        repeat(2) @(negedge clk);

        for (b = 15; b >= 0; b = b-1) begin
            // Setup MOSI before rising edge
            @(negedge clk);
            mosi = tx_word[b];
            sclk = 1'b0;
            repeat(3) @(negedge clk);

            // Rising edge — DUT samples MOSI
            sclk = 1'b1;
            @(negedge clk);

            // Capture MISO during data byte (bits 7..0)
            if (b < 8)
                rx_byte[b] = miso;

            repeat(3) @(negedge clk);
        end

        // Final falling edge + deassert
        sclk = 1'b0;
        repeat(2) @(negedge clk);
        cs_n = 1'b1;
        repeat(4) @(negedge clk);
    endtask

    // ----------------------------------------------------------
    // Main stimulus
    // ----------------------------------------------------------
    logic [7:0] captured;
    integer      fail_cnt;

    initial begin
        // Initialise
        rst_n = 1'b0;
        cs_n  = 1'b1;
        sclk  = 1'b0;
        mosi  = 1'b0;

        repeat(4) @(negedge clk);
        rst_n = 1'b1;
        repeat(2) @(negedge clk);

        fail_cnt = 0;

        // ========================================================
        // Test 1: Write 0xA5 to register 0x01
        // ========================================================
        $display("[TB] Test 1 — SPI Write: addr=0x01 data=0xA5");
        spi_transaction(8'h01, 8'hA5, captured);

        repeat(4) @(negedge clk);

        if (bus_wdata !== 8'hA5 || bus_addr !== 2'h1) begin
            $display("  FAIL — expected addr=0x01 wdata=0xA5, got addr=0x%0h wdata=0x%0h",
                     bus_addr, bus_wdata);
            fail_cnt = fail_cnt + 1;
        end else begin
            $display("  PASS — register write captured correctly (addr=0x%0h, data=0x%0h)",
                     bus_addr, bus_wdata);
        end

        // ========================================================
        // Test 2: Read back register 0x01 → expect 0xA5 on MISO
        // ========================================================
        $display("[TB] Test 2 — SPI Read:  addr=0x81 (R/W#=1, addr=0x01)");
        spi_transaction(8'h81, 8'h00, captured);

        if (captured !== 8'hA5) begin
            $display("  FAIL — expected MISO=0xA5, got 0x%0h", captured);
            fail_cnt = fail_cnt + 1;
        end else begin
            $display("  PASS — MISO returned 0x%0h as expected", captured);
        end

        // ========================================================
        // Summary
        // ========================================================
        $display("--------------------------------------------");
        if (fail_cnt == 0)
            $display("PASS — all SPI interface tests passed");
        else
            $display("FAIL — %0d test(s) failed", fail_cnt);
        $display("--------------------------------------------");

        #50;
        $finish;
    end

endmodule
