// ============================================================
// tb_top.sv — End-to-End Co-Simulation Testbench
// Ternary KWS Inference Accelerator — Milestone 3
// ECE 410/510 — Spring 2026
//
// Simulator: Icarus Verilog 13.0
//   From project/m3/:
//   iverilog -g2012 -o sim/cosim_sim \
//     tb/tb_top.sv rtl/top.sv rtl/spi_slave_ext.sv \
//     ../../m2/rtl/compute_core.sv
//   vvp sim/cosim_sim
//
// TEST VECTOR (matches M2 golden reference):
//   Weights: W[row][col] — same as M2 tb_compute_core.sv
//   Input: x = [10, -5, 3, 7, -2, 8, 0, 4] (INT16, little-endian over SPI)
//   Expected y = [12, 12, 6, -8]
//   Expected argmax = 0 (lowest index wins tie between y[0]=y[1]=12)
//
// PROTOCOL:
//   Host drives only the SPI pins (sclk, cs_n, mosi).
//   All feature writes, start command, status polling, and
//   result reads go through SPI register transactions.
//   No direct access to compute_core ports.
//
// SPI MODE 0: CPOL=0 (idle low), CPHA=0 (sample on rising edge)
// System clock: 100 MHz (10 ns period)
// SPI clock:    10 MHz  (100 ns period, ratio = 10:1 ≥ 4× required)
// ============================================================

`timescale 1ns/1ps

module tb_top;

    // Parameters matching top.sv
    localparam int N_IN  = 8;
    localparam int N_OUT = 4;
    localparam int X_W   = 16;
    localparam int ACC_W = 32;

    // DUT signals
    logic clk, rst_n;
    logic sclk, cs_n, mosi;
    logic miso;

    // DUT instantiation
    top #(.N_IN(N_IN), .N_OUT(N_OUT), .X_W(X_W), .ACC_W(ACC_W)) dut (
        .clk  (clk),
        .rst_n(rst_n),
        .sclk (sclk),
        .cs_n (cs_n),
        .mosi (mosi),
        .miso (miso)
    );

    // System clock: 100 MHz
    initial clk = 0;
    always #5 clk = ~clk;

    // SPI timing: SCLK period = 100 ns → half = 50 ns
    localparam int SCLK_HALF = 50;  // ns

    // VCD dump for waveform
    initial begin
        $dumpfile("sim/cosim_waveform.vcd");
        $dumpvars(0, tb_top);
    end

    // ----------------------------------------------------------
    // SPI write task: 16-bit transaction, SPI Mode 0
    //   addr: 7-bit address (bit[6] = 0 for write)
    //   data: 8-bit data byte
    // ----------------------------------------------------------
    task spi_write;
        input [6:0] addr;
        input [7:0] data;
        integer b;
        logic [7:0] addr_byte;
        begin
            addr_byte = {1'b0, addr};  // R/W# = 0 for write
            cs_n = 0;
            sclk = 0;
            #(SCLK_HALF);

            // Byte 0: address byte
            for (b = 7; b >= 0; b = b-1) begin
                mosi = addr_byte[b];
                #(SCLK_HALF);
                sclk = 1;
                #(SCLK_HALF);
                sclk = 0;
            end

            // Byte 1: data byte
            for (b = 7; b >= 0; b = b-1) begin
                mosi = data[b];
                #(SCLK_HALF);
                sclk = 1;
                #(SCLK_HALF);
                sclk = 0;
            end

            #(SCLK_HALF);
            cs_n = 1;
            mosi = 0;
            #(SCLK_HALF * 2);  // recovery time
        end
    endtask

    // ----------------------------------------------------------
    // SPI read task: returns 8-bit data from reg[addr]
    // ----------------------------------------------------------
    task spi_read;
        input  [6:0] addr;
        output [7:0] data;
        integer b;
        logic [7:0] addr_byte;
        logic [7:0] rx;
        begin
            addr_byte = {1'b1, addr};  // R/W# = 1 for read
            rx = 8'h0;
            cs_n = 0;
            sclk = 0;
            #(SCLK_HALF);

            // Byte 0: address byte (send)
            for (b = 7; b >= 0; b = b-1) begin
                mosi = addr_byte[b];
                #(SCLK_HALF);
                sclk = 1;
                #(SCLK_HALF);
                sclk = 0;
            end

            // Byte 1: dummy send, capture MISO (MSB first)
            mosi = 0;
            for (b = 7; b >= 0; b = b-1) begin
                #(SCLK_HALF);
                sclk = 1;
                rx[b] = miso;  // sample on rising SCLK
                #(SCLK_HALF);
                sclk = 0;
            end

            #(SCLK_HALF);
            cs_n = 1;
            mosi = 0;
            #(SCLK_HALF * 2);
            data = rx;
        end
    endtask

    // ----------------------------------------------------------
    // Test input: x = [10, -5, 3, 7, -2, 8, 0, 4] (INT16 LE)
    //   10  = 0x000A → bytes [0x0A, 0x00]
    //   -5  = 0xFFFB → bytes [0xFB, 0xFF]
    //    3  = 0x0003 → bytes [0x03, 0x00]
    //    7  = 0x0007 → bytes [0x07, 0x00]
    //   -2  = 0xFFFE → bytes [0xFE, 0xFF]
    //    8  = 0x0008 → bytes [0x08, 0x00]
    //    0  = 0x0000 → bytes [0x00, 0x00]
    //    4  = 0x0004 → bytes [0x04, 0x00]
    // Expected argmax = 0 (y = [12, 12, 6, -8]; y[0] wins tie)
    // ----------------------------------------------------------
    localparam int FEAT_BYTES = (N_IN * X_W) / 8;  // 16

    logic [7:0] feat_bytes [0:15];
    logic [7:0] rd_status, rd_result;
    integer     poll_cnt, errors;

    initial begin
        // Feature bytes: little-endian INT16 per element
        feat_bytes[0]=8'h0A; feat_bytes[1]=8'h00;   // 10
        feat_bytes[2]=8'hFB; feat_bytes[3]=8'hFF;   // -5
        feat_bytes[4]=8'h03; feat_bytes[5]=8'h00;   //  3
        feat_bytes[6]=8'h07; feat_bytes[7]=8'h00;   //  7
        feat_bytes[8]=8'hFE; feat_bytes[9]=8'hFF;   // -2
        feat_bytes[10]=8'h08;feat_bytes[11]=8'h00;  //  8
        feat_bytes[12]=8'h00;feat_bytes[13]=8'h00;  //  0
        feat_bytes[14]=8'h04;feat_bytes[15]=8'h00;  //  4
    end

    integer k;
    initial begin
        $display("=== M3 End-to-End Co-Simulation ===");
        $display("Input: x = [10, -5, 3, 7, -2, 8, 0, 4]");
        $display("Expected: y = [12, 12, 6, -8], argmax = 0");
        $display("---");

        errors    = 0;
        sclk      = 0;
        cs_n      = 1;
        mosi      = 0;
        rst_n     = 0;
        @(posedge clk); @(posedge clk); @(posedge clk);
        rst_n = 1;

        // Wait for weight init FSM: N_OUT*N_IN + 5 extra cycles
        repeat(N_OUT * N_IN + 10) @(posedge clk);
        $display("[TB] Weight init complete. Starting SPI transactions.");

        // ---- Phase 1: Load x_flat via 16 SPI writes to reg[0] ----
        $display("[TB] Writing %0d feature bytes to reg[0]...", FEAT_BYTES);
        for (k = 0; k < FEAT_BYTES; k = k+1) begin
            spi_write(7'h00, feat_bytes[k]);
        end
        $display("[TB] Feature load complete.");

        // ---- Phase 2: Write control reg (start = 1) ----
        $display("[TB] Writing start to reg[1]...");
        spi_write(7'h01, 8'h01);

        // ---- Phase 3: Poll status reg until done ----
        $display("[TB] Polling status reg[3]...");
        rd_status = 8'h00;
        poll_cnt  = 0;
        while (rd_status[0] !== 1'b1 && poll_cnt < 200) begin
            spi_read(7'h03, rd_status);
            poll_cnt = poll_cnt + 1;
        end

        if (rd_status[0] !== 1'b1) begin
            $display("FAIL — timeout waiting for status done");
            errors = errors + 1;
        end else begin
            $display("[TB] Status done after %0d polls. reg[3] = 0x%02X", poll_cnt, rd_status);
        end

        // ---- Phase 4: Read result ----
        spi_read(7'h02, rd_result);
        $display("[TB] reg[2] = result = %0d  (expected 0)", rd_result);

        if (rd_result !== 8'h00) begin
            $display("FAIL — argmax mismatch: got %0d, expected 0", rd_result);
            errors = errors + 1;
        end

        // ---- Final verdict ----
        $display("---");
        if (errors == 0)
            $display("PASS — end-to-end inference correct: argmax = %0d", rd_result);
        else
            $display("FAIL — %0d error(s)", errors);
        $display("=================================");

        #100;
        $finish;
    end

endmodule
