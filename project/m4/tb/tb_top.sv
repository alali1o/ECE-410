// ============================================================
// tb_top.sv
// End-to-End Testbench for kws_top
// ECE 410/510 — Spring 2026
// ============================================================
//
// COVERAGE:
//   1. Weight loading (direct parallel port, N_IN=8 × N_OUT=4 matrix)
//   2. Activation load (x_flat port)
//   3. SPI write to reg[1] (start command) — Mode 0, 16-bit transaction
//   4. SPI poll of reg[3] (status / done flag)
//   5. SPI read of reg[2] (argmax result)
//   6. Numeric verification: expected argmax = 2
//
// WEIGHT MATRIX (ternary {-1=2'b11, 0=2'b00, +1=2'b01}):
//   Row 0: [+1,+1,+1,+1, 0, 0, 0, 0]  → y[0] = x[0]+x[1]+x[2]+x[3]
//   Row 1: [-1,-1,-1,-1, 0, 0, 0, 0]  → y[1] = -(x[0]+...+x[3])
//   Row 2: [ 0, 0, 0, 0,+1,+1,+1,+1]  → y[2] = x[4]+x[5]+x[6]+x[7]
//   Row 3: [ 0, 0, 0, 0,-1,-1,-1,-1]  → y[3] = -(x[4]+...+x[7])
//
// ACTIVATIONS: x = {1,2,3,4,5,6,7,8} (all positive INT16)
//   y[0] = 1+2+3+4     =  10
//   y[1] = -(1+2+3+4)  = -10
//   y[2] = 5+6+7+8     =  26   ← maximum
//   y[3] = -(5+6+7+8)  = -26
//   Expected argmax = 2
//
// CLOCK / SPI TIMING:
//   System clock: 10 ns period (100 MHz)
//   SPI clock:   100 ns period (10 MHz) — 10× slower than sys clk
// ============================================================

`timescale 1ns/1ps

module tb_top;

    // ── Parameters ──────────────────────────────────────────
    localparam int N_IN  = 8;
    localparam int N_OUT = 4;
    localparam int X_W   = 16;
    localparam int ACC_W = 32;

    localparam int CLK_PERIOD  = 10;   // ns  (100 MHz sys clk)
    localparam int SCLK_HALF   = 50;   // ns  (10 MHz SPI)

    // ── DUT signals ─────────────────────────────────────────
    logic clk, rst_n;
    logic sclk, cs_n, mosi, miso;
    logic                       w_wr_en;
    logic [$clog2(N_OUT)-1:0]  w_row;
    logic [$clog2(N_IN) -1:0]  w_col;
    logic [1:0]                 w_din;
    logic [X_W*N_IN-1:0]       x_flat;

    // ── DUT instantiation ────────────────────────────────────
    kws_top #(
        .N_IN  (N_IN),
        .N_OUT (N_OUT),
        .X_W   (X_W),
        .ACC_W (ACC_W)
    ) dut (
        .clk     (clk),
        .rst_n   (rst_n),
        .sclk    (sclk),
        .cs_n    (cs_n),
        .mosi    (mosi),
        .miso    (miso),
        .w_wr_en (w_wr_en),
        .w_row   (w_row),
        .w_col   (w_col),
        .w_din   (w_din),
        .x_flat  (x_flat)
    );

    // ── Clock generation ─────────────────────────────────────
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ── SPI helpers ──────────────────────────────────────────
    // Send 16-bit SPI word MSB-first, Mode 0.
    // Format: {R/W#[15], ADDR[14:8], DATA[7:0]}
    //   R/W# = 0 → write, 1 → read
    task spi_transfer(
        input  logic [15:0] tx_word,
        output logic [7:0]  rx_byte   // data captured during byte 1
    );
        logic [15:0] shift_out;
        logic [7:0]  captured;
        int          b;

        cs_n       = 1'b0;
        sclk       = 1'b0;
        shift_out  = tx_word;
        captured   = 8'h00;

        #(SCLK_HALF);   // setup before first edge

        for (b = 15; b >= 0; b--) begin
            // Drive MOSI on falling edge (or start)
            mosi = shift_out[b];
            #(SCLK_HALF);
            sclk = 1'b1;    // rising edge — DUT samples MOSI
            if (b < 8)
                captured[b] = miso;   // capture MISO during byte 1
            #(SCLK_HALF);
            sclk = 1'b0;    // falling edge — DUT updates MISO
        end

        #(SCLK_HALF);
        cs_n = 1'b1;
        mosi = 1'b0;
        #(SCLK_HALF * 2);  // inter-transaction gap

        rx_byte = captured;
    endtask

    task spi_write(input logic [1:0] addr, input logic [7:0] data);
        logic [7:0] dummy;
        // R/W#=0 (write), addr in bits [14:8] (7-bit field, addr in [1:0])
        spi_transfer({1'b0, 5'b0, addr, data}, dummy);
    endtask

    task spi_read(input logic [1:0] addr, output logic [7:0] data);
        // R/W#=1 (read), addr in bits [14:8]
        spi_transfer({1'b1, 5'b0, addr, 8'hxx}, data);
    endtask

    // ── Weight loading helper ────────────────────────────────
    task load_weight(
        input logic [$clog2(N_OUT)-1:0] row,
        input logic [$clog2(N_IN) -1:0] col,
        input logic [1:0]               wt
    );
        @(negedge clk);
        w_wr_en = 1'b1;
        w_row   = row;
        w_col   = col;
        w_din   = wt;
        @(negedge clk);
        w_wr_en = 1'b0;
    endtask

    // ── VCD dump ─────────────────────────────────────────────
    initial begin
        $dumpfile("project/m4/sim/final_waveform.vcd");
        $dumpvars(0, tb_top);
    end

    // ── Main test sequence ───────────────────────────────────
    integer timeout_cnt;
    logic [7:0] spi_rx;
    logic [7:0] status_byte;
    logic [7:0] result_byte;
    integer pass_count;

    initial begin
        // --------------------------------------------------
        // 1. Initialise
        // --------------------------------------------------
        $display("=== KWS Top-Level Testbench START ===");
        rst_n   = 1'b0;
        sclk    = 1'b0;
        cs_n    = 1'b1;
        mosi    = 1'b0;
        w_wr_en = 1'b0;
        w_row   = '0;
        w_col   = '0;
        w_din   = 2'b00;
        x_flat  = '0;
        pass_count = 0;

        // Activate reset for 5 cycles
        repeat(5) @(negedge clk);
        rst_n = 1'b1;
        repeat(2) @(negedge clk);
        $display("[%0t ns] Reset released.", $time);

        // --------------------------------------------------
        // 2. Load weight matrix (4×8, ternary)
        //    2'b01 = +1,  2'b11 = -1,  2'b00 = 0
        // --------------------------------------------------
        $display("[%0t ns] Loading weights...", $time);

        // Row 0: [+1,+1,+1,+1, 0, 0, 0, 0]
        load_weight(0, 0, 2'b01); load_weight(0, 1, 2'b01);
        load_weight(0, 2, 2'b01); load_weight(0, 3, 2'b01);
        load_weight(0, 4, 2'b00); load_weight(0, 5, 2'b00);
        load_weight(0, 6, 2'b00); load_weight(0, 7, 2'b00);

        // Row 1: [-1,-1,-1,-1, 0, 0, 0, 0]
        load_weight(1, 0, 2'b11); load_weight(1, 1, 2'b11);
        load_weight(1, 2, 2'b11); load_weight(1, 3, 2'b11);
        load_weight(1, 4, 2'b00); load_weight(1, 5, 2'b00);
        load_weight(1, 6, 2'b00); load_weight(1, 7, 2'b00);

        // Row 2: [ 0, 0, 0, 0,+1,+1,+1,+1]
        load_weight(2, 0, 2'b00); load_weight(2, 1, 2'b00);
        load_weight(2, 2, 2'b00); load_weight(2, 3, 2'b00);
        load_weight(2, 4, 2'b01); load_weight(2, 5, 2'b01);
        load_weight(2, 6, 2'b01); load_weight(2, 7, 2'b01);

        // Row 3: [ 0, 0, 0, 0,-1,-1,-1,-1]
        load_weight(3, 0, 2'b00); load_weight(3, 1, 2'b00);
        load_weight(3, 2, 2'b00); load_weight(3, 3, 2'b00);
        load_weight(3, 4, 2'b11); load_weight(3, 5, 2'b11);
        load_weight(3, 6, 2'b11); load_weight(3, 7, 2'b11);

        $display("[%0t ns] Weights loaded.", $time);

        // --------------------------------------------------
        // 3. Set activations: x = [1,2,3,4,5,6,7,8] (INT16)
        //    x_flat is packed: element 0 at LSB
        // --------------------------------------------------
        x_flat = { 16'sd8, 16'sd7, 16'sd6, 16'sd5,
                   16'sd4, 16'sd3, 16'sd2, 16'sd1 };
        $display("[%0t ns] Activations set: x = [1,2,3,4,5,6,7,8]", $time);

        // --------------------------------------------------
        // 4. SPI write to reg[1] with data=0x01 → start
        // --------------------------------------------------
        repeat(2) @(negedge clk);
        $display("[%0t ns] Sending SPI start command (write reg[1] = 0x01)...", $time);
        spi_write(2'b01, 8'h01);
        $display("[%0t ns] Start command sent.", $time);

        // --------------------------------------------------
        // 5. Poll SPI reg[3] (status) until bit[0] = 1
        //    Timeout after 200 poll attempts
        // --------------------------------------------------
        $display("[%0t ns] Polling status register...", $time);
        status_byte = 8'h00;
        timeout_cnt = 0;
        while (!status_byte[0] && timeout_cnt < 200) begin
            spi_read(2'b11, status_byte);
            timeout_cnt++;
        end

        if (!status_byte[0]) begin
            $display("[FAIL] Timeout waiting for done flag. status=0x%02h", status_byte);
            $finish;
        end
        $display("[%0t ns] Done flag asserted after %0d polls.", $time, timeout_cnt);
        pass_count++;

        // --------------------------------------------------
        // 6. Read result from reg[2]
        // --------------------------------------------------
        spi_read(2'b10, result_byte);
        $display("[%0t ns] Argmax result from SPI reg[2] = %0d", $time, result_byte);

        // --------------------------------------------------
        // 7. Verify result
        // --------------------------------------------------
        if (result_byte == 8'd2) begin
            $display("[PASS] Argmax = 2 — correct. (y[2]=26 is maximum)");
            pass_count++;
        end else begin
            $display("[FAIL] Argmax = %0d — expected 2.", result_byte);
        end

        // --------------------------------------------------
        // 8. Summary
        // --------------------------------------------------
        $display("=== %0d/2 checks PASS ===", pass_count);
        if (pass_count == 2)
            $display("RESULT: PASS");
        else
            $display("RESULT: FAIL");

        #100;
        $finish;
    end

    // ── Watchdog ─────────────────────────────────────────────
    initial begin
        #5_000_000;  // 5 ms hard timeout
        $display("[ERROR] Watchdog timeout — simulation hung.");
        $finish;
    end

endmodule
