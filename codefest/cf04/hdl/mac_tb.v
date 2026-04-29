// mac_tb.v — Testbench for mac_correct.v
// ECE 410/510 Spring 2026 — Codefest 4 CLLM
//
// Sequence:
//   Cycles 1-3:  a=3,  b=4   → out = 12, 24, 36
//   Cycle  4:    rst=1       → out = 0
//   Cycles 5-6:  a=-5, b=2   → out = -10, -20
//
// Run: iverilog -g2012 -o mac_sim mac_tb.v mac_correct.v && ./mac_sim

`timescale 1ns/1ps

module mac_tb;
    logic        clk, rst;
    logic signed [7:0]  a, b;
    logic signed [31:0] out;

    // DUT
    mac dut (.clk(clk), .rst(rst), .a(a), .b(b), .out(out));

    // 10 ns clock
    initial clk = 0;
    always #5 clk = ~clk;

    // Task: check output on rising edge
    task check(input signed [31:0] expected, input string label);
        @(posedge clk); #1;
        if (out !== expected)
            $display("FAIL %s: got %0d, expected %0d", label, out, expected);
        else
            $display("PASS %s: out = %0d", label, out);
    endtask

    integer errors = 0;

    initial begin
        $dumpfile("mac_sim.vcd");
        $dumpvars(0, mac_tb);

        // Reset
        rst = 1; a = 0; b = 0;
        @(posedge clk); #1;
        rst = 0;

        // Phase 1: a=3, b=4 for 3 cycles → accumulates 12, 24, 36
        a = 3; b = 4;
        @(posedge clk); #1; $display("Cycle 1: out = %0d (expect 12)",  out);
        if (out !== 32'sd12)  errors++;
        @(posedge clk); #1; $display("Cycle 2: out = %0d (expect 24)",  out);
        if (out !== 32'sd24)  errors++;
        @(posedge clk); #1; $display("Cycle 3: out = %0d (expect 36)",  out);
        if (out !== 32'sd36)  errors++;

        // Phase 2: assert reset
        rst = 1; a = 0; b = 0;
        @(posedge clk); #1; $display("Cycle 4: out = %0d (expect 0 after rst)", out);
        if (out !== 32'sd0)   errors++;
        rst = 0;

        // Phase 3: a=-5, b=2 for 2 cycles → -10, -20
        a = -5; b = 2;
        @(posedge clk); #1; $display("Cycle 5: out = %0d (expect -10)", out);
        if (out !== -32'sd10) errors++;
        @(posedge clk); #1; $display("Cycle 6: out = %0d (expect -20)", out);
        if (out !== -32'sd20) errors++;

        if (errors == 0)
            $display("\n=== ALL TESTS PASSED ===");
        else
            $display("\n=== %0d TEST(S) FAILED ===", errors);

        $finish;
    end
endmodule
