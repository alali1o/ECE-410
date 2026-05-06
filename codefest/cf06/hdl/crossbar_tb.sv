// crossbar_tb.sv
// CF06 CLLM — Testbench for crossbar_mac
// ECE 410/510 Spring 2026
//
// Weight matrix (row i contributes to output j, +1=0 -1=1):
//   W = [[+1,-1,+1,-1],   row 0
//        [+1,+1,-1,-1],   row 1
//        [-1,+1,+1,-1],   row 2
//        [-1,-1,-1,+1]]   row 3
//
// Input: in = [10, 20, 30, 40]
// Expected output: [-40, 0, -20, -20]
//
// Hand calculation:
//   out[0] = +10+20-30-40 = -40
//   out[1] = -10+20+30-40 =   0
//   out[2] = +10-20+30-40 = -20
//   out[3] = -10-20-30+40 = -20

`timescale 1ns/1ps

module crossbar_tb;

    localparam N      = 4;
    localparam M      = 4;
    localparam DATA_W = 8;
    localparam ACC_W  = 32;

    reg  clk, rst, valid_in;

    // weight flat: bit [i*M+j] encodes weight[i][j], 0=+1, 1=-1
    // Row 0: j=0→+1(0), j=1→-1(1), j=2→+1(0), j=3→-1(1)  bits[3:0]  = 4'b1010
    // Row 1: j=0→+1(0), j=1→+1(0), j=2→-1(1), j=3→-1(1)  bits[7:4]  = 4'b1100
    // Row 2: j=0→-1(1), j=1→+1(0), j=2→+1(0), j=3→-1(1)  bits[11:8] = 4'b1001
    // Row 3: j=0→-1(1), j=1→-1(1), j=2→-1(1), j=3→+1(0)  bits[15:12]= 4'b0111
    reg [N*M-1:0] weight;

    // in_vec flat: bits[(i+1)*DATA_W-1 : i*DATA_W] = in[i]
    // in[0]=10, in[1]=20, in[2]=30, in[3]=40
    reg [N*DATA_W-1:0] in_vec;

    wire [M*ACC_W-1:0] out_vec;
    wire valid_out;

    // DUT
    crossbar_mac #(.N(N), .M(M), .DATA_W(DATA_W), .ACC_W(ACC_W)) dut (
        .clk      (clk),
        .rst      (rst),
        .weight   (weight),
        .in_vec   (in_vec),
        .valid_in (valid_in),
        .out_vec  (out_vec),
        .valid_out(valid_out)
    );

    // 10 ns clock
    initial clk = 0;
    always #5 clk = ~clk;

    // Signed readout helpers
    `define OUT(j) $signed(out_vec[(j)*ACC_W +: ACC_W])

    integer errors;

    initial begin
        $display("=== CF06 Crossbar MAC Testbench ===");
        errors = 0;

        // Weight encoding (see comment above)
        weight  = 16'b0111_1001_1100_1010;
        // in[0]=10, in[1]=20, in[2]=30, in[3]=40
        in_vec  = {8'd40, 8'd30, 8'd20, 8'd10};   // [3] at MSB, [0] at LSB
        // Wait — in_vec layout: bits[i*DATA_W +: DATA_W] = in[i]
        // So in_vec = {in[3], in[2], in[1], in[0]} = {40,30,20,10}
        valid_in = 0;
        rst = 1;
        @(posedge clk); #1;
        rst = 0;

        // Apply inputs for one cycle
        valid_in = 1;
        @(posedge clk); #1;
        valid_in = 0;

        // Outputs are registered — available this cycle
        $display("out[0] = %0d  (expected -40)", `OUT(0));
        $display("out[1] = %0d  (expected   0)", `OUT(1));
        $display("out[2] = %0d  (expected -20)", `OUT(2));
        $display("out[3] = %0d  (expected -20)", `OUT(3));

        if (`OUT(0) !== -40) begin errors = errors+1; $display("FAIL out[0]: got %0d", `OUT(0)); end
        if (`OUT(1) !==   0) begin errors = errors+1; $display("FAIL out[1]: got %0d", `OUT(1)); end
        if (`OUT(2) !== -20) begin errors = errors+1; $display("FAIL out[2]: got %0d", `OUT(2)); end
        if (`OUT(3) !== -20) begin errors = errors+1; $display("FAIL out[3]: got %0d", `OUT(3)); end

        if (errors == 0)
            $display("ALL PASS — C = [-40, 0, -20, -20]");
        else
            $display("%0d FAILURE(s)", errors);

        $finish;
    end

endmodule
