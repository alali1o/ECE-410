// ============================================================
// tb_compute_core.sv — Testbench for compute_core
// ECE 410/510 — Spring 2026
// Simulator: Icarus Verilog 13.0
//   iverilog -g2012 -o sim/sim_cc tb/tb_compute_core.sv rtl/compute_core.sv
//   vvp sim/sim_cc
// ============================================================
//
// TEST VECTOR (N_IN=8, N_OUT=4):
//   Weight encoding: 0->0, 1->+1, 3->-1
//     Row 0: [+1,-1,+1, 0,+1,-1, 0,+1]   enc: 1 3 1 0 1 3 0 1
//     Row 1: [ 0,+1, 0,+1,-1,+1,+1, 0]   enc: 0 1 0 1 3 1 1 0
//     Row 2: [-1, 0,-1,+1, 0,+1,-1,+1]   enc: 3 0 3 1 0 1 3 1
//     Row 3: [+1,+1, 0,-1,+1, 0,+1,-1]   enc: 1 1 0 3 1 0 1 3
//   Input:   x = [10, -5, 3, 7, -2, 8, 0, 4]
//   Golden (Python-verified):
//     y[0]=12  y[1]=12  y[2]=6  y[3]=-8
// ============================================================

`timescale 1ns/1ps

module tb_compute_core;

    localparam int N_IN  = 8;
    localparam int N_OUT = 4;
    localparam int X_W   = 16;
    localparam int ACC_W = 32;

    // DUT signals
    logic                      clk, rst_n;
    logic                      w_wr_en;
    logic [1:0]                w_row;  // log2(4)=2
    logic [2:0]                w_col;  // log2(8)=3
    logic [1:0]                w_din;
    logic [127:0]              x_flat; // 8 × 16 bits
    logic                      start;
    logic [127:0]              y_flat; // 4 × 32 bits
    logic                      valid;

    // DUT
    compute_core #(
        .N_IN(N_IN), .N_OUT(N_OUT), .X_W(X_W), .ACC_W(ACC_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .w_wr_en(w_wr_en), .w_row(w_row), .w_col(w_col), .w_din(w_din),
        .x_flat(x_flat), .start(start),
        .y_flat(y_flat), .valid(valid)
    );

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    // VCD dump
    initial begin
        $dumpfile("sim/waveform.vcd");
        $dumpvars(0, tb_compute_core);
    end

    // Weight encodings (row-major, col=0..7)
    // enc: 0=zero, 1=+1, 3=-1
    integer w_enc [0:3][0:7];
    integer x_val [0:7];
    integer y_gold[0:3];

    integer r, c, fail_cnt;
    logic signed [31:0] y_got;

    initial begin
        // ---- Golden reference (Python-verified) ---------------
        w_enc[0][0]=1; w_enc[0][1]=3; w_enc[0][2]=1; w_enc[0][3]=0;
        w_enc[0][4]=1; w_enc[0][5]=3; w_enc[0][6]=0; w_enc[0][7]=1;

        w_enc[1][0]=0; w_enc[1][1]=1; w_enc[1][2]=0; w_enc[1][3]=1;
        w_enc[1][4]=3; w_enc[1][5]=1; w_enc[1][6]=1; w_enc[1][7]=0;

        w_enc[2][0]=3; w_enc[2][1]=0; w_enc[2][2]=3; w_enc[2][3]=1;
        w_enc[2][4]=0; w_enc[2][5]=1; w_enc[2][6]=3; w_enc[2][7]=1;

        w_enc[3][0]=1; w_enc[3][1]=1; w_enc[3][2]=0; w_enc[3][3]=3;
        w_enc[3][4]=1; w_enc[3][5]=0; w_enc[3][6]=1; w_enc[3][7]=3;

        x_val[0]=10; x_val[1]=-5; x_val[2]=3; x_val[3]=7;
        x_val[4]=-2; x_val[5]=8;  x_val[6]=0; x_val[7]=4;

        y_gold[0]=12; y_gold[1]=12; y_gold[2]=6; y_gold[3]=-8;

        // ---- Reset -------------------------------------------
        rst_n=0; w_wr_en=0; start=0; x_flat=0;
        w_row=0; w_col=0; w_din=0;
        repeat(3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // ---- Load weights ------------------------------------
        $display("[TB] Loading %0d weights...", N_OUT*N_IN);
        for (r=0; r<N_OUT; r=r+1) begin
            for (c=0; c<N_IN; c=c+1) begin
                @(negedge clk);
                w_wr_en = 1'b1;
                w_row   = r[1:0];
                w_col   = c[2:0];
                w_din   = w_enc[r][c][1:0];
            end
        end
        @(negedge clk);
        w_wr_en = 1'b0;
        @(negedge clk);

        // ---- Pack input vector -------------------------------
        for (c=0; c<N_IN; c=c+1)
            x_flat[c*16 +: 16] = x_val[c][15:0];

        // ---- Pulse start ------------------------------------
        $display("[TB] Pulsing start...");
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // ---- Wait for valid ---------------------------------
        @(posedge valid);
        @(negedge clk);

        // ---- Compare outputs --------------------------------
        $display("[TB] Checking outputs:");
        fail_cnt = 0;
        for (r=0; r<N_OUT; r=r+1) begin
            y_got = y_flat[r*32 +: 32];
            if (y_got !== y_gold[r]) begin
                $display("  MISMATCH y[%0d]: got %0d, expected %0d", r, y_got, y_gold[r]);
                fail_cnt = fail_cnt + 1;
            end else
                $display("  y[%0d] = %4d   OK", r, y_got);
        end

        $display("--------------------------------------------");
        if (fail_cnt == 0)
            $display("PASS — all %0d outputs match golden reference", N_OUT);
        else
            $display("FAIL — %0d mismatch(es)", fail_cnt);
        $display("--------------------------------------------");
        #20; $finish;
    end

endmodule
