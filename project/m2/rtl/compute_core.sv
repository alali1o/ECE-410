// ============================================================
// compute_core.sv
// Ternary Matrix-Vector Multiply Accelerator
// ECE 410/510 — Spring 2026
// ============================================================
//
// TOP MODULE: compute_core
//
// PURPOSE:
//   Computes y = W * x where W is a ternary weight matrix
//   (weights in {-1, 0, +1}) and x is a signed INT16 input
//   vector. All multiplications reduce to conditional
//   add/subtract — no multiplier hardware required.
//
// PARAMETERS:
//   N_IN   : input vector length    (default 8;  production 1960)
//   N_OUT  : output vector length   (default 4;  production  512)
//   X_W    : input element width    (16 bits, signed INT16)
//   ACC_W  : accumulator width      (32 bits, signed INT32)
//
// PORTS:
//   clk      in  1     system clock
//   rst_n    in  1     synchronous active-low reset
//   w_wr_en  in  1     weight write enable (load one weight/cycle)
//   w_row    in  log2(N_OUT)  row address  (output neuron index)
//   w_col    in  log2(N_IN)   col address  (input feature index)
//   w_din    in  2     ternary weight: 2'b00=0, 2'b01=+1, 2'b11=-1
//   x_flat   in  X_W*N_IN   packed input vector (element 0 at LSB)
//   start    in  1     one-cycle pulse: begin MAC computation
//   y_flat   out ACC_W*N_OUT packed output vector (element 0 at LSB)
//   valid    out 1     one-cycle pulse: y_flat is ready
//
// CLOCK DOMAIN: single clock (clk)
// RESET:        synchronous, active-low (rst_n)
//
// OPERATION:
//   1. Load weights via w_wr_en / w_row / w_col / w_din.
//   2. Set x_flat, then pulse start for one cycle.
//   3. After N_IN cycles the FSM pulses valid; read y_flat.
//
// LATENCY: N_IN + 2 clock cycles from start to valid.
// ============================================================

`timescale 1ns/1ps

module compute_core #(
    parameter int N_IN  = 8,
    parameter int N_OUT = 4,
    parameter int X_W   = 16,
    parameter int ACC_W = 32
) (
    input  logic                        clk,
    input  logic                        rst_n,

    // Weight write port
    input  logic                        w_wr_en,
    input  logic [$clog2(N_OUT)-1:0]   w_row,
    input  logic [$clog2(N_IN) -1:0]   w_col,
    input  logic [1:0]                  w_din,

    // Inference port
    input  logic [X_W*N_IN-1:0]        x_flat,
    input  logic                        start,
    output logic [ACC_W*N_OUT-1:0]      y_flat,
    output logic                        valid
);

    // ----------------------------------------------------------
    // Weight memory: 2 bits per weight, N_OUT rows × N_IN cols
    // ----------------------------------------------------------
    logic [1:0] w_mem [0:N_OUT-1][0:N_IN-1];

    always_ff @(posedge clk) begin
        if (w_wr_en)
            w_mem[w_row][w_col] <= w_din;
    end

    // ----------------------------------------------------------
    // Unpack input vector into an array (avoids variable-base
    // bit-selects inside always_ff which some tools reject)
    // ----------------------------------------------------------
    logic signed [X_W-1:0] x_arr [0:N_IN-1];
    genvar gi;
    generate
        for (gi = 0; gi < N_IN; gi++) begin : g_unpack
            assign x_arr[gi] = x_flat[gi*X_W +: X_W];
        end
    endgenerate

    // ----------------------------------------------------------
    // Accumulators & state machine
    // ----------------------------------------------------------
    localparam [1:0] IDLE = 2'b00,
                     BUSY = 2'b01,
                     DONE = 2'b10;

    logic [1:0]                    state;
    logic [$clog2(N_IN)-1:0]      col_cnt;
    logic signed [ACC_W-1:0]      acc [0:N_OUT-1];

    // Sign-extended input element for current column
    logic signed [ACC_W-1:0] xi_ext;
    assign xi_ext = {{(ACC_W-X_W){x_arr[col_cnt][X_W-1]}}, x_arr[col_cnt]};

    integer j;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state   <= IDLE;
            col_cnt <= '0;
            valid   <= 1'b0;
            for (j = 0; j < N_OUT; j = j+1)
                acc[j] <= '0;
        end else begin
            valid <= 1'b0;
            case (state)

                IDLE: begin
                    if (start) begin
                        for (j = 0; j < N_OUT; j = j+1)
                            acc[j] <= '0;
                        col_cnt <= '0;
                        state   <= BUSY;
                    end
                end

                BUSY: begin
                    for (j = 0; j < N_OUT; j = j+1) begin
                        case (w_mem[j][col_cnt])
                            2'b01:   acc[j] <= acc[j] + xi_ext;  // +1
                            2'b11:   acc[j] <= acc[j] - xi_ext;  // -1
                            default: acc[j] <= acc[j];            //  0
                        endcase
                    end
                    if (col_cnt == N_IN - 1)
                        state <= DONE;
                    else
                        col_cnt <= col_cnt + 1'b1;
                end

                DONE: begin
                    valid <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    // ----------------------------------------------------------
    // Pack accumulator outputs
    // ----------------------------------------------------------
    genvar go;
    generate
        for (go = 0; go < N_OUT; go++) begin : g_pack
            assign y_flat[go*ACC_W +: ACC_W] = acc[go];
        end
    endgenerate

endmodule
