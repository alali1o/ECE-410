// ============================================================
// top.sv — Integrated Top Module
// Ternary KWS Inference Accelerator — Milestone 3
// ECE 410/510 — Spring 2026
// ============================================================
//
// PORTS:
//   clk      in  1    System clock (100 MHz nominal)
//   rst_n    in  1    Synchronous active-low reset
//   sclk     in  1    SPI clock from MCU host (≤10 MHz)
//   cs_n     in  1    SPI chip select, active low
//   mosi     in  1    SPI master→slave data
//   miso     out 1    SPI slave→master data
//
// PARAMETERS:
//   N_IN     : input vector length   (8 for simulation, 1960 production)
//   N_OUT    : output vector length  (4 for simulation,  512 production)
//   X_W      : input element width   (16 bits, signed INT16)
//   ACC_W    : accumulator width     (32 bits)
//
// INTERNAL STRUCTURE:
//   spi_slave_ext  — SPI slave (Mode 0, 4-register file)
//   compute_core   — Ternary matrix-vector multiply engine
//   Glue logic     — Weight init FSM, feature buffer, argmax, status
//
// REGISTER MAP (SPI):
//   Addr 0x00  RW  feature_byte  — Write 16 bytes here to load x_flat
//                                  (little-endian, element 0 first)
//   Addr 0x01  RW  control       — bit[0]=1: start inference
//   Addr 0x02  RO  result        — argmax label (written by glue logic)
//   Addr 0x03  RO  status        — bit[0]=1: inference done
//
// OPERATION SEQUENCE (host side):
//   1. Write 16 bytes to reg[0] (2 bytes per INT16 input, 8 inputs).
//   2. Write 0x01 to reg[1] to start inference.
//   3. Poll reg[3] until bit[0]=1 (done).
//   4. Read reg[2] for the argmax result.
//
// WEIGHT INIT:
//   Weights are hardcoded (matching M2 golden test vector).
//   After reset deasserts, a small FSM auto-loads all N_OUT*N_IN
//   weights into compute_core before enabling SPI operation.
//   Latency: N_OUT*N_IN + 2 cycles after rst_n goes high.
//
// GLUE LOGIC:
//   Feature buffer: 128-bit shift register assembled byte-by-byte
//     from writes to SPI reg[0].
//   Start FSM: detects write of 0x01 to reg[1]; pulses compute_core.start.
//   Result FSM: on compute_core.valid, computes argmax of y_flat,
//     writes result to reg[2] and sets reg[3]=0x01 via cc_wr.
//   Status clear: write of 0x00 to reg[1] clears reg[3] and feat_cnt.
// ============================================================

`timescale 1ns/1ps

module top #(
    parameter int N_IN  = 8,
    parameter int N_OUT = 4,
    parameter int X_W   = 16,
    parameter int ACC_W = 32
) (
    input  logic clk,
    input  logic rst_n,
    // SPI pins
    input  logic sclk,
    input  logic cs_n,
    input  logic mosi,
    output logic miso
);

    // ----------------------------------------------------------
    // Hardcoded weights (M2 golden test vector, row-major)
    // Encoding: 2'b00=0, 2'b01=+1, 2'b11=-1
    // Row 0: [+1,-1,+1, 0,+1,-1, 0,+1]
    // Row 1: [ 0,+1, 0,+1,-1,+1,+1, 0]
    // Row 2: [-1, 0,-1,+1, 0,+1,-1,+1]
    // Row 3: [+1,+1, 0,-1,+1, 0,+1,-1]
    // ----------------------------------------------------------
    localparam int W_TOTAL = N_OUT * N_IN;  // 32 weights

    logic [1:0] w_rom [0:N_OUT-1][0:N_IN-1];

    initial begin
        // Row 0
        w_rom[0][0]=2'b01; w_rom[0][1]=2'b11; w_rom[0][2]=2'b01; w_rom[0][3]=2'b00;
        w_rom[0][4]=2'b01; w_rom[0][5]=2'b11; w_rom[0][6]=2'b00; w_rom[0][7]=2'b01;
        // Row 1
        w_rom[1][0]=2'b00; w_rom[1][1]=2'b01; w_rom[1][2]=2'b00; w_rom[1][3]=2'b01;
        w_rom[1][4]=2'b11; w_rom[1][5]=2'b01; w_rom[1][6]=2'b01; w_rom[1][7]=2'b00;
        // Row 2
        w_rom[2][0]=2'b11; w_rom[2][1]=2'b00; w_rom[2][2]=2'b11; w_rom[2][3]=2'b01;
        w_rom[2][4]=2'b00; w_rom[2][5]=2'b01; w_rom[2][6]=2'b11; w_rom[2][7]=2'b01;
        // Row 3
        w_rom[3][0]=2'b01; w_rom[3][1]=2'b01; w_rom[3][2]=2'b00; w_rom[3][3]=2'b11;
        w_rom[3][4]=2'b01; w_rom[3][5]=2'b00; w_rom[3][6]=2'b01; w_rom[3][7]=2'b11;
    end

    // ----------------------------------------------------------
    // Weight init FSM
    // ----------------------------------------------------------
    logic                        w_wr_en;
    logic [$clog2(N_OUT)-1:0]   w_row;
    logic [$clog2(N_IN) -1:0]   w_col;
    logic [1:0]                  w_din;
    logic                        weight_init_done;

    logic [$clog2(N_IN) -1:0]   init_col;
    logic [$clog2(N_OUT)-1:0]   init_row;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weight_init_done <= 1'b0;
            init_col <= '0;
            init_row <= '0;
            w_wr_en  <= 1'b0;
        end else if (!weight_init_done) begin
            w_wr_en  <= 1'b1;
            w_row    <= init_row;
            w_col    <= init_col;
            w_din    <= w_rom[init_row][init_col];
            if (init_col == N_IN - 1) begin
                init_col <= '0;
                if (init_row == N_OUT - 1) begin
                    weight_init_done <= 1'b1;
                    w_wr_en          <= 1'b0;
                end else
                    init_row <= init_row + 1'b1;
            end else
                init_col <= init_col + 1'b1;
        end else begin
            w_wr_en <= 1'b0;
        end
    end

    // ----------------------------------------------------------
    // SPI slave (extended)
    // ----------------------------------------------------------
    logic [1:0]  bus_addr;
    logic [7:0]  bus_wdata;
    logic        bus_wr;
    logic        cc_wr;
    logic [1:0]  cc_addr;
    logic [7:0]  cc_wdata;

    spi_slave_ext #(.N_REGS(4), .AW(2)) u_spi (
        .clk      (clk),
        .rst_n    (rst_n),
        .sclk     (sclk),
        .cs_n     (cs_n),
        .mosi     (mosi),
        .miso     (miso),
        .bus_addr (bus_addr),
        .bus_wdata(bus_wdata),
        .bus_wr   (bus_wr),
        .cc_wr    (cc_wr),
        .cc_addr  (cc_addr),
        .cc_wdata (cc_wdata)
    );

    // ----------------------------------------------------------
    // Feature buffer (assembles x_flat from byte writes to reg[0])
    // 16 bytes = 8 × INT16 = 128 bits
    // ----------------------------------------------------------
    localparam int FEAT_BYTES = (N_IN * X_W) / 8;  // 16

    logic [X_W*N_IN-1:0] feat_buf;
    logic [4:0]           feat_cnt;   // 0..16

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            feat_buf <= '0;
            feat_cnt <= '0;
        end else if (bus_wr && bus_addr == 2'h0) begin
            feat_buf[feat_cnt*8 +: 8] <= bus_wdata;
            if (feat_cnt < FEAT_BYTES)
                feat_cnt <= feat_cnt + 1'b1;
        end else if (bus_wr && bus_addr == 2'h1 && bus_wdata == 8'h00) begin
            // Clear on control write of 0 (reset for next inference)
            feat_cnt <= '0;
        end
    end

    // ----------------------------------------------------------
    // Compute core
    // ----------------------------------------------------------
    logic                    start_pulse;
    logic [X_W*N_IN-1:0]    x_flat;
    logic [ACC_W*N_OUT-1:0]  y_flat;
    logic                    valid;

    assign x_flat = feat_buf;

    compute_core #(
        .N_IN(N_IN), .N_OUT(N_OUT), .X_W(X_W), .ACC_W(ACC_W)
    ) u_core (
        .clk    (clk),
        .rst_n  (rst_n),
        .w_wr_en(w_wr_en),
        .w_row  (w_row),
        .w_col  (w_col),
        .w_din  (w_din),
        .x_flat (x_flat),
        .start  (start_pulse),
        .y_flat (y_flat),
        .valid  (valid)
    );

    // ----------------------------------------------------------
    // Start detection: write 0x01 to reg[1] → one-cycle pulse
    // Only fire if weight init is done
    // ----------------------------------------------------------
    always_ff @(posedge clk) begin
        start_pulse <= 1'b0;
        if (weight_init_done && bus_wr && bus_addr == 2'h1 && bus_wdata[0])
            start_pulse <= 1'b1;
    end

    // ----------------------------------------------------------
    // Argmax (N_OUT = 4, combinational)
    // ----------------------------------------------------------
    logic [7:0] argmax;
    logic signed [ACC_W-1:0] y0, y1, y2, y3;

    assign y0 = y_flat[0*ACC_W +: ACC_W];
    assign y1 = y_flat[1*ACC_W +: ACC_W];
    assign y2 = y_flat[2*ACC_W +: ACC_W];
    assign y3 = y_flat[3*ACC_W +: ACC_W];

    always_comb begin
        argmax = 8'd0;
        if ($signed(y1) > $signed(y0))  argmax = 8'd1;
        if ($signed(y2) > $signed(y_flat[argmax*ACC_W +: ACC_W])) argmax = 8'd2;
        if ($signed(y3) > $signed(y_flat[argmax*ACC_W +: ACC_W])) argmax = 8'd3;
    end

    // ----------------------------------------------------------
    // Result FSM: on valid, write argmax→reg[2], done→reg[3]
    // ----------------------------------------------------------
    localparam [1:0] RES_IDLE      = 2'd0,
                     RES_WR_RESULT = 2'd1,
                     RES_WR_STATUS = 2'd2;

    logic [1:0] res_state;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            res_state <= RES_IDLE;
            cc_wr     <= 1'b0;
            cc_addr   <= 2'h0;
            cc_wdata  <= 8'h0;
        end else begin
            cc_wr <= 1'b0;
            case (res_state)
                RES_IDLE: begin
                    if (valid) begin
                        cc_wr    <= 1'b1;
                        cc_addr  <= 2'h2;
                        cc_wdata <= argmax;
                        res_state <= RES_WR_STATUS;
                    end
                end
                RES_WR_STATUS: begin
                    cc_wr    <= 1'b1;
                    cc_addr  <= 2'h3;
                    cc_wdata <= 8'h01;
                    res_state <= RES_IDLE;
                end
                default: res_state <= RES_IDLE;
            endcase
        end
    end

endmodule
