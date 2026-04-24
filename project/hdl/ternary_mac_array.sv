// ternary_mac_array.sv — Compute core for keyword-spotting ternary NN
// ECE 410/510 Spring 2026 — Codefest 4 COPT Part B / Milestone 2
//
// Project: Edge keyword spotting (10-class) on MCU via SPI
// Network: FC(40->256->128->10), ternary weights {-1, 0, +1}, INT8 activations
//
// This module implements a single dot-product engine:
//   out = SUM_i [ activation[i] * weight[i] ]   for i = 0..LENGTH-1
// where weight[i] in {-1, 0, +1} encoded as 2-bit signed.
//
// Interface: SPI-loaded activations and weights, result valid after LENGTH cycles.
//
// Parameters:
//   LENGTH    — dot-product length (max 256, set per layer)
//   ACT_W     — activation bit width (default 8, INT8)
//   ACC_W     — accumulator bit width (default 32, prevents overflow)
//
// Port directions follow the SPI slave model from M1:
//   - Controller loads activation and weight each cycle via act_in / wt_in
//   - valid_in pulses high for LENGTH cycles, then result appears on acc_out

`default_nettype none

module ternary_mac_array #(
    parameter int LENGTH = 256,   // dot-product vector length
    parameter int ACT_W  = 8,     // activation width (INT8)
    parameter int ACC_W  = 32     // accumulator width
) (
    input  logic                  clk,
    input  logic                  rst,        // active-high synchronous

    // Data inputs (one element per cycle)
    input  logic signed [ACT_W-1:0] act_in,  // INT8 activation
    input  logic signed [1:0]       wt_in,   // ternary weight: -1, 0, +1

    // Control
    input  logic                  valid_in,   // high for LENGTH cycles
    output logic                  valid_out,  // pulses when acc_out is ready

    // Result
    output logic signed [ACC_W-1:0] acc_out
);

    // ── Internal accumulator register ─────────────────────────────────────────
    logic signed [ACC_W-1:0] acc;

    // ── Ternary multiply: act * wt where wt in {-1, 0, +1} ──────────────────
    // No multiplier needed — just conditional negate or zero.
    logic signed [ACT_W-1:0] term;
    always_comb begin
        unique case (wt_in)
            2'b01:   term =  act_in;          // weight = +1
            2'b11:   term = -act_in;          // weight = -1 (2's complement)
            default: term = '0;               // weight =  0
        endcase
    end

    // ── Accumulate ────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (rst) begin
            acc       <= '0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            acc       <= acc + ACC_W'(signed'(term));
            valid_out <= 1'b0;
        end else begin
            // valid_in just went low — result is ready
            valid_out <= (acc != '0);   // suppress spurious pulse on idle
            acc       <= acc;           // hold
        end
    end

    assign acc_out = acc;

endmodule

`default_nettype wire
