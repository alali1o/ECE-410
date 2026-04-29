// mac_correct.v — Corrected, synthesizable SystemVerilog MAC unit
// ECE 410/510 Spring 2026 — Codefest 4 CLLM
//
// Fixes applied vs LLM outputs:
//   1. Inputs declared 'logic signed [7:0]' — no manual sign extension needed.
//   2. All ports use SystemVerilog 'logic' (not Verilog 'wire'/'reg').
//   3. Uses always_ff as required; plain 'always' rejected.
//   4. Product explicitly cast to 32-bit signed before accumulation.

module mac (
    input  logic              clk,
    input  logic              rst,
    input  logic signed [7:0] a,
    input  logic signed [7:0] b,
    output logic signed [31:0] out
);
    always_ff @(posedge clk) begin
        if (rst)
            out <= '0;
        else
            out <= out + 32'(signed'(a * b));
            // a*b: signed [7:0]*signed [7:0] = signed [15:0]
            // signed'() preserves sign; 32'() zero/sign-extends to 32 bits correctly
    end
endmodule
