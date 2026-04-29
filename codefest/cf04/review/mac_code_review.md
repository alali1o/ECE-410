# CF04 CLLM — MAC Code Review
ECE 410/510 Spring 2026

## LLM Attribution

| File | LLM | Model Version |
|------|-----|---------------|
| mac_llm_A.v | Claude | Claude Sonnet 4.6 |
| mac_llm_B.v | ChatGPT | GPT-4o (gpt-4o-2024-11-20) |

---

## Compilation Results

**mac_llm_A.v**
```
$ iverilog -g2012 -o /dev/null mac_llm_A.v
(no errors — compiles cleanly)
```

**mac_llm_B.v**
```
$ iverilog -g2012 -o /dev/null mac_llm_B.v
(no errors — compiles cleanly)
```

Both files compile without errors under iverilog 13.0. The issues are semantic/style, not syntax — which is why code review matters beyond just compilation.

---

## Simulation Log — mac_correct.v

```
$ iverilog -g2012 -o mac_sim mac_tb.v mac_correct.v && ./mac_sim

Cycle 1: out = 12  (expect 12)
Cycle 2: out = 24  (expect 24)
Cycle 3: out = 36  (expect 36)
Cycle 4: out = 0   (expect 0 after rst)
Cycle 5: out = -10 (expect -10)
Cycle 6: out = -20 (expect -20)

=== ALL TESTS PASSED ===
```

---

## Issue 1 — Missing `signed` on input ports (mac_llm_A.v)

**Offending lines:**
```systemverilog
input  logic [7:0]  a,   // line 9
input  logic [7:0]  b,   // line 10
```

**Why this is wrong:** Without the `signed` qualifier, `a` and `b` are treated as unsigned 8-bit values. The product `a * b` is then unsigned, and negative inputs like `a = -5` (stored as `0xFB`) would be interpreted as `+251`, producing wildly incorrect accumulator values. The LLM patched around this with a manual sign-extension workaround (`{{16{a[7]}}, a}`), which works but is verbose and fragile — any change in bit width requires updating the constant `16` in two places.

**Corrected version:**
```systemverilog
input  logic signed [7:0] a,
input  logic signed [7:0] b,
```
With `signed` ports, `a * b` is a signed 16-bit result and sign-extends cleanly to 32 bits with `32'(signed'(a * b))`.

---

## Issue 2 — `always` instead of `always_ff` (mac_llm_B.v)

**Offending line:**
```verilog
always @(posedge clk) begin   // line 16
```

**Why this is wrong:** The specification explicitly requires `always_ff`. The `always` construct is plain Verilog and does not enforce that only flip-flop-legal statements appear inside it. A synthesizer targeting an FPGA or ASIC may infer latches or combinational paths unintentionally. SystemVerilog's `always_ff` triggers a compile-time error if the body contains non-clocked logic, catching mistakes early. For a course on hardware for AI, where synthesis correctness matters, `always` is not acceptable.

**Corrected version:**
```systemverilog
always_ff @(posedge clk) begin
```

---

## Issue 3 — Verilog-style `wire`/`reg` instead of SystemVerilog `logic` (mac_llm_B.v)

**Offending lines:**
```verilog
input  wire                 clk,   // line 4
input  wire                 rst,   // line 5
input  wire signed [7:0]    a,     // line 6
input  wire signed [7:0]    b,     // line 7
output reg  signed [31:0]   out    // line 8
```

**Why this is wrong:** `wire` and `reg` are legacy Verilog-2001 types. `reg` does not mean "register" — it is a storage class that causes confusion and has led to countless bugs. SystemVerilog's `logic` type unifies both and works correctly in `always_ff` blocks. Mixing `wire`/`reg` with SystemVerilog constructs like `always_ff` is valid but non-idiomatic and rejected by some strict synthesis flows.

**Corrected version:**
```systemverilog
input  logic              clk,
input  logic              rst,
input  logic signed [7:0] a,
input  logic signed [7:0] b,
output logic signed [31:0] out
```

---

## mac_correct.v — Final Corrected Version

All three issues resolved. Compiles cleanly and passes the full testbench:

```systemverilog
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
    end
endmodule
```

`32'(signed'(a * b))`: `a * b` produces `signed [15:0]`; `signed'()` preserves the sign interpretation; `32'()` sign-extends to 32 bits before adding to the accumulator.
