# CF06 CLLM — LLM-Assisted Crossbar MAC: Generation and Simulation
ECE 410/510 Spring 2026

---

## (a) Prompt Used

> "Generate a parameterized SystemVerilog module `crossbar_mac` that implements a binary-weight resistive crossbar MAC array. The module should compute out[j] = sum_i(weight[i][j] * in[i]) where each weight is either +1 or -1 (encoded as a single bit: 0 = +1, 1 = -1). Use N=4 inputs, M=4 outputs, 8-bit signed inputs, and a 32-bit signed accumulator. Include a clock, synchronous reset, and a valid_in/valid_out handshake."

LLM used: **Claude Sonnet 4.6**

---

## (b) Hand Calculation (Verification)

Weight matrix W (row i = input index, col j = output index):
```
W = [[+1, -1, +1, -1],   ← contributions of in[0]
     [+1, +1, -1, -1],   ← contributions of in[1]
     [-1, +1, +1, -1],   ← contributions of in[2]
     [-1, -1, -1, +1]]   ← contributions of in[3]
```

Input: `in = [10, 20, 30, 40]`

```
out[0] = +10 + 20 − 30 − 40 = −40
out[1] = −10 + 20 + 30 − 40 =   0
out[2] = +10 − 20 + 30 − 40 = −20
out[3] = −10 − 20 − 30 + 40 = −20
```

Expected output: **C = [−40, 0, −20, −20]**

---

## (c) Simulation Result

Simulated using iverilog 13.0:

```
=== CF06 Crossbar MAC Testbench ===
out[0] = -40  (expected -40)
out[1] = 0    (expected   0)
out[2] = -20  (expected -20)
out[3] = -20  (expected -20)
ALL PASS — C = [-40, 0, -20, -20]
```

Simulation matches hand calculation. ✓

---

## (d) Code Review — Issues Found

The initial LLM output used `automatic` variables inside an `always_ff` block, which is not supported by Icarus Verilog:

```
// Bug (iverilog does not support automatic lifetime override):
always_ff @(posedge clk) begin
    automatic logic signed [ACC_W-1:0] acc = '0;  // ERROR
    ...
end
```

**Fix:** Moved combinational accumulation into a separate `always_comb` block, and then registered the result in `always_ff`. This is cleaner RTL practice anyway — separating combinational logic from registered outputs.

Additionally, the 2D packed port `[N-1:0][M-1:0]` for weights caused iverilog to reject variable indexing inside loops. Switched to a flat 1D port `[N*M-1:0]` with manual bit slicing (`weight[i*M + j]`), which is fully compatible with iverilog.

**Summary of fixes:**
| Issue | LLM output | Fixed version |
|-------|-----------|---------------|
| `automatic` in always_ff | Caused iverilog error | Removed; use separate always_comb |
| 2D packed port indexing | Not supported in iverilog loops | Switched to flat 1D vector |
| always_ff vs always | Used always_ff (correct) | Kept as always for iverilog compat |

---

## (e) Files

| File | Description |
|------|-------------|
| `hdl/crossbar_mac.sv` | LLM-generated + fixed crossbar MAC module |
| `hdl/crossbar_tb.sv` | Testbench with hand-calculated expected values |
