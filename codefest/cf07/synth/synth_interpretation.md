# Synthesis Interpretation — ternary_mac_array
**Tool:** Yosys 0.65 (technology-independent synthesis, ABC backend)
**Note:** Full OpenLane 2 requires Docker (not available on this host); Yosys was run directly. Cell counts are real; timing and area are estimated from logic depth and sky130 typical values.

---

## Clock Period and Worst-Case Slack

Target clock period: **10 ns (100 MHz)**. The combinational critical path is estimated at ~1.5 ns, giving a **worst-case slack of +8.5 ns** with no violations. The design is nowhere near timing-limited at this frequency.

## Critical Path

The path runs: **act_in / wt_in inputs** → ternary weight decode (2–3 gate levels of NAND/NOR implementing the `unique case` on `wt_in[1:0]`) → **32-bit Brent-Kung carry-lookahead adder** (`acc <= acc + term`) → **acc[31:0]** register (SDFFE_PP0P, synchronous reset + clock enable). Yosys instantiated a Brent-Kung LCU during techmap, subsequently optimized by ABC into basic gates. The dominant cell types along the path are **NAND (76 instances)** for carry propagation and **XNOR (39 instances)** for sum bits.

## Total Cell Area and Top Contributors

Final netlist: **264 cells** — 231 combinational, 33 sequential (32 SDFFE_PP0P + 1 SDFF_PP0 for `valid_out`). Estimated sky130 area: **~609 μm²**. Top three contributors by count:

1. **NAND — 76 cells** (carry logic dominates)
2. **XNOR — 39 cells** (sum generation)
3. **AND / OR — 38 cells each** (mux selects and sign extension)

The 33 flip-flops are fewer in count but heavier per cell (~8 μm² vs ~1 μm² for NAND2), accounting for roughly 43% of total estimated area.

## Warnings and Constraint Notes

Yosys check pass: **0 problems**. One functional issue identified: `valid_out <= (acc != '0)` will silently suppress a valid output if the true dot product is exactly zero. This is a correctness bug to fix before M3, not a timing or synthesis concern.
