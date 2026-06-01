# Milestone 3 — Synthesis

**Project:** Ternary KWS Inference Accelerator  
**ECE 410/510 — Spring 2026 — Manaf Alali**

---

## Summary

M3 performed technology-independent synthesis of the `compute_core` module
(the ternary MAC array) using Yosys 0.65. OpenLane 2 was not available on
the development machine (Docker not installed); Yosys was run directly
against the sky130 standard-cell library (technology-independent gates only).

Cell counts are exact from Yosys. Area and timing numbers are **estimated**
from published sky130 cell parameters and should be treated as order-of-
magnitude figures pending a full OpenLane 2 physical synthesis run.

---

## Synthesis Results (compute_core, N_IN=8, N_OUT=4)

| Metric | Value |
|---|---|
| Tool | Yosys 0.65 |
| Target frequency | 100 MHz (10 ns period) |
| Total cells (post-ABC) | **264** |
| Combinational cells | 231 |
| Sequential cells (FFs) | 33 |
| Estimated area | **~609 μm²** |
| Worst-case slack (est.) | **+8.5 ns** (timing MET) |
| Yosys check | **0 problems** |

---

## Files

| File | Description |
|---|---|
| `m3_plan.md` | M3 synthesis plan and identified bugs |
| `synth/synthesis_report.txt` | Full Yosys synthesis stdout log |
| `synth/synth_netlist.v` | Technology-mapped Verilog netlist |
| `synth/metrics.csv` | Structured synthesis metrics |
| `synth/synth_interpretation.md` | Human-readable analysis of results |

---

## Key Findings

1. **Timing**: Critical path (~1.5 ns) is dominated by the 32-bit
   Brent-Kung carry-lookahead adder in the accumulator update path.
   Design closes at 100 MHz with +8.5 ns estimated slack.

2. **Area**: 609 μm² estimated. FFs (33 × ~8 μm² = 264 μm²) account
   for ~43% of area. NAND cells (76) dominate combinational area.

3. **Bug identified**: `valid_out <= (acc != '0)` would suppress valid
   for a legitimate zero dot product. Fixed in M4 compute_core using
   cycle-counter FSM (IDLE → BUSY → DONE).

4. **OpenLane 2 TODO**: Docker installation required for accurate
   post-layout STA, real sky130 cell area, and switching-activity power.
