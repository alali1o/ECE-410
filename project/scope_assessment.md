# Project Scope Assessment
**ECE 410/510 Spring 2026 | Updated: CF07 (May 2026)**

## Project Summary

**Title:** Edge Keyword Spotting via Ternary Neural Network on MCU over SPI
**Core deliverable:** A hardware accelerator (`ternary_mac_array`) implementing dot-product computation for a three-layer fully connected network (FC 40→256→128→10) with ternary weights {−1, 0, +1} and INT8 activations, communicating with a host MCU over SPI.

## Scope Confirmation

The CF07 Yosys synthesis of `ternary_mac_array` produced **264 cells** with an estimated area of ~609 μm² at a 100 MHz clock target. This is a small, well-contained design. The scope **remains on track and unchanged** from the M1/M2 plan.

Key evidence:
- **Area headroom:** 264 cells per lane. Three lanes (one per FC layer) ≈ 800–1,000 cells total, well within reasonable ASIC/FPGA resource budgets.
- **Timing headroom:** Estimated +8.5 ns slack at 100 MHz. The ternary MAC is not the bottleneck — the SPI interface will dominate system throughput.
- **Functional risk identified:** The `valid_out` suppression bug (masked when accumulation result = 0) must be fixed before M3. This is a straightforward RTL patch, not a scope change.

## One Adjustment

The M2 plan assumed a single shared MAC core that would be time-multiplexed across layers. Based on the CF07 area result, instantiating three independent cores (one per layer, parameterized by LENGTH) is feasible within the same footprint and simplifies control logic. This is a minor architectural refinement, not a scope expansion.

## M3 Readiness

- RTL: Functional (needs `valid_out` fix)
- Synthesis: Yosys clean; sky130 OpenLane run targeted before M3 (May 24)
- Testbench: cocotb testbench exists from CF04; needs extension to cover zero-result case
- SPI integration: Not yet implemented — highest remaining risk item
