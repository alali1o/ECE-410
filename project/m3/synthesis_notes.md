# Synthesis Notes — Milestone 3
ECE 410/510 Spring 2026 — Ternary KWS Inference Accelerator

## What Synthesized

Yosys 0.65 successfully elaborated and synthesized the complete `top` module with all submodules (`spi_slave_ext`, `compute_core`) flattened into a single netlist. The design elaborated cleanly with parameters N_IN=8, N_OUT=4, X_W=16, ACC_W=32. After optimization (`opt -full`) and technology mapping to generic internal cells, the design totals 2877 cells including 433 flip-flops and 2442 combinational cells. No elaboration errors and no unresolved hierarchies.

The `compute_core` module synthesized without issues. The sequential MAC FSM (IDLE/BUSY/DONE states), weight RAM (unpacked 2D array of 2-bit registers), accumulator array (4 × 32-bit signed registers), and pack/unpack generate blocks all mapped cleanly to Yosys internal cells. The 32-bit signed addition in the BUSY state mapped to an XOR/XNOR tree with carry propagation, which is the expected result for a generic technology library.

The `spi_slave_ext` module synthesized without issues. The 3-FF synchronizers for SCLK and CS_N, the 16-bit shift register, 5-bit bit counter, and 4-entry × 8-bit register file all elaborated correctly. The backdoor write port (cc_wr / cc_addr / cc_wdata) added for M3 integration mapped to a simple priority MUX in front of the register file write port, adding only a few cells.

The top-level glue logic — weight init FSM, 128-bit feature buffer, argmax (4-way combinational comparator), and 2-state result FSM — all synthesized correctly. The weight ROM (`initial` block with hardcoded values) was correctly inferred as constants by Yosys and either folded into logic or eliminated as don't-cares where outputs are not used downstream.

## What Did Not Synthesize (OpenLane 2 Failure)

The full OpenLane 2 RTL-to-GDSII flow failed before any EDA tools were invoked. The error occurs inside the OpenLane Python framework during PDK configuration loading:

```
_tkinter.TclError: too many nested evaluations (infinite loop?)
```

This error is in the OpenLane 2 Python package's Tcl interpreter bridge (`openlane/common/tcl.py`) and is unrelated to the RTL design. It appears to be a compatibility issue between OpenLane 2.3.10, Python 3.9 on macOS, and the version of the sky130 PDK that was auto-downloaded (`0fe599b2afb6708d281543108caf8310912f54af`). Specifically, the `Config.__get_pdk_raw` method tries to evaluate a PDK-provided Tcl environment script, and something in that script causes the Tcl interpreter to recurse infinitely.

This was not debugged further for M3 due to time. The workaround was to run Yosys directly (outside OpenLane) to complete synthesis, obtain cell counts, and perform manual critical path analysis. This provides synthesis results sufficient for M3 grading while the OpenLane environment issue is resolved for M4.

## Scope Adjustment

No scope change to the design itself. The RTL is complete and functionally verified (PASS in co-simulation). The scope adjustment for M3 is limited to the synthesis flow: instead of sky130-mapped OpenLane output, the synthesis report uses Yosys generic cells with estimated sky130 equivalents documented in `synth/area_report.txt`.

For M4, the plan is to:
1. Resolve the OpenLane 2 / Python 3.9 / Tcl compatibility issue (likely by using a Docker-based OpenLane 2 environment or upgrading to Python 3.11).
2. Re-run OpenLane with the same RTL and obtain proper timing, area, and power reports from OpenSTA.
3. Address the estimated timing violation on the SPI path by lowering the target clock to 50 MHz (see `synth/critical_path.md`).

The change to 50 MHz does not affect the M1 baseline comparison because the M1 benchmark measured software throughput (inference latency on CPU), not clock frequency. The accelerator's inference latency is determined by the compute_core pipeline (N_IN + 2 = 10 cycles for N_IN=8), which remains correct at any clock frequency.

## Modules and Their Status

| Module | File | Synthesis | Notes |
|---|---|---|---|
| `compute_core` | m2/rtl/compute_core.sv | ✓ Pass | No changes from M2 |
| `spi_slave_ext` | m3/rtl/spi_slave_ext.sv | ✓ Pass | M2 spi_slave + cc_wr backdoor port |
| `top` | m3/rtl/top.sv | ✓ Pass | New for M3; instantiates both modules + glue |

## Glue Logic Between Interface and Compute Core

Three pieces of glue logic bridge spi_slave_ext and compute_core:

**Weight init FSM.** After reset deasserts, a nested pair of counters (init_row, init_col) walks through all 32 hardcoded weights and pulses `w_wr_en` with the correct row, column, and encoding for each weight. This takes N_OUT × N_IN = 32 cycles. The SPI start command is gated by `weight_init_done` so the host cannot accidentally trigger inference before weights are loaded. In production, weights would come from a ROM or be loaded via SPI at boot time; the hardcoded approach is used here for deterministic simulation.

**Feature buffer.** The spi_slave_ext signals `bus_wr / bus_addr / bus_wdata` indicate when the SPI master has completed a write transaction. Each write to register address 0x00 appends one byte to a 128-bit shift register. After 16 bytes (8 inputs × 2 bytes each, little-endian INT16), the full input vector `x_flat` is assembled and held stable until the next inference.

**Result FSM.** When `compute_core.valid` pulses, a 2-state FSM writes the argmax index to reg[2] and 0x01 to reg[3] using the backdoor port. This makes both values immediately readable by the SPI master without requiring an extra SPI master write-back cycle.

## Warnings During Synthesis

Yosys reported 34 unique warnings. The most common:

- 22 instances of `"Wire ... is not used"` — from the `assign bus_rdata = reg_file[bus_addr]` statement in the original M2 spi_slave (the `bus_rdata` output was removed in spi_slave_ext and replaced by the backdoor write port). Yosys correctly eliminates these as unused.
- 8 instances of `"Replacing memory ... with FF"` — Yosys converted the weight `w_mem` array in compute_core from behavioral memory to flip-flops, which is correct for this register-file-sized array (32 entries × 2 bits = 64 bits).
- 4 instances related to the `initial` block for `w_rom` in top.sv — Yosys notes that initial blocks are not synthesizable but correctly folds the constant values into logic before synthesis.

None of the warnings indicate functional correctness issues.
