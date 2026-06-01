# Milestone 4 — Final Deliverables

**Project:** Ternary KWS Inference Accelerator  
**ECE 410/510 — Spring 2026 — Manaf Alali**  
**Due:** June 7, 2026

---

## Quick Navigation

| Section | File |
|---|---|
| Design justification report (PDF) | [report/design_justification.pdf](report/design_justification.pdf) |
| Final simulation log (PASS) | [sim/final_run.log](sim/final_run.log) |
| Benchmark summary | [bench/benchmark.md](bench/benchmark.md) |
| Roofline plot | [bench/roofline_final.png](bench/roofline_final.png) |
| Synthesis results | [synth/area_report.txt](synth/area_report.txt) |

---

## Deliverable File Catalog

### 1. README (this file)
| Path | Description | Checklist item |
|---|---|---|
| `project/m4/README.md` | This file — catalogs all M4 deliverables | §1 README files |

### 2. RTL Source Code
| Path | Description | Checklist item |
|---|---|---|
| `rtl/top.sv` | Top-level integration module (`kws_top`) connecting compute_core + spi_slave | §2 Source code |
| `rtl/compute_core.sv` | Ternary MAC array — same as M2, valid_out FSM bug fixed (see §9 of report) | §2 Source code |
| `rtl/interface.sv` | SPI slave (Mode 0) — M4 version: bus_rdata direction fixed, ext_wr ports added | §2 Source code |

**Changes from M3:** M4 adds `top.sv` (new), fixes `valid_out` bug in `compute_core.sv`, and corrects `bus_rdata` port direction in `interface.sv`. See Section 9 of the design justification report for details.

### 3. Testbench
| Path | Description | Checklist item |
|---|---|---|
| `tb/tb_top.sv` | End-to-end testbench for `kws_top` via SPI interface; 2/2 checks PASS | §2 Testbench |

### 4. Simulation Outputs
| Path | Description | Checklist item |
|---|---|---|
| `sim/final_run.log` | Icarus Verilog simulation log showing RESULT: PASS | §2 Sim log |
| `sim/final_waveform.png` | Annotated waveform showing all simulation phases | §2 Waveform |
| `sim/final_waveform.vcd` | Raw VCD dump from Icarus Verilog (source for PNG) | Supporting |

### 5. Synthesis Results
| Path | Description | Checklist item |
|---|---|---|
| `synth/config.json` | Yosys synthesis configuration (design, clock, source files) | §3 Config |
| `synth/openlane_run.log` | Full Yosys synthesis stdout log for `kws_top` | §3 Run log |
| `synth/timing_report.txt` | Critical path, slack, and clock closure analysis | §3 Timing |
| `synth/area_report.txt` | Cell counts, gate breakdown, and area estimate | §3 Area |
| `synth/power_report.txt` | Power methodology and estimate (manual — OpenLane unavailable) | §3 Power |
| `synth/synth_netlist.v` | Technology-mapped Verilog netlist from Yosys | Supporting |

**Synthesis tool:** Yosys 0.65 (technology-independent, sky130 area/timing estimated).  
OpenLane 2 was not available (Docker not installed). See power_report.txt for explanation.

### 6. Benchmark
| Path | Description | Checklist item |
|---|---|---|
| `bench/benchmark.md` | Throughput, speedup vs M1 SW baseline, energy comparison | §4 Benchmark |
| `bench/benchmark_data.csv` | Raw numbers underlying the benchmark summary | §4 Raw data |
| `bench/roofline_final.png` | M4 roofline with measured SW and HW operating points | §4 Roofline |

### 7. Design Justification Report
| Path | Description | Checklist item |
|---|---|---|
| `report/design_justification.pdf` | 9-section report (Problem, Roofline, Precision, Dataflow, Interface, Verification, Synthesis, Benchmark, What-did-not-work) | §5 Report |
| `report/figures/` | Figures referenced in the report | §5 Figures |

---

## Key Results Summary

| Metric | SW Baseline (M1) | HW Accelerator (M4) |
|---|---|---|
| Platform | Apple M4, PyTorch CPU | Simulation @ 100 MHz |
| FC1 latency | 409 μs | 19.6 μs |
| **FC1 speedup** | — | **20.9×** |
| Full network latency | 436 μs | 26 μs |
| **Full network speedup** | — | **16.8×** |
| Arithmetic intensity | 0.50 FLOP/byte | 406 FLOP/byte |
| Effective GFLOP/s (FC1) | 4.91 | 102.4 |
| Total cells (Yosys) | — | 3,728 |
| Estimated area | — | ~5,683 μm² |
| Estimated power | — | ~35 μW |
| Testbench result | — | **2/2 PASS** |
