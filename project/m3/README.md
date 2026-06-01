# Milestone 3 — Integration and Synthesis
**Project:** Ternary KWS Inference Accelerator
**ECE 410/510 — Spring 2026**

---

## File Manifest

All paths are relative to `project/m3/`.

| File | Description |
|---|---|
| `README.md` | This file — catalog and reproduction instructions |
| `rtl/top.sv` | Integrated top module: instantiates spi_slave_ext + compute_core with glue logic (weight init FSM, feature buffer, argmax, result FSM) |
| `rtl/spi_slave_ext.sv` | M2 spi_slave extended with cc_wr/cc_addr/cc_wdata backdoor write port for compute core result updates |
| `tb/tb_top.sv` | End-to-end co-simulation testbench: drives SPI only (no direct compute_core port access), writes 16 feature bytes, starts inference, polls status, reads argmax result |
| `sim/cosim_sim` | Compiled simulation binary (iverilog output) |
| `sim/cosim_run.log` | Co-simulation transcript — contains PASS line |
| `sim/cosim_waveform.vcd` | Raw VCD from co-simulation |
| `sim/cosim_waveform.png` | Annotated waveform PNG (3 regions: weight init, SPI writes, compute+result) |
| `sim/plot_waveform.py` | Python script that generated cosim_waveform.png from the VCD |
| `synth/config.json` | OpenLane 2 configuration (design name, clock period, source files, PDK) |
| `synth/synth.ys` | Yosys synthesis script used as fallback when OpenLane failed |
| `synth/openlane_run.log` | Full OpenLane stdout/stderr — shows failure + Yosys synthesis run |
| `synth/synth_out.v` | Yosys-synthesized netlist (generic cells) |
| `synth/area_report.txt` | Cell count and area estimate from Yosys stat output |
| `synth/timing_report.txt` | Timing analysis and critical path estimate (OpenSTA not available) |
| `synth/power_report.txt` | Power estimation (manual calculation; OpenLane did not complete) |
| `synth/critical_path.md` | Critical path identification: SPI synchronizer → addr_latch, ~16 ns, violates 10 ns target |
| `synthesis_notes.md` | Narrative: what synthesized, what failed, scope status, glue logic explanation |

---

## How to Reproduce Co-Simulation

**Simulator:** Icarus Verilog 13.0

**Install on macOS:**
```bash
brew install icarus-verilog
```

**Compile and run (from `project/m3/`):**
```bash
iverilog -g2012 -o sim/cosim_sim \
  tb/tb_top.sv rtl/top.sv rtl/spi_slave_ext.sv \
  ../m2/rtl/compute_core.sv

vvp sim/cosim_sim
```

**Expected output (last lines):**
```
PASS — end-to-end inference correct: argmax = 0
```

VCD is written to `sim/cosim_waveform.vcd`.

---

## How to Reproduce Synthesis

**OpenLane 2 version:** 2.3.10 (`pip show openlane`)

**Note:** OpenLane 2 failed on this machine due to a Python 3.9 / Tcl compatibility issue (see `synth/openlane_run.log`). Yosys 0.65 was used directly as a fallback.

**OpenLane (intended command, from `project/m3/synth/`):**
```bash
python3 -m openlane config.json
```

**Yosys fallback (from `project/m3/synth/`):**
```bash
yosys synth.ys
```

---

## Design Parameters

| Parameter | Simulation | Production |
|---|---|---|
| N_IN  | 8  | 1960 (full MFCC feature vector) |
| N_OUT | 4  | 512  (FC1 output neurons) |
| X_W   | 16 | 16   (INT16 activations) |
| ACC_W | 32 | 32   (INT32 accumulator) |

---

## Test Vector

Input: `x = [10, −5, 3, 7, −2, 8, 0, 4]` (INT16, matches M2 golden reference)

Expected output: `y = [12, 12, 6, −8]`, argmax = **0** (lowest index wins tie)

All 16 input bytes are written to reg[0] via SPI; start is issued to reg[1];
status is polled from reg[3]; result is read from reg[2].
