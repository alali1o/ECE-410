# M4 Remaining Tasks
ECE 410/510 Spring 2026 — Ternary KWS Inference Accelerator

Three highest-priority changes before Milestone 4:

---

## 1. Fix OpenLane 2 environment and complete sky130 synthesis

**Specific action:** Run OpenLane 2 on a Linux host (or Docker container) to bypass the Python 3.9 / Tcl interpreter crash that caused the M3 OpenLane failure on macOS. Use the existing `project/m3/synth/config.json` with the production parameters (N_IN=1960, N_OUT=512) and lower the clock period from 10 ns to 20 ns (50 MHz target) to give the SPI synchronizer chain enough timing margin. Commit the full OpenLane output: timing report (WNS, TNS), area report (cell count and µm² for the 512-accumulator MAC array), and power report from OpenSTA.

**Why:** The M3 synthesis used Yosys generic cells with estimated sky130 areas and timing. M4 requires verified numbers from an actual PDK-mapped synthesis. Without it, the area and power claims in M4 are unverifiable.

---

## 2. Replace register-per-byte SPI protocol with burst-mode streaming

**Specific action:** Modify `spi_slave_ext.sv` to hold CS_N low and stream all N_IN × 2 = 3,920 input bytes in a single burst SPI transaction instead of 1,960 separate 16-bit register write transactions. Change the feature buffer in `top.sv` to count raw SPI bytes directly (remove the register-address overhead). This removes ~50% of the SPI transfer overhead (the address byte per transaction), cutting feature-load time from ~6.27 ms to ~3.14 ms and pushing end-to-end throughput from ~130 to ~252 inferences/s at 10 MHz SPI.

**Why:** The current protocol (8-bit addr + 8-bit data per byte) was designed for register access, not streaming. Streaming mode is the correct protocol for bulk input loading and is standard in SPI sensor/ADC datasheets. This is the single change that most directly improves measured benchmark throughput for M4.

---

## 3. Run production-size cocotb simulation (N_IN=1960, N_OUT=512) and measure actual inference cycle count

**Specific action:** Write a cocotb testbench (or extend M3 `tb_top.sv`) that instantiates the full production compute_core with N_IN=1960, N_OUT=512 and a known weight matrix with a hand-computed golden output. Use `cocotb.start_soon` to stream the 1960 input bytes via the SPI task and time the full transaction (from first SCLK to valid pulse) in simulation cycles. Record the latency in a `project/m3/sim/production_cosim_run.log`. This converts the M4 throughput claim from "projected" to "measured."

**Why:** All M4 benchmark numbers currently use the projected path. The M4 rubric requires either a measured result or a very well-justified projection. A production-size simulation, even if slow to run, provides the measured latency and throughput that directly converts the roofline point from projected to measured.
