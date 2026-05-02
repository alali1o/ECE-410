# Milestone 2 — RTL, Testbenches, and Simulation
**Project:** Ternary KWS Inference Accelerator
**ECE 410/510 — Spring 2026**

---

## File Manifest

All paths are relative to the repository root (`ECE-410/`).

| File | Role |
|---|---|
| `project/m2/rtl/compute_core.sv` | Synthesizable SystemVerilog: ternary matrix-vector multiply core |
| `project/m2/rtl/interface.sv` | Synthesizable SystemVerilog: SPI slave (Mode 0) register interface |
| `project/m2/tb/tb_compute_core.sv` | Testbench: loads weights, applies input vector, checks outputs, prints PASS/FAIL |
| `project/m2/tb/tb_interface.sv` | Testbench: SPI write transaction, SPI read-back transaction, prints PASS/FAIL |
| `project/m2/sim/compute_core_run.log` | Icarus VVP transcript — must contain `PASS` |
| `project/m2/sim/interface_run.log` | Icarus VVP transcript — must contain `PASS` |
| `project/m2/sim/waveform.png` | Annotated waveform: compute pipeline (top) + SPI write transaction (bottom) |
| `project/m2/precision.md` | Numerical format choice (INT16 × ternary → INT32), error analysis, acceptability statement |
| `project/m2/README.md` | This file |

---

## How to Reproduce Both Simulations

### Requirements

| Tool | Version tested |
|---|---|
| Icarus Verilog | 13.0 (stable) |
| `iverilog` / `vvp` | shipped with Icarus 13.0 |

Install on macOS:
```bash
brew install icarus-verilog
```

### Simulate the compute core

```bash
# From the repository root
cd project/m2
iverilog -g2012 -o sim/sim_cc tb/tb_compute_core.sv rtl/compute_core.sv
vvp sim/sim_cc
```

Expected last lines:
```
PASS — all 4 outputs match golden reference
```

VCD waveform written to `sim/waveform.vcd`.

### Simulate the SPI interface

```bash
# From the repository root
cd project/m2
iverilog -g2012 -o sim/sim_if tb/tb_interface.sv rtl/interface.sv
vvp sim/sim_if
```

Expected last lines:
```
PASS — all SPI interface tests passed
```

---

## Deviations from the M2 Specification

### 1. Module name in `interface.sv`

**Specification:** "Top module name must match the filename."
**Deviation:** The file is `interface.sv` but the top module is named `spi_slave`.
**Reason:** `interface` is a **reserved keyword** in SystemVerilog (IEEE 1800-2017 §25). Using it as a module name causes a compile error in all SV-compliant tools including Icarus Verilog 13:

```
error: 'interface' is a keyword and may not be used as a module name
```

**Resolution:** The module is named `spi_slave` (a descriptive name matching its function) and lives in `interface.sv`. The testbench `tb_interface.sv` instantiates `spi_slave`. This is standard practice when the required filename conflicts with an SV reserved word.

### 2. Test parameters (N_IN=8, N_OUT=4) vs. production (N_IN=1960, N_OUT=512)

The `compute_core` module is **parametric**. The testbench uses small parameters (N_IN=8, N_OUT=4) for fast simulation with a fully hand-verified golden reference. The production parameters (N_IN=1960, N_OUT=512) are documented in the module header and require only changing the parameter values at instantiation. No logic changes are needed.

### 3. No deviations from M1 interface selection

The SPI interface chosen in M1 is implemented as specified: Mode 0 (CPOL=0, CPHA=0), 16-bit transactions (8-bit address + 8-bit data), 4-register file. No interface change.

---

## Design Notes

### compute_core.sv

- **Architecture:** Sequential MAC. One column of the weight matrix is processed per clock cycle. N_IN cycles after `start` is pulsed, `valid` pulses for one cycle and `y_flat` is stable.
- **Weight encoding:** 2'b00 = 0, 2'b01 = +1, 2'b11 = −1. Loaded via `w_wr_en / w_row / w_col / w_din`.
- **Latency:** N_IN + 2 clock cycles from `start` to `valid`.
- **Reset:** Synchronous, active-low (`rst_n`). All accumulators and state cleared.
- **Single clock domain:** One clock (`clk`). No clock crossings.

### interface.sv (module: spi_slave)

- **Protocol:** SPI Mode 0 (CPOL=0 — idle low; CPHA=0 — sample on rising SCLK).
- **Transaction format:** 16 bits per CS assertion. Byte 0: `{R/W#[7], ADDR[6:0]}`. Byte 1: data (write) or don't-care (read, MISO returns register value MSB-first).
- **Synchronisation:** SCLK and CS_N pass through 2-FF synchronisers. Requires system clock ≥ 4× SCLK.
- **Register map:**

  | Address | Name | Access | Description |
  |---|---|---|---|
  | 0x00 | `feature_byte` | RW | MFCC feature staging register |
  | 0x01 | `control` | RW | bit[0]: start inference |
  | 0x02 | `result` | RO | argmax label from compute core |
  | 0x03 | `status` | RO | bit[0]: inference done flag |

- **Reset:** Synchronous, active-low (`rst_n`). All registers and shift state cleared.

### Waveform (`sim/waveform.png`)

The PNG shows two sections:
- **Top 6 rows:** `compute_core` pipeline — clock, reset, start pulse, `col_cnt` counting 0→7, `valid` pulse, and `y[0]` stabilising at 12.
- **Bottom 2 rows:** `spi_slave` write transaction — SCLK with 16-bit burst (address byte 0x01 + data byte 0xA5), and CS_N framing the transaction.
