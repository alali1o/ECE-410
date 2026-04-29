# ECE 410/510 — Hardware for AI & ML (HW4AI)

**Portland State University · Spring 2026**  
Instructor: Prof. Christof Teuscher · Course: ECE 410/510

---

## Author

**Manaf Alali** — ECE Student, Portland State University

---

## About This Course

This course covers the **design, simulation, optimization, and evaluation** of specialized hardware for AI/ML workloads. It bridges the gap between algorithm theory and physical silicon — exactly where software meets the metal.

**Key areas include:**

- HW/SW co-design for CNNs, DNNs, and transformer-based LLMs
- Mapping algorithms onto GPUs, TPUs, FPGAs, systolic arrays, and neuromorphic processors
- Computational profiling with Python and CUDA
- In-memory computing with memristive crossbar arrays
- LLM-assisted HDL generation and physical design automation (RTL → GDSII)

---

## Project — Ternary NN Keyword Spotting Accelerator

**Goal:** Edge inference accelerator for a 3-layer FC ternary neural network (40→256→128→10) targeting keyword spotting on an MCU host over SPI.

### Compute Core (`project/hdl/ternary_mac_array.sv`)

Implements a parameterized dot-product engine for ternary weights {−1, 0, +1} and INT8 activations. Each cycle, one activation × weight pair is accumulated; after LENGTH cycles the result is valid.

- **Precision:** INT8 activations, 2-bit ternary weights (no multiplier — just conditional negate)
- **Accumulator:** 32-bit signed (prevents overflow across a 256-element dot product)
- **Interface:** SPI (exposed via future `spi_slave.sv` wrapper)

### Interface Choice — SPI

The dominant layer (FC1, 40×256) has an arithmetic intensity of **0.50 FLOP/byte** in software (M1 measurement). The bottleneck is DRAM bandwidth, not compute. SPI at 10 MHz delivers ~1.25 MB/s, which comfortably covers the 0.394 MB/s weight-transfer rate needed per inference. AXI would be over-engineered for an MCU host that has no AXI bus, and I²C is too slow (0.4 Mbit/s). SPI matches the MCU's native peripheral, keeps the interface simple, and leaves enough bandwidth margin (3.2×) for burst weight loading while the accelerator is computing.
