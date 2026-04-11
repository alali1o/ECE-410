# Heilmeier Catechism
**ECE 410/510 Spring 2026**
**Project: Binary/Ternary Neural Network Inference Accelerator for Keyword Spotting**
**Updated: Codefest 2 — informed by cProfile profiling and roofline analysis**

---

## Question 1: What are you trying to do?

I am building a custom hardware chiplet that runs keyword spotting (recognizing spoken words
like "yes", "no", or "stop") directly on a battery-powered edge device — without sending audio
to the cloud. The chip accelerates a small neural network that uses only -1, 0, and +1 as weight
values (ternary), replacing expensive multiplications with simple additions. The chiplet connects
to a microcontroller via SPI and must classify a 40-dimensional audio feature vector (MFCC frame)
in real time while consuming minimal power and area.

---

## Question 2: How is it done today, and what are the limits of current practice?

**How it is done today:**
- **MCU inference (e.g., TFLite Micro on ARM Cortex-M4):** Small INT8 or FP32 models run
  sequentially on the CPU. No spatial parallelism. The CPU handles everything from feature
  extraction to classification.
- **Commercial edge NPUs (e.g., Syntiant NDP, Arm Ethos-U55):** Purpose-built silicon with
  proprietary architectures. Efficient but closed-source, expensive for low-volume use, and
  not openly synthesizable.

**Limits — now grounded in profiling data:**
Profiling the software baseline (50 inference runs, Apple M4 CPU, NumPy FP32) reveals:
- Mean inference time: **7.0 µs per frame** (59.4 GFLOPS attainable)
- Dominant kernel: `ternary_linear` (np.dot) — 3 calls/inference, >90% of arithmetic
- Arithmetic intensity: **AI = 0.495 FLOP/byte** → firmly **memory-bound** on the M4 CPU
- The CPU achieves only **59.4 GFLOPS** out of a theoretical 230 GFLOPS peak — **74% of
  compute capacity is wasted** waiting for data from DRAM
- FP32 weight storage (177 KB) dominates data movement; only 1.7 KB are activations
- Standard INT8 quantization still requires hardware multipliers, which are area- and
  power-expensive in standard-cell silicon

---

## Question 3: What is new in your approach and why do you think it will be successful?

**What is new:**
I will design a fully synthesizable, open-source co-processor chiplet in SystemVerilog that:

1. **Stores weights as 2-bit ternary (INT8 in SW, 2-bit in HW)** — reducing weight memory
   from 177 KB (FP32) to 44 KB (INT8), raising arithmetic intensity from 0.495 to
   **1.92 FLOP/byte** (a 4× AI improvement).
2. **Executes ternary MAC operations using adder trees only** — no multipliers. Each weight
   in {-1, 0, +1} maps to a conditional add/subtract, dramatically reducing area and power.
3. **Keeps all weights in on-chip SRAM** (200 GB/s effective bandwidth vs. 120 GB/s DRAM),
   pushing the kernel from memory-bound into **compute-bound** on the accelerator roofline
   (attainable: 100 GFLOPS vs. 59.4 GFLOPS on CPU — a **×1.7 speedup on this kernel**).
4. **Exposes an SPI slave interface** — the 200 bytes/inference data requirement needs only
   0.16 Mbit/s, well within SPI's 50 Mbit/s capability, so the interface is never the
   bottleneck.
5. **Is synthesizable via OpenLane 2** — making the design reproducible and measurable.

**Why it will be successful:**
- The roofline analysis confirms a clear path: reducing precision from FP32 → INT8 alone
  shifts the operating point from deeply memory-bound to compute-bound, unlocking the full
  arithmetic throughput of a dedicated adder array.
- The kernel is small (88,576 FLOPs, 44 KB weights in INT8) — well-suited for a compact
  on-chip SRAM and a fixed-size systolic-style adder tree.
- The scope is deliberately narrow (3 FC layers, fixed dimensions, 1 SPI interface), making
  synthesis and functional verification achievable within the term timeline.
