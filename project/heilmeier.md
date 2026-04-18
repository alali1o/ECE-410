# Heilmeier Catechism
**ECE 410/510 Spring 2026**
**Project: Binary/Ternary Neural Network Inference Accelerator for Keyword Spotting**

---

## Q1. What are you trying to do?

I am building a custom hardware chiplet that recognizes spoken keywords (e.g., "yes," "no," "stop") directly on a battery-powered embedded device — without sending audio to the cloud or a general-purpose processor. The chiplet accelerates a ternary neural network (weights in {−1, 0, +1}) whose dominant kernel is a matrix-vector multiply: a fully-connected layer mapping 1960 MFCC input features to 512 hidden units (FC1). By replacing floating-point multiply-accumulate units with adder trees operating on {−1, 0, +1} weights, the chiplet eliminates all multipliers. It communicates with a host MCU over SPI and must classify one 1-second audio window in under 10 ms while consuming sub-milliwatt power.

**Algorithm:** 3-layer ternary fully-connected network (TernaryKWS).  
**Target kernel:** FC1 matrix-vector multiply, dimensions 1960 × 512, weights ∈ {−1, 0, +1}.

---

## Q2. How is it done today, and what are the limits of current practice?

**Current practice:**  
Keyword spotting on edge devices runs either (a) on a general-purpose ARM Cortex-M MCU using TensorFlow Lite Micro with INT8 quantized models, or (b) on dedicated always-on NPUs (Syntiant NDP, Arm Ethos-U55). Both approaches rely on hardware multipliers for the dominant MAC operation.

**Measured limits (this project, Apple M4, Python 3.9.6, PyTorch 2.8.0):**

| Metric | Measured value |
|---|---|
| Median inference latency (batch=1, 100 runs) | **0.436 ms** |
| Throughput | **2,293 inferences/sec** |
| Effective GFLOP/s (2.14M FLOPs / 0.436 ms) | **4.91 GFLOP/s** |
| Peak process RSS | **187 MB** |
| Model parameters (FP32 simulation) | **1,072,266 (4,096 KB)** |

**Profiling breakdown (50-run cProfile, dominant hotspots):**

| Function | Total time (50 runs) | % of runtime |
|---|---|---|
| `ternarize` (clamp + sign + mean + abs) | 31.0 ms | **75.6 %** |
| `torch._C._nn.linear` (GEMV) | 4.0 ms | **9.7 %** |
| `torch.batch_norm` | 3.0 ms | **7.3 %** |
| Other (ReLU, module dispatch) | 3.0 ms | 7.4 % |

The `ternarize` overhead is a software simulation artifact — in hardware the weights are pre-quantized and stored in SRAM, so 100% of silicon compute is the GEMV. The measured AI of FC1 (float32 DRAM baseline) is **0.50 FLOP/byte**, placing the kernel deep in the memory-bound region of the Apple M4 roofline (ridge point ≈ 33 FLOP/byte at 120 GB/s memory bandwidth). See `codefest/cf02/profiling/roofline_project.png`.

**Root cause:** The FC1 weight matrix (1960 × 512 × 4 bytes = 3.84 MB) must be streamed from DRAM on every inference. Even with ternary packing (2 bits/weight = 245 KB), the kernel remains memory-bound on a CPU without an on-chip cache large enough to hold the weights.

---

## Q3. What is new in your approach and why do you think it will be successful?

**Novel elements:**

1. **Ternary MAC datapath** — Replace each 32-bit FP multiply-accumulate with a conditional add/subtract on a {−1, 0, +1} weight. A ternary MAC requires one 2-bit weight register and one adder; a float32 MAC requires a 32-bit multiplier (~100 gates vs. ~3,000 gates). FC1 alone uses 1,003,520 weights — eliminating multipliers is the dominant area/power reduction.

2. **On-chip weight SRAM** — FC1 ternary weights (1960 × 512 × 2 bits = 245 KB) fit entirely in on-chip SRAM. This raises the effective arithmetic intensity from **0.50 FLOP/byte** (DRAM-bound) to **406 FLOP/byte** (SPI I/O only: input 1960 × 2 bytes + output 512 × 2 bytes = 4,944 bytes per inference), moving the kernel from memory-bound to **compute-bound** on the accelerator's own roofline.

3. **SPI slave interface** — A standard SPI slave register map allows any MCU host to stream MFCC frames in and read back the top-1 keyword label and confidence score without a custom driver. At a target operating rate of 100 inferences/second, the required interface bandwidth is 100 × (1960 + 512) × 2 bytes = **0.49 MB/s**, well within SPI's rated 10 Mb/s (1.25 MB/s) — the design is **not interface-bound**.

4. **OpenLane 2 synthesizability** — The entire datapath is described in synthesizable SystemVerilog. This enables area, timing, and power estimation in a standard-cell flow, making the speedup claim (versus the M4 SW baseline) verifiable in Milestone 4.

**Why it will succeed:**  
The key insight from the roofline analysis is that on-chip weight storage flips the kernel from memory-bound (AI = 0.5) to compute-bound (AI = 406). Because ternary ops eliminate multipliers, the compute can be implemented with far fewer gates than INT8, and the 245 KB weight SRAM is small enough to synthesize in a realistic cell area. Prior BNN/TNN tapeouts (FINN, BRein) and published accuracy results (>92% on Google Speech Commands 10-class) confirm both the hardware feasibility and model quality.
