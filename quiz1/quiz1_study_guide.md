# ECE 410/510 — Quiz 1 Study Guide
**Bao Nguyen | Portland State University | Spring 2026**

---

## How to Use This Guide
Work through each section in order. For each concept: read the explanation, close the guide, and try to reproduce the key formula or definition from memory. Then check. For CMAN problems, always work through the full calculation — partial credit depends on showing formulas before numbers.

---

## Unit 1: Why Hardware for AI?

### The Core Problem
Modern AI workloads are **memory-bound**, not compute-bound. Processors got faster much quicker than memory got wider. The gap between what a chip can compute and how fast it can be fed data has grown for 30+ years — this is the **Memory Wall**.

### Energy Is the Real Constraint
Moving data costs far more than computing with it:
- FP32 multiply: ~**3.7 pJ**
- DRAM 64-bit read: ~**640 pJ** → **170× more expensive**

This means the primary goal of hardware design for AI is to **minimize data movement**, not just maximize compute throughput.

### Architecture Evolution
```
CPU → CPU + GPU → Heterogeneous → Extreme Heterogeneity
                                  (NPU, TPU, FPGA, ASIC, VPU)
```

### HW/SW Codesign
Design the algorithm and hardware **together**. Don't optimize software first, then bolt on hardware. The best systems co-optimize both simultaneously.

---

## Unit 2: Performance Analysis — Roofline Model

### Arithmetic Intensity (AI)
```
AI = FLOPs / Bytes    [FLOP/byte]
```
Measures how compute-heavy a kernel is relative to its memory traffic. High AI = compute-bound candidate.

### Roofline Model
```
Attainable Performance = min(Peak_Compute, AI × Bandwidth)
```

**Ridge Point** — the boundary between memory-bound and compute-bound:
```
I* = Peak_Compute / Bandwidth    [FLOP/byte]
```

| If AI < I* | Memory-bound | attainable = AI × BW |
|---|---|---|
| If AI > I* | Compute-bound | attainable = Peak Compute |

**Example (H100):** Peak = 67 TFLOPS, BW = 3.35 TB/s → I* = 20 FLOP/byte

### Hardware Knobs vs Algorithm Knobs
- **Higher BW** → lower ridge point → more kernels become compute-bound
- **Higher Peak Compute** → higher ridge point → harder to be compute-bound
- **Higher AI** (via tiling/reuse) → operating point moves right → toward compute-bound
- You cannot change AI by changing hardware — AI is a property of the algorithm

---

## Unit 3: GEMM and Tiling — The Most Important CMAN Topic

### What is N?
N is the **dimension of the square matrix**. If N=32, you have a 32×32 matrix:
- A is N×N, B is N×N, C (output) is N×N
- Total elements in one matrix = **N²**
- Total multiply-adds to compute C = **N³** (each of the N² output elements needs N multiply-adds)

### Naive GEMM
Triple loop, no data reuse. Every access goes to DRAM.
```
Traffic_naive = 2N³ × 4 bytes
AI_naive      = 2N³ / (2N³ × 4) = 1/4 = 0.25 FLOP/byte
```
Always deeply memory-bound regardless of N.

### Tiled GEMM
Tiles loaded into shared memory (SRAM). Each element of A and B loaded from DRAM **exactly once**.
```
Traffic_tiled = 2N² × 4 bytes
AI_tiled      = 2N³ / (2N² × 4) = N/4 FLOP/byte
```

### Traffic Ratio
```
Traffic_naive / Traffic_tiled = 2N³ / 2N² = N
```
**Ratio = N, not T.** T cancels because ideal tiling loads each element once regardless of tile size.

### AI Formula Derivation (know this cold)
```
AI = 2N³ / (2N² × 4)
   → 2s cancel: N³ / (N² × 4)
   → N³/N² = N: N / 4
   = N/4
```

### Execution Time
```
t_memory  = Bytes / Bandwidth
t_compute = FLOPs / Peak_Compute
Bottleneck = whichever is larger
```

### Worked Example (N=32, T=8, BW=320 GB/s, Peak=10 TFLOPS)
| | Naive | Tiled |
|---|---|---|
| Traffic | 2×32³×4 = 262,144 B | 2×32²×4 = 8,192 B |
| AI | 0.25 FLOP/byte | N/4 = 8 FLOP/byte |
| t_mem | 0.820 μs | 0.0256 μs |
| t_compute | 0.0066 μs | 0.0066 μs |
| Bound | Memory (125×) | Memory (3.9×) |

Ridge point = 10,000/320 = 31.25 → both memory-bound; tiled needs N > 125 to cross ridge.

### Common Kernels and their AI
| Kernel | AI | Bound |
|---|---|---|
| Vector add | 0.083 FLOP/byte | Memory |
| Naive GEMM | 0.25 FLOP/byte | Memory |
| Tiled GEMM (N=64, T=8) | 16 FLOP/byte | Depends on hardware |
| Tiled GEMM (N=1024, T=8) | 256 FLOP/byte | Compute |
| Conv layers | 10–100 FLOP/byte | Often Compute |

---

## Unit 4: What Is a Kernel?

A **kernel** is the function/operation in your code that takes the most total runtime — the bottleneck worth accelerating in hardware.

Key distinction: it's not necessarily the line that runs the *most times*, it's the one that takes the most *total time*:
```python
def train():
    load_data()        # runs 1000×, takes 1% of total time
    matrix_multiply()  # runs once per layer, takes 95% of total time  ← kernel
    apply_relu()       # fast, 4% of time
```

You find it by **profiling** — timing every function and finding what consumes >10% of runtime (Amdahl's law). That's what you build hardware to accelerate. Speeding up everything else barely moves the needle.

The hardware (GPU, TPU, ASIC) is what you build to run the kernel faster — the kernel is the *what*, the hardware is the *how you speed it up*.

---

## Unit 5: GPU Architecture

### SIMT Execution Model
- **SIMT**: Single Instruction, Multiple Threads
- **Warp**: 32 threads executing the same instruction in lockstep — the basic scheduling unit
- **SM** (Streaming Multiprocessor): execution unit containing CUDA cores, tensor cores, SRAM, warp schedulers
- **Thread Block**: group of threads assigned to one SM; shares that SM's shared memory
- **Grid**: all thread blocks for a kernel, distributed across all SMs

### Warp Divergence
When threads in a warp take **different branches**, the GPU serializes both paths:
```c
// BAD — causes divergence within a warp
if (threadIdx.x % 2 == 0) { ... } else { ... }
```
- Throughput drops up to **2× per branch level**
- Fix: ensure all 32 threads in a warp take the same path

### Occupancy
```
Occupancy = Active warps / Max warps per SM
```
- Higher occupancy → warp scheduler has more ready warps → better latency hiding
- **Limited by**: registers per thread, shared memory per block, thread block size
- Low occupancy = idle compute units, poor performance even with high AI

### CUDA Memory Hierarchy
| Memory | Scope | Speed | Size |
|---|---|---|---|
| Registers | per-thread | fastest | 16K × 32-bit/thread |
| Shared Memory | per-block (SM) | ~1 TB/s | 192–228 KB/SM |
| L2 Cache | GPU-wide | medium | 40 MB (A100) |
| Global Memory | all threads | slow (~BW) | GBs (DRAM) |
| Local Memory | per-thread spill | slow (DRAM) | — |
| Constant Memory | all threads, R/O | cached | 64 KB |

### CUDA Kernel Launch
```cuda
kernel<<<M, T>>>(args)   // M blocks, T threads per block
// Total threads = M × T
// Threads per block T should be multiple of 32 (warp size)
```

### GPU Comparison
| GPU | Process | FP32 TFLOPS | SMs | SRAM/SM | HBM BW |
|---|---|---|---|---|---|
| V100 | 12nm | 125 | 80 | — | 900 GB/s |
| A100 | 7nm | 312 (TF32) | 108 | 192 KB | 1.56 TB/s |
| H100 | 4nm | 67 FP32 | 132 | 228 KB | 3.35 TB/s |

### Tensor Cores
- Perform **small matrix multiplications** in one operation (4×4 and larger)
- Mixed precision: **FP16 inputs → FP32 accumulation**
- Generational speedups: Volta 8× → Ampere 16× → Hopper 32× → Blackwell further

---

## Unit 6: CNN/DNN Fundamentals

### Key Formula
```
FLOPs = 2 × MACs
Conv2D FLOPs = 2 × N × C_in × K² × H_out × W_out
```

### ResNet-18 Reference Numbers
- 11.69M parameters | **1.81B MACs** | 46.76 MB weights | 39.75 MB activations
- Top MAC layer: Conv2d 7×7, 3→64 channels, 112×112 output → AI ≈ 61 FLOP/byte

### Precision Formats
| Format | Bits | Key property |
|---|---|---|
| FP32 | 32 | Standard baseline |
| FP16 | 16 | Narrow range, stability risk |
| BF16 | 16 | FP32 exponent range, safer |
| INT8 | 8 | 4× bandwidth reduction |
| FP4 (E2M1) | 4 | Only 16 distinct values |

**PTQ** — quantize trained model; fast, less accurate
**QAT** — simulate quantization during training; better accuracy

---

## Unit 7: HW/SW Partitioning

### Decision Framework
| Question | Yes → | No → |
|---|---|---|
| Kernel > 10% runtime? | Consider HW | Leave in SW |
| Compute-bound at tile size? | Build fixed datapath | Optimize memory access first |
| Regular access pattern? | Good HW candidate | Keep in SW |

### Accelerator Template
```
DRAM → [Input Buffer SRAM] → [GEMM Engine] → [Output Buffer] → DRAM
                ↑
         AXI4-Lite (CPU control)
         AXI4-Stream (data)
         Controller FSM + Vector Engine
```

**PPAC**: Performance · Power · Area · Cost — all four matter in every hardware decision.

---

## Unit 8: VLSI Design Basics

### Abstraction Levels
1. **Algorithm** — Python, MATLAB
2. **RTL** — Verilog, SystemVerilog, VHDL
3. **Gate** — AND, OR, flip-flops
4. **Physical** — transistors, silicon layout

### Tool Chain
| Tool | Type | Role |
|---|---|---|
| Yosys / Synopsys | Synthesis | RTL → gates |
| OpenROAD / Cadence | PnR | gates → layout |
| Icarus / ModelSim | Simulation | verify behavior |
| cocotb | Testbench | Python-driven verification |

### Hardware Types
| Type | Flexible | Perf | Power Eff | Cost |
|---|---|---|---|---|
| CPU | ✓✓✓ | low | low | low |
| GPU | ✓✓ | high | medium | medium |
| FPGA | ✓ | high | medium | medium |
| ASIC | ✗ | highest | highest | very high |

---

## Unit 9: The Codefest Problems — What Was Tested

### CF01 — FC Network Workload Accounting
- Count MACs layer by layer: input_dim × output_dim per layer
- Weight bytes = total params × 4; activation bytes = total neurons × 4
- AI = 2×MACs / (weight_bytes + activation_bytes) → typically ~0.5 FLOP/byte (memory-bound)

### CF02 — Roofline Classification
- Given hardware (Peak, BW), compute ridge point
- For each kernel: compute FLOPs, Bytes, AI; compare to ridge; find attainable performance
- Dense GEMM → compute-bound; vector add → memory-bound

### CF03 — DRAM Traffic Analysis
- Naive: 2N³×4; Tiled: 2N²×4; Ratio = N
- Show formula first, then numbers — rubric checks both
- Both naive and tiled can still be memory-bound; crossing ridge requires AI > ridge point

---

## Quick-Reference: Numbers to Memorize

| Fact | Value |
|---|---|
| FP32 multiply energy | 3.7 pJ |
| DRAM 64-bit read energy | 640 pJ (170× multiply) |
| Warp size | 32 threads |
| H100 ridge point | ~20 FLOP/byte |
| Naive GEMM AI | 0.25 FLOP/byte (always) |
| Tiled GEMM AI | N/4 FLOP/byte |
| Traffic ratio (naive/tiled) | N |
| ResNet-18 MACs | 1.81 billion |
| A100 SRAM/SM | 192 KB |
| H100 SRAM/SM | 228 KB |

---

## Last-Minute Checklist

- [ ] Can I compute naive and tiled DRAM traffic for any N and T?
- [ ] Can I derive AI = N/4 step by step without looking?
- [ ] Do I know why the ratio equals N and not T?
- [ ] Can I classify a kernel as memory/compute-bound given AI and hardware specs?
- [ ] Do I know what limits occupancy?
- [ ] Can I explain warp divergence with a code example?
- [ ] Do I know the 4 CUDA memory spaces and their speeds?
- [ ] Do I know the 4 VLSI abstraction levels?
- [ ] Do I know PPAC and the accelerator datapath template?
- [ ] Do I know the trap: higher BW does NOT make a kernel compute-bound?
