# ECE 410/510 — Quiz 1 Cheat Sheet

---

## 1. Big Picture: Why Hardware for AI?

- **Memory Wall**: Compute has outpaced memory bandwidth for 30+ years → most workloads are **memory-bound**
- **Energy**: DRAM read (64-bit) ≈ **640 pJ** vs FP32 multiply ≈ **3.7 pJ** → 170× more expensive to move data than compute
- **Architecture evolution**: CPU → CPU+GPU → heterogeneous → extreme heterogeneity (NPU, TPU, FPGA, ASIC)
- **HW/SW Codesign**: Co-optimize algorithm + hardware together; outperforms siloed approaches

---

## 2. Key Formulas

| Formula | What it means |
|---|---|
| **Arithmetic Intensity** I = FLOPs / Bytes | How compute-heavy a kernel is |
| **Roofline Performance** = min(Peak_FLOPS, I × Peak_BW) | Achievable FLOPS given I and hardware |
| **Ridge Point** I* = Peak_FLOPS / Peak_BW | Boundary between memory-bound and compute-bound |
| 1 MAC = 2 FLOPs | 1 multiply + 1 add |

**H100 example**: Peak = 67 TFLOPS FP32, BW = 3.35 TB/s → I* ≈ 20 FLOP/byte

---

## 3. Arithmetic Intensity of Common Kernels

| Kernel | AI (FLOP/byte) | Bound |
|---|---|---|
| Vector add | 0.083 | Memory |
| Naive GEMM (N=64) | ~0.25 | Memory |
| Tiled GEMM (T=16) | ~10.7 | Compute |
| Conv layers | 10–100 | Compute |

---

## 4. GEMM Tiling — Critical Numbers

**Naive GEMM (M=N=K=64, FP32)**
- DRAM traffic: **2 MB** → output: 16 KB
- AI ≈ 0.25 FLOP/byte → attainable: **0.84 TFLOPS** (1.25% of peak)

**Tiled GEMM (T=16)**
- DRAM traffic: **32 KB** (64× reduction)
- AI ≈ 10.7 FLOP/byte → attainable: **35.8 TFLOPS** (53% of peak)

**Key idea**: Tiling brings data into SRAM (fast) to be reused many times before eviction.

---

## 5. Memory Hierarchy

| Level | Bandwidth | Latency | Where |
|---|---|---|---|
| Registers | ~fastest | cycles | per-thread |
| SRAM (Shared Mem) | ~1 TB/s | low | per-SM |
| L2 Cache | — | mid | GPU-wide |
| DRAM (HBM) | 3.35 TB/s | ~100× SRAM | off-chip |

- **H100 SRAM**: 228 KB per SM
- **A100 SRAM**: 192 KB per SM (split L1/SMEM)
- **A100 L2**: 40 MB (6.7× larger than V100)

---

## 6. GPU Architecture

### Key Terms
| Term | Definition |
|---|---|
| **SIMT** | Single Instruction, Multiple Threads — all threads in a warp run same instruction |
| **Warp** | 32 threads executing lockstep (NVIDIA) |
| **SM** (Streaming Multiprocessor) | Core GPU execution unit; has CUDA cores, tensor cores, SRAM |
| **Thread Block** | Group of threads sharing one SM's resources |
| **Grid** | All thread blocks executing a kernel |
| **Occupancy** | Active warps / max warps per SM |

### CUDA Kernel Launch
```
kernel<<<M, T>>>(args)   // M = blocks, T = threads/block
```

### CUDA Memory Spaces
| Space | Scope | Speed |
|---|---|---|
| Registers | per-thread | fastest |
| Shared Memory | per-block (SM) | fast (~1 TB/s) |
| Local Memory | per-thread (spills) | slow (DRAM) |
| Global Memory | all threads | slow (DRAM) |
| Constant Memory | all threads, read-only | cached |

---

## 7. Warp Divergence

- If threads in a warp take **different if/else paths**, GPU **serializes both paths**
- Idle threads are masked off → throughput drops up to **2× per branch level**
- **Avoid**: `if (threadIdx.x % 2 == 0)` within a warp
- **Goal**: Keep all 32 threads on the same code path

---

## 8. Occupancy

- **Definition**: Active warps / max warps per SM
- Higher occupancy → more latency hiding → better throughput
- **Limited by**: register usage, shared memory size, block size
- Low occupancy → idle compute units, memory latency not hidden

---

## 9. GPU Hardware Comparison

| GPU | Process | TFLOPS (FP32) | Tensor Cores | SMs | HBM BW |
|---|---|---|---|---|---|
| V100 | 12nm | 125 | 640 | 80 | 900 GB/s |
| A100 | 7nm | 312 (TF32) | 432 | 108 | 1.56 TB/s |
| H100 | 4nm | 67 (FP32) / ~1000 (FP8) | — | 132 | 3.35 TB/s |

**A100 improvements over V100**: 3× SMEM/L1 BW, 6.7× L2, NVLink 3 (600 GB/s), async copies

---

## 10. Tensor Cores

- Specialized hardware for **4×4 (or larger) matrix ops**
- Mixed precision: **FP16 inputs → FP32 accumulation**
- Evolution by architecture:
  - Volta (V100): 8× FP16 throughput
  - Ampere (A100): 16× with TF32, INT8, FP64
  - Hopper (H100): 32× with FP8 support
  - Blackwell (B200): further scaling

---

## 11. Precision Formats

| Format | Bits | Use case |
|---|---|---|
| FP32 | 32 | Standard training/inference |
| FP16 | 16 | Reduced memory; stability concerns |
| BF16 | 16 | Same exponent range as FP32; safer than FP16 |
| INT8 | 8 | Post-training quantization, inference |
| FP4 (E2M1) | 4 | 16 distinct values; 2 exp bits, 1 mantissa bit |
| NVFP4 | 4+scale | Micro-block scaled FP4 (16 values/block) |

**Quantization types**:
- **PTQ** (Post-Training Quantization): quantize already-trained model; simpler but less accurate
- **QAT** (Quantization-Aware Training): simulate quantization during training; better accuracy

---

## 12. CNN / DNN Operations

- **ResNet-18**: 11.69M params, **1.81B MACs**, 46.76 MB weights, 39.75 MB activations
- **Parallelism types**:
  - **Data parallelism**: multiple training samples at once
  - **Model parallelism**: partition network across processors
  - **Operator-level**: parallelize within GEMM/conv

---

## 13. HW/SW Partitioning — Decision Framework

| Condition | Action |
|---|---|
| Kernel >10% runtime (Amdahl) | Accelerate in hardware |
| Compute-bound at target tile size | Build fixed datapath (GEMM engine, conv) |
| Memory-bound | Restructure data access (tiling, reuse) |
| Irregular access / data-dependent branches | Keep in software (CPU) |

---

## 14. VLSI / ASIC Design Concepts

### Abstraction Levels (top → bottom)
1. **Behavior/Algorithm**: Python, MATLAB
2. **RTL**: Verilog, SystemVerilog, VHDL
3. **Gate Level**: logic gates
4. **Transistor/Physical**: silicon layout

### Tools
| Tool | Type | Purpose |
|---|---|---|
| Yosys | Open-source | Synthesis |
| Synopsys / Cadence | Commercial | Full EDA suite |
| OpenROAD | Open-source | RTL-to-GDS |
| Icarus Verilog | Open-source | Simulation |
| cocotb | Open-source | Python-based testbenches |
| ModelSim / VCS | Commercial | Simulation |

### Hardware Types Compared
| Type | Flexibility | Performance | Power Eff. | Cost |
|---|---|---|---|---|
| CPU | High | Low | Low | Low |
| GPU | Medium | High | Medium | Medium |
| FPGA | Medium | High | Medium | Medium |
| ASIC | None | Highest | Highest | Very High |
| NPU/TPU | Low | Very high | Very high | High |

---

## 15. Accelerator Datapath Template

```
[Input Buffer (SRAM)] → [GEMM Engine] → [Output Buffer] → DMA → DRAM
        ↑                      ↑
   AXI4-Stream data       AXI4-Lite control (CPU host)
        ↑
  [Controller FSM — tile iteration]
  [Vector Engine — optional post-processing]
```

**PPAC trade-offs**: Performance · Power · Area · Cost — every HW decision involves all four.

---

## 16. Quick-Reference Numbers

| Fact | Value |
|---|---|
| FP32 multiply energy | ~3.7 pJ |
| DRAM 64-bit read energy | ~640 pJ (170× multiply) |
| A100 SRAM/SM | 192 KB |
| H100 SRAM/SM | 228 KB |
| A100 L2 cache | 40 MB |
| H100 HBM3 BW | 3.35 TB/s |
| H100 ridge point | ~20 FLOP/byte |
| Warp size (NVIDIA) | 32 threads |
| ResNet-18 MACs | 1.81 billion |
| Tiling DRAM reduction | 64× (T=16 example) |

---

## 23. Likely Quiz Question Types

### High probability — CMAN numerical (show all work, ±10%)
- New matrix size or network shape → compute DRAM traffic, AI, execution time, bound
- Formula template: Traffic_naive = 2N³×4, Traffic_tiled = 2N²×4, ratio = N
- AI = FLOPs / Bytes; ridge = Peak_compute / BW; bound = compare AI to ridge

### High probability — conceptual
- Why does the traffic ratio equal **N** and not T?
  → Each element loaded once in tiled vs. N times naive; T cancels out of 2N³/2N² = N
- What limits tiled GEMM from achieving theoretical speedup?
  → Low occupancy (small T → few threads/block) + L2 cache absorbing naive's traffic
- Why is DRAM 170× more expensive than FP32 multiply?
  → Energy: 640 pJ (DRAM read) vs 3.7 pJ (FP32 multiply)

### High probability — roofline reading
- Given a plot: identify memory-bound vs compute-bound kernels
- Given peak + BW: compute ridge point and classify a kernel by its AI

### Short definitions (know cold)
- **SIMT**: Single Instruction Multiple Threads — all 32 threads in a warp run same instruction
- **Warp**: 32 threads executing lockstep
- **Occupancy**: active warps / max warps per SM; limited by registers, SMEM, block size
- **AI**: FLOPs / Bytes — measures compute-to-memory ratio
- **Shared memory**: per-SM fast SRAM (~1 TB/s); **global memory**: DRAM, slow, all threads

### Trap questions — common wrong answers
| Question | Wrong answer | Correct answer |
|---|---|---|
| Does tiling make kernel compute-bound? | Yes | Only if AI > ridge; depends on N and T |
| Does higher BW make kernel compute-bound? | Yes | No — lowers ridge point but AI is unchanged |
| Does increasing T always help? | Yes | Raises AI but too-small T kills occupancy |
| Traffic ratio = ? | T | **N** (2N³ / 2N² = N) |
| AI formula for tiled N=1024 | N/2 = 512 | **N/4 = 256** (2N³ / (2N²×4) = N/4) |

---

## 17. CMAN Worked Example — Codefest 1

**Network**: 3-layer fully connected [784 → 256 → 128 → 10], batch=1, FP32, no bias

### Step 1: MACs per layer
| Layer | Computation | MACs |
|---|---|---|
| Layer 1 | 784 × 256 | 200,704 |
| Layer 2 | 256 × 128 | 32,768 |
| Layer 3 | 128 × 10 | 1,280 |
| **Total** | | **234,752** |

### Step 2: Parameters & Memory
- **Weights** = 234,752 params × 4 bytes = **939,008 bytes**
- **Activations** = (784 + 256 + 128 + 10) × 4 = 1,178 × 4 = **4,712 bytes**

### Step 3: Arithmetic Intensity
```
AI = (2 × 234,752 MACs) / (939,008 + 4,712) bytes
   = 469,504 / 943,720
   ≈ 0.497 FLOP/byte   → deeply MEMORY-BOUND
```

---

## 18. CMAN Worked Example — Codefest 2

**Hardware**: Peak = 10 TFLOPS, BW = 320 GB/s → Ridge point = 31.25 FLOP/byte

### Kernel A: Dense GEMM (N=1024, square)
- FLOPs = 2 × 1024³ = **2.147 B FLOPs**
- Bytes = 3 × 1024² × 4 = **12.58 MB**
- **AI = 170.67 FLOP/byte** → **COMPUTE-BOUND** (>> 31.25)
- Attainable = **10,000 GFLOP/s** (hits peak compute ceiling)
- Fix: Add more FP32 parallel units

### Kernel B: Vector Addition (N = 4,194,304)
- FLOPs = **4.19 M FLOPs**
- Bytes = 3 × 4,194,304 × 4 = **50.3 MB** (2 reads + 1 write)
- **AI = 0.083 FLOP/byte** → **MEMORY-BOUND** (<< 31.25)
- Attainable = 0.083 × 320 = **26.67 GFLOP/s**
- Fix: Widen memory bus / improve buffering

### ResNet-18 Conv2d Layer 1 Analysis
Config: 7×7 kernel, 3 in-ch, 64 out-ch, 112×112 output, FP32
- MACs = 118,013,952 → FLOPs = **236,027,904**
- Weights: 9,408 × 4 = 37,632 bytes
- Input activations: 3 × 224 × 224 × 4 = 602,112 bytes
- Output activations: 64 × 112 × 112 × 4 = 3,211,264 bytes
- Total bytes = **3,851,008 bytes**
- **AI ≈ 61.29 FLOP/byte** — decent reuse; tiling pushes toward compute-bound

**Top-5 MAC layers in ResNet-18:**

| Rank | Layer | MACs | Params |
|---|---|---|---|
| 1 | Conv2d: 1-1 | 118,013,952 | 9,408 |
| 2–5 | Conv2d: 3-x | 115,605,504 each | 36,864 each |

---

## 19. CMAN Worked Example — Codefest 3

**Setup**: C = A × B, N=32, FP32 (4 bytes), Tile T=8, BW=320 GB/s, Peak=10 TFLOP/s

### (a) Naive DRAM Traffic
Only count reads of A and B (assignment does not count writes to C):
```
Each element of A read N times, each element of B read N times
Traffic = 2N³ × 4 = 2 × 32768 × 4 = 262,144 bytes
```

### (b) Tiled DRAM Traffic (T=8)
Each tile of A and B is loaded exactly once across the full computation:
```
Traffic = 2N² × 4 = 2 × 1024 × 4 = 8,192 bytes
```

### (c) Traffic Reduction Ratio
```
2N³ / 2N² = N = 32
```
Each element was read N times naively; tiling loads each element exactly once → N× reduction.

### (d) Execution Times

| | Naive | Tiled |
|---|---|---|
| Memory time | 262,144 / 320e9 = **0.820 μs** | 8,192 / 320e9 = **0.0256 μs** |
| Compute time | 65,536 / 10e12 = **0.0066 μs** | 65,536 / 10e12 = **0.0066 μs** |
| Bottleneck | Memory (125× slower) | Memory (3.9× slower) |
| Bound | Memory-bound | Memory-bound (much closer to ridge) |

**Key insight**: Tiling reduces DRAM traffic by factor N (not T). The ratio equals N because each of the N² elements in A and B was accessed N times naively.

---

## 20. Conv2D FLOPs Formula

```
FLOPs = 2 × N × C_in × K × K × H_out × W_out
```
Where: N = batch, C_in = input channels, K = kernel size, H_out/W_out = output spatial dims

---

## 21. Execution Time Formula

```
T_total  = max(T_compute, T_memory)
T_compute = FLOPs / Peak_compute
T_memory  = Bytes / Bandwidth
```
If T_memory >> T_compute → memory-bound; optimize data movement
If T_compute >> T_memory → compute-bound; add arithmetic units

---

## 22. Profiling Tools (Know These)

| Tool | What it does |
|---|---|
| `cProfile` + `pstats` | Function-level timing in Python |
| `line_profiler` + `kernprof` | Line-level timing |
| `snakeviz` | Interactive visualization of cProfile output |
| `timeit` | Micro-benchmarking small snippets |
| `time.perf_counter()` | High-res wall-clock timer |
| `pycallgraph2` | Dynamic call graph visualization |
