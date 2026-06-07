# ECE 410/510 — Quiz 1 Practice Questions & Answers

---

## Section A: CMAN — DRAM Traffic & Arithmetic Intensity

**Q1.** For naive GEMM with N=64, FP32, what is the total DRAM traffic in bytes?
> Traffic = 2N³ × 4 = 2 × 262,144 × 4 = **2,097,152 bytes (2 MB)**

**Q2.** For tiled GEMM with N=64, T=8, FP32, what is the total DRAM traffic in bytes?
> Traffic = 2N² × 4 = 2 × 4,096 × 4 = **32,768 bytes (32 KB)**

**Q3.** What is the traffic ratio for Q1 vs Q2? Why?
> Ratio = 2N³ / 2N² = **N = 64**. In tiled GEMM each element is loaded from DRAM exactly once; in naive each element is loaded N times, so tiling removes N-fold redundancy.

**Q4.** What is the arithmetic intensity of naive GEMM with N=64, FP32?
> FLOPs = 2N³ = 524,288; Bytes = 2N³×4 = 2,097,152; **AI = 0.25 FLOP/byte**

**Q5.** What is the arithmetic intensity of tiled GEMM with N=64, T=8, FP32?
> FLOPs = 2N³ = 524,288; Bytes = 2N²×4 = 32,768; **AI = N/4 = 16 FLOP/byte**

**Q6.** Hardware: Peak = 10 TFLOPS, BW = 320 GB/s. What is the ridge point?
> Ridge = 10,000 / 320 = **31.25 FLOP/byte**

**Q7.** Using hardware from Q6: is naive GEMM (N=64) memory-bound or compute-bound?
> AI = 0.25 << 31.25 → **memory-bound**. Attainable = 0.25 × 320 = **80 GFLOP/s**

**Q8.** Using hardware from Q6: is tiled GEMM (N=64, T=8) memory-bound or compute-bound?
> AI = 16 < 31.25 → still **memory-bound**. Attainable = 16 × 320 = **5,120 GFLOP/s**

**Q9.** Using hardware from Q6: what N would make tiled GEMM compute-bound?
> Need AI = N/4 > 31.25 → N > 125 → **N ≥ 128** makes tiled GEMM compute-bound

**Q10.** For naive GEMM (N=64, BW=320 GB/s, Peak=10 TFLOPS), compute execution times and identify bottleneck.
> t_mem = 2,097,152 / 320e9 = **6.55 μs**; t_compute = 524,288 / 10e12 = **0.052 μs**; bottleneck = **memory** (126× slower)

**Q11.** For tiled GEMM (N=64, T=8, same hardware), compute execution times.
> t_mem = 32,768 / 320e9 = **0.102 μs**; t_compute = 524,288 / 10e12 = **0.052 μs**; bottleneck = **memory** (2× slower, near ridge)

**Q12.** FC network [512 → 256 → 10], batch=1, FP32, no bias. Compute total MACs, weight memory, activation memory, and AI.
> MACs: 512×256 + 256×10 = 131,072 + 2,560 = **133,632 MACs**
> Weights: 133,632 × 4 = **534,528 bytes**
> Activations: (512 + 256 + 10) × 4 = **3,112 bytes**
> AI = (2 × 133,632) / (534,528 + 3,112) = 267,264 / 537,640 = **0.497 FLOP/byte → memory-bound**

**Q13.** What does the N² term represent and should it be included in GEMM traffic?
> N²×4 bytes = writes to output matrix C. The assignment counts only **reads of A and B**, so the N² write term is excluded. For large N it's negligible anyway (N² << 2N³).

**Q14.** Simplify AI = 2N³ / (2N² × 4) step by step.
> 2s cancel → N³ / (N² × 4); N³/N² = N → **N/4**

---

## Section B: Roofline Model

**Q15.** What are the two axes of the roofline model?
> X-axis: **Arithmetic Intensity (FLOP/byte)**; Y-axis: **Performance (GFLOP/s)**

**Q16.** What are the two ceilings on a roofline plot?
> **Memory bandwidth ceiling** (diagonal line, slope = BW); **Peak compute ceiling** (horizontal line)

**Q17.** A kernel has AI = 50 FLOP/byte on hardware with ridge = 31.25. Is it compute-bound? What is attainable performance?
> AI > ridge → **compute-bound**; attainable = **peak compute** (10,000 GFLOP/s)

**Q18.** A kernel has AI = 5 FLOP/byte, BW = 320 GB/s, Peak = 10 TFLOPS. What is attainable performance?
> AI < ridge (31.25) → memory-bound; attainable = 5 × 320 = **1,600 GFLOP/s**

**Q19.** If you double peak compute but BW stays the same, what happens to the ridge point?
> Ridge = Peak/BW doubles → **harder to be compute-bound**; memory-bound region grows

**Q20.** If you double memory bandwidth but peak compute stays the same, what happens?
> Ridge = Peak/BW halves → **easier to be compute-bound**; more kernels become compute-bound

**Q21.** Does increasing memory bandwidth make a kernel compute-bound?
> **No** — bandwidth shifts the ridge point but does not change the kernel's AI. A kernel becomes compute-bound only if its AI exceeds the (new, lower) ridge point.

**Q22.** Vector addition (N=1M, FP32): FLOPs = N, Bytes = 3N×4 (2 reads + 1 write). Compute AI.
> AI = N / (12N) = **1/12 ≈ 0.083 FLOP/byte → deeply memory-bound**

---

## Section C: GPU Architecture

**Q23.** What is SIMT?
> **Single Instruction, Multiple Threads** — all 32 threads in a warp execute the same instruction simultaneously on different data.

**Q24.** What is a warp?
> A group of **32 threads** that execute lockstep on NVIDIA GPUs; the basic scheduling unit.

**Q25.** What is warp divergence and why is it bad?
> When threads in a warp take different if/else branches, the GPU **serializes both paths**, masking idle threads. Throughput drops up to 2× per branch level.

**Q26.** Give a code example that causes warp divergence.
> `if (threadIdx.x % 2 == 0) { ... } else { ... }` — odd and even threads within the same warp take different paths.

**Q27.** What is occupancy?
> **Active warps / maximum warps per SM**. Higher occupancy lets the warp scheduler hide memory latency by switching to ready warps.

**Q28.** What three things limit occupancy?
> **Register usage**, **shared memory size**, **thread block size** (too small = too few warps per SM)

**Q29.** What is the difference between shared memory and global memory?
> **Shared memory**: per-SM SRAM, ~1 TB/s bandwidth, low latency, shared within a thread block.
> **Global memory**: DRAM, ~192–3350 GB/s, high latency, accessible by all threads.

**Q30.** What is a Streaming Multiprocessor (SM)?
> The core GPU execution unit containing CUDA cores, tensor cores, shared memory, and warp schedulers. Thread blocks are assigned to SMs.

**Q31.** What does `kernel<<<M, T>>>(args)` mean?
> Launch kernel with **M thread blocks**, each containing **T threads**. Total threads = M × T.

**Q32.** How many total threads in `kernel<<<256, 128>>>()`?
> 256 × 128 = **32,768 threads**

**Q33.** Can two thread blocks share data through shared memory?
> **No** — shared memory is per-SM and per-block. Cross-block communication requires global memory.

**Q34.** List CUDA memory spaces from fastest to slowest.
> Registers → Shared Memory → L1/L2 Cache → Global Memory (DRAM) → Local Memory (spills to DRAM)

---

## Section D: GPU Hardware

**Q35.** Compare V100 vs A100: key differences.
> A100: 3× SMEM/L1 BW, 6.7× larger L2 (40 MB), NVLink 3 (600 GB/s), async copies, 7nm vs 12nm

**Q36.** What are tensor cores and what do they do?
> Specialized hardware units that perform **small matrix multiplications** (4×4 or larger) in one operation. Use mixed precision: FP16 inputs → FP32 accumulation.

**Q37.** What precision formats does A100 support in tensor cores?
> **TF32, FP16, BF16, INT8, FP64**

**Q38.** What is the difference between FP16 and BF16?
> Both are 16-bit. BF16 has the **same exponent range as FP32** (8 exponent bits) → better dynamic range, less overflow risk. FP16 has more mantissa bits → higher precision but narrower range.

**Q39.** What is the energy cost of DRAM vs compute and why does it matter?
> DRAM read ≈ **640 pJ**; FP32 multiply ≈ **3.7 pJ** → 170× difference. Minimizing data movement is the primary optimization target for both performance and energy efficiency.

---

## Section E: HW/SW Codesign & Partitioning

**Q40.** What is the HW/SW codesign principle?
> Co-optimize the algorithm and hardware together rather than designing them independently. Identify compute-bound kernels and build fixed datapaths; keep irregular or memory-bound ops in software.

**Q41.** When should you accelerate a kernel in hardware?
> When it consumes **>10% of runtime** (Amdahl's law) AND is **compute-bound at the target tile size** with regular, predictable memory access patterns.

**Q42.** When should you keep a kernel in software (CPU)?
> When it has **irregular/data-dependent access patterns**, sparse operations, or pointer chasing — hardware fixed datapaths cannot handle these efficiently.

**Q43.** What are the four PPAC trade-offs in hardware design?
> **Performance · Power · Area · Cost** — every hardware decision involves all four.

**Q44.** What is the accelerator datapath template?
> Input Buffer (SRAM) → GEMM Engine → Output Buffer → DMA → DRAM, controlled via AXI4-Lite from CPU host.

---

## Section F: VLSI & Design Abstraction

**Q45.** List the four abstraction levels from algorithm to silicon.
> 1. Behavior/Algorithm (Python, MATLAB)
> 2. RTL (Verilog, SystemVerilog)
> 3. Gate Level (logic gates)
> 4. Transistor/Physical (silicon layout)

**Q46.** What is the difference between FPGA and ASIC?
> **FPGA**: reconfigurable after fabrication, flexible, medium power efficiency.
> **ASIC**: fixed function, maximum performance and power efficiency, very high design cost, no reconfiguration.

**Q47.** What tool converts RTL to a gate-level netlist?
> **Synthesis** tool — e.g., Yosys (open-source) or Synopsys Design Compiler (commercial).

**Q48.** What is cocotb?
> A **Python-based co-simulation testbench framework** for verifying hardware designs without writing testbenches in Verilog/VHDL.

---

## Section G: CNN/DNN

**Q49.** What is a MAC and how many FLOPs is it?
> **Multiply-Accumulate** operation: 1 multiply + 1 add = **2 FLOPs**

**Q50.** ResNet-18 stats from memory:
> 11.69M parameters, **1.81B MACs**, 46.76 MB weights, 39.75 MB activations

**Q51.** What is the Conv2D FLOPs formula?
> FLOPs = 2 × N × C_in × K × K × H_out × W_out
> (N=batch, C_in=input channels, K=kernel size, H/W_out=output spatial dims)

**Q52.** What is quantization and why use it?
> Reducing weight/activation precision (e.g., FP32 → INT8) to **reduce memory, bandwidth, and energy** with minimal accuracy loss.

**Q53.** PTQ vs QAT — what's the difference?
> **PTQ** (Post-Training Quantization): quantize after training; simple but less accurate.
> **QAT** (Quantization-Aware Training): simulate quantization during training; better accuracy, more expensive.

---

## Section H: Trap / Trick Questions

**Q54.** Does tiling a kernel always make it compute-bound?
> **No.** Tiling raises AI but the kernel is only compute-bound if AI > ridge point. For small N or small T, tiled may still be memory-bound.

**Q55.** Does increasing tile size T always improve performance?
> **No.** Larger T raises AI and reduces DRAM traffic, but too-large T reduces the number of thread blocks → **low occupancy** → poor latency hiding.

**Q56.** The traffic ratio for naive vs tiled GEMM equals T, right?
> **No.** Ratio = 2N³ / 2N² = **N**, not T. T cancels out because ideal tiling loads each element exactly once regardless of T.

**Q57.** Higher memory bandwidth makes kernels compute-bound, right?
> **No.** Higher BW lowers the ridge point (makes it easier to be compute-bound), but a kernel's AI is fixed by its algorithm. It only becomes compute-bound if AI > new ridge.

**Q58.** The AI formula for tiled GEMM simplifies to N/2, right?
> **No.** 2N³ / (2N²×4) → 2s cancel → N³/(N²×4) → **N/4**, not N/2.

**Q59.** Should the N²×4 write-to-C term be included in GEMM traffic?
> **No** — the assignment counts only reads of input matrices A and B. Writes to C are excluded.

**Q60.** If a kernel is theoretically compute-bound (AI >> ridge), will it always run near peak compute?
> **No.** Other bottlenecks can limit performance: low occupancy, insufficient parallelism, shared memory bank conflicts, or instruction-level bottlenecks. (e.g., tiled T=8 is theoretically compute-bound but achieves only 3.8% of peak due to low occupancy.)
