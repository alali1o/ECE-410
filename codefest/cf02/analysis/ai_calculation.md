# Arithmetic Intensity Calculation — Dominant Kernel
**ECE 410/510 Spring 2026 — Codefest 2 CLLM**
**Project: Ternary NN Inference Accelerator for Keyword Spotting**

---

## Dominant Kernel Identified

From cProfile output (`project_profile.txt`), the function `ternary_linear` (which calls
`numpy.dot`) is invoked **150 times** across 50 inference runs (3 calls/run — one per FC layer).
It accounts for the vast majority of arithmetic operations in the forward pass.

**Kernel:** Ternary fully-connected forward pass — matrix-vector multiply across all 3 layers:
- Layer 1: input (40) × W1 (40 × 256) → hidden1 (256)
- Layer 2: hidden1 (256) × W2 (256 × 128) → hidden2 (128)
- Layer 3: hidden2 (128) × W3 (128 × 10) → logits (10)

---

## Step 1 — Count FLOPs

Each FC layer: one multiply + one accumulate per weight = **2 FLOPs per weight**.

```
Layer 1:  2 × 40  × 256 =  20,480 FLOPs
Layer 2:  2 × 256 × 128 =  65,536 FLOPs
Layer 3:  2 × 128 × 10  =   2,560 FLOPs
─────────────────────────────────────────
Total FLOPs = 20,480 + 65,536 + 2,560 = 88,576 FLOPs
```

> Note: In hardware, ternary weights {-1, 0, +1} replace multipliers with conditional adders.
> But for FLOPs counting we still charge 2 ops (1 multiply + 1 accumulate) per weight,
> matching the software baseline which uses FP32 np.dot.

---

## Step 2 — Count Bytes Transferred (no DRAM reuse)

All weights and activations are assumed loaded from DRAM fresh each inference.
Software baseline uses **FP32 (4 bytes per element)**.

### Weight bytes (load once per inference)
```
Layer 1 weights:  40  × 256 × 4 =  40,960 bytes
Layer 2 weights:  256 × 128 × 4 = 131,072 bytes
Layer 3 weights:  128 × 10  × 4 =   5,120 bytes
─────────────────────────────────────────────────
Total weight bytes = 177,152 bytes
```

### Activation bytes (load input + store/load each layer output)
```
Input  (40  elements): 40  × 4 =    160 bytes
h1     (256 elements): 256 × 4 =  1,024 bytes
h2     (128 elements): 128 × 4 =    512 bytes
output (10  elements): 10  × 4 =     40 bytes
─────────────────────────────────────────────
Total activation bytes = 1,736 bytes
```

### Total bytes
```
Total bytes = 177,152 + 1,736 = 178,888 bytes ≈ 174.7 KB
```

---

## Step 3 — Arithmetic Intensity

```
AI = Total FLOPs ÷ Total bytes
   = 88,576 ÷ 178,888
   = 0.495 FLOP/byte
```

---

## Step 4 — Placement on Apple M4 Roofline

| Hardware parameter     | Value                        |
|------------------------|------------------------------|
| Peak FP32 compute      | 230 GFLOPS                   |
| Peak memory bandwidth  | 120 GB/s                     |
| Ridge point            | 230 ÷ 120 = **1.92 FLOP/byte** |

```
Kernel AI = 0.495 FLOP/byte  <  Ridge point = 1.92 FLOP/byte

→ MEMORY-BOUND on Apple M4 CPU

Attainable performance = AI × BW = 0.495 × 120 = 59.4 GFLOPS
```

The kernel sits firmly in the memory-bound region. The weights dominate data movement —
177 KB of weights vs only 1.7 KB of activations — so the bottleneck is DRAM bandwidth,
not arithmetic throughput.

---

## Summary

| Metric               | Value            |
|----------------------|------------------|
| Total FLOPs          | 88,576           |
| Weight bytes (FP32)  | 177,152 bytes    |
| Activation bytes     | 1,736 bytes      |
| **Total bytes**      | **178,888 bytes**|
| **AI**               | **0.495 FLOP/byte** |
| Bound (M4 CPU)       | **Memory-bound** |
| Attainable perf      | 59.4 GFLOPS      |
