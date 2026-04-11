# Arithmetic Intensity Calculation
**Project:** Ternary KWS Inference Accelerator  
**ECE 410/510 Spring 2026**

---

## Dominant Kernel

From profiling (`codefest/cf02/profiling/project_profile.txt`, 50 runs, cProfile):

> **The dominant kernel is FC1 (1960 → 512 fully-connected layer), accounting for 93.8% of all floating-point operations (2,007,040 / 2,140,672 FLOPs).**

Note: in the software simulation the `ternarize` function (on-the-fly weight quantization) consumes 75.6% of wall-clock time, but this is a software artifact. In the hardware target, weights are pre-quantized and stored in on-chip SRAM. The compute kernel that hardware accelerates is the matrix-vector multiply (`torch._C._nn.linear`, 9.7% of SW runtime), which in hardware becomes 100% of the compute workload.

---

## FLOPs — Analytical Derivation

For a single fully-connected layer with N_in inputs and N_out outputs, each output neuron requires N_in multiply-accumulate (MAC) operations. Each MAC = 1 multiply + 1 add = **2 FLOPs**.

**Formula:**
```
FLOPs(FC) = 2 × N_in × N_out
```

**FC1 (dominant kernel):**
```
FLOPs_FC1 = 2 × 1960 × 512
           = 2 × 1,003,520
           = 2,007,040 FLOPs
```

**Full network:**
```
FLOPs_FC1 = 2 × 1960 × 512 = 2,007,040
FLOPs_FC2 = 2 ×  512 × 128 =   131,072
FLOPs_FC3 = 2 ×  128 ×  10 =     2,560
─────────────────────────────────────────
Total FLOPs                 = 2,140,672
```

FC1 fraction: 2,007,040 / 2,140,672 = **93.8%**

---

## Bytes Transferred — DRAM, No Reuse (Float32 Baseline)

Assuming all operands are loaded from DRAM once per inference (worst-case, no cache reuse):

| Operand | Dimensions | Element type | Bytes |
|---|---|---|---|
| Weight matrix **W** | 512 × 1960 | float32 (4 B) | 512 × 1960 × 4 = **4,014,080** |
| Input vector **x** | 1960 × 1 | float32 (4 B) | 1960 × 4 = **7,840** |
| Output vector **y** | 512 × 1 | float32 (4 B) | 512 × 4 = **2,048** |
| **Total** | | | **4,023,968 bytes ≈ 3.84 MB** |

---

## Arithmetic Intensity (Float32 Baseline)

```
AI_fp32 = FLOPs / Bytes
        = 2,007,040 / 4,023,968
        = 0.499 FLOP/byte
        ≈ 0.50 FLOP/byte
```

---

## Bytes Transferred — Ternary Weights + Float32 Activations

With ternary packing (2 bits per weight), the weight operand shrinks:

| Operand | Dimensions | Element type | Bytes |
|---|---|---|---|
| Weight matrix **W** | 512 × 1960 | 2-bit ternary | 512 × 1960 × 2 / 8 = **250,880** |
| Input vector **x** | 1960 × 1 | float32 (4 B) | **7,840** |
| Output vector **y** | 512 × 1 | float32 (4 B) | **2,048** |
| **Total** | | | **260,768 bytes ≈ 255 KB** |

```
AI_ternary_DRAM = 2,007,040 / 260,768 = 7.70 FLOP/byte
```

Even with ternary packing, the kernel remains memory-bound on the Apple M4 CPU (ridge point ≈ 33 FLOP/byte).

---

## Arithmetic Intensity — Hardware Accelerator (On-Chip Weights)

In the target hardware design, the 250,880-byte ternary weight matrix for FC1 fits entirely in on-chip SRAM. Only the input and output vectors cross the SPI interface:

| Operand | Source | Bytes |
|---|---|---|
| Weight matrix **W** | On-chip SRAM (no interface traffic) | 0 |
| Input vector **x** | SPI (INT16, 2 bytes/coeff) | 1960 × 2 = **3,920** |
| Output vector **y** | SPI (INT16) | 512 × 2 = **1,024** |
| **Total interface bytes** | | **4,944 bytes** |

```
AI_hw = 2,007,040 / 4,944 = 406 FLOP/byte
```

This places the accelerator well above the ridge point of any realistic ASIC, making it **compute-bound**. See `codefest/cf02/profiling/roofline_project.png`.

---

## Summary

| Configuration | AI (FLOP/byte) | Bound |
|---|---|---|
| Float32, DRAM | 0.50 | Memory-bound (SW baseline) |
| Ternary 2-bit, DRAM | 7.70 | Memory-bound |
| Ternary 2-bit, on-chip weights | **406** | **Compute-bound (HW target)** |
