# CMAN — DRAM Traffic Analysis: Naive vs. Tiled Matrix Multiply
ECE 410/510 Spring 2026 — Codefest 3

N = 32, T = 8, FP32 (4 bytes/element), FLOPs = 2 × 32³ = 65,536

---

## (a) Naive DRAM Traffic

Each output element C[i][j] requires a full dot product over k = 0..31.
That means every element of B gets loaded once per output row — **N times total**.
Same applies to A. With no caching, every access hits DRAM.

```
A accesses = N³ = 32,768 elements → 32,768 × 4 = 131,072 bytes
B accesses = N³ = 32,768 elements → 32,768 × 4 = 131,072 bytes
C writes   = N² =  1,024 elements →  1,024 × 4 =   4,096 bytes

Total = 266,240 bytes ≈ 260 KB
```

---

## (b) Tiled DRAM Traffic (T = 8)

Blocking into 8×8 tiles means each tile is loaded once and reused T = 8 times inside fast memory before the next DRAM load. Each element is now loaded N/T = 4 times instead of 32.

```
Tile loads per matrix = (N/T)³ = 4³ = 64 loads
Bytes per tile = T² × 4 = 256 bytes

A traffic  = 64 × 256 = 16,384 bytes
B traffic  = 64 × 256 = 16,384 bytes
C writes   =            4,096 bytes

Total = 36,864 bytes ≈ 36 KB
```

---

## (c) Traffic Ratio

```
266,240 / 36,864 ≈ 7.2×  (including C writes)
262,144 / 32,768 =  8  =  T  (A+B loads only)
```

The ratio equals N/T because tiling gives each tile T-fold reuse, reducing each element's load count from N to N/T — a factor of T reduction per matrix.

---

## (d) Execution Time and Bound

```
Ridge point = 10,000 GFLOPS / 320 GB/s = 31.25 FLOP/byte

AI_naive = 65,536 / 266,240 = 0.25 FLOP/byte  → memory-bound
AI_tiled = 65,536 /  36,864 = 1.78 FLOP/byte  → memory-bound (closer to ridge)

Naive: memory time = 266,240 / 320e9 ≈ 832 ns   → bottleneck: MEMORY
Tiled: memory time =  36,864 / 320e9 ≈ 115 ns   → bottleneck: MEMORY (7.2× faster)
Compute time (both) = 65,536 / 10e12 ≈ 6.6 ns   (not the bottleneck)
```

Both kernels are memory-bound since their AI falls well below the ridge point of 31.25 FLOP/byte. Tiling reduces traffic by ~7× and cuts execution time proportionally, but a larger tile size would be needed to reach the compute-bound regime.
