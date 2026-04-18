# GEMM Analysis — Naive vs. Tiled CUDA Kernel
ECE 410/510 Spring 2026 — Codefest 3 CLLM

GPU: NVIDIA T4 | Peak FP32: 8,100 GFLOPS | Peak BW: 300 GB/s | Ridge: 27 FLOP/byte
Matrix size: N = 1024 × 1024 FP32 | AI = 2N³ / (3N² × 4) = **170.67 FLOP/byte**

---

## (a) Why the naive kernel is memory-bound

Even though the arithmetic intensity of a 1024×1024 GEMM is 170.67 FLOP/byte — well above the T4 ridge point of 27 FLOP/byte — the naive kernel fails to exploit this. Each thread independently computes one output element by sweeping through an entire row of A and column of B from global memory. No data is shared between threads: every element of B is reloaded from DRAM once per output row (N = 1024 times total). With no caching or reuse, the effective memory traffic is O(N³) and the kernel is severely bandwidth-limited, achieving only a fraction of the GPU's 8.1 TFLOPS peak.

## (b) How tiling reduces DRAM traffic

The tiled kernel partitions A and B into 8×8 tiles that are cooperatively loaded into shared memory by all threads in a block. Once a tile is in shared memory (latency ~5 cycles), it is reused 8 times — once for each element in the perpendicular tile dimension — before the next DRAM load. This cuts the number of DRAM accesses per element from N = 1024 down to N/T = 128, reducing total DRAM traffic by a factor of T = 8. The kernel can therefore feed its arithmetic units from fast shared memory (>1 TB/s on-chip bandwidth) instead of waiting on the 300 GB/s DRAM bus.

## (c) Did the tiled kernel achieve the expected improvement?

Measured on Google Colab NVIDIA T4 GPU, N = 1024, FP32:

| Kernel | GFLOP/s | BW (GB/s) | % of Peak | Bound |
|--------|---------|-----------|-----------|-------|
| Naive  | 316.09  | 1.85      | 3.9%      | Compute (severely underutilized) |
| Tiled  | 392.46  | 2.30      | 4.8%      | Compute (underutilized) |

**Speedup: 1.24×**

Both kernels sit well above the ridge point (AI = 170.67 >> 27 FLOP/byte), so they are nominally compute-bound. However, neither approaches the 8,100 GFLOPS peak — the naive kernel achieves only 3.9% and tiled 4.8%. The modest 1.24× speedup is expected given T = 8: each tile reuses data only 8 times before the next DRAM load, leaving most of the SM's FP32 pipelines idle. The tiled kernel does reduce effective DRAM pressure (1024 → 128 loads per element), but with such a small tile the arithmetic units spend most of their time waiting. A larger tile (T = 32) would push utilization much higher and likely yield a 3–5× speedup over naive. The result confirms the tiling concept works — shared memory reuse is real — but T = 8 is too small to close the gap to peak on a T4.
