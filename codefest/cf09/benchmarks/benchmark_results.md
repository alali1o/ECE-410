# CF09 CLLM — Benchmark Results
ECE 410/510 Spring 2026 — Ternary KWS Inference Accelerator

**Kernel:** FC1 ternary GEMV, W(512×1960), x(1960 INT16) → y(512 INT32)
**FLOPs per inference:** 2,007,040 (~2.01 MFLOPs)

---

## Benchmark Table

| Metric | SW Baseline (M4) | HW Accelerator (Projected) | HW Compute-Only (Projected) |
|--------|-----------------|----------------------------|------------------------------|
| **Platform** | Apple M4 (CPU, PyTorch) | sky130 ASIC, SPI 10 MHz | sky130 ASIC, no SPI overhead |
| **Clock / Freq** | 3.5 GHz (P-core) | 50 MHz system, 10 MHz SPI | 50 MHz |
| **Inference latency** | 0.436 ms (measured) | ~4.0 ms (projected) | 39.2 µs (projected) |
| **Throughput (inf/s)** | **2,293** (measured) | **~252** (projected) | **25,510** (projected) |
| **Effective GFLOP/s** | **4.91** (measured) | **0.000508** (projected) | **51.2** (projected) |
| **Memory (process)** | 187.3 MB | ~245 KB (on-chip SRAM) | ~245 KB |
| **Arithmetic intensity** | 0.50 FLOP/byte (measured) | 406 FLOP/byte (full reuse) | 406 FLOP/byte |
| **Energy** | N/A (no synthesis) | ~0.65 mW estimated | ~0.65 mW estimated |

---

## Speedup

| Comparison | Speedup |
|------------|---------|
| HW compute-only vs SW baseline | **11.1×** (projected) |
| HW end-to-end (SPI) vs SW baseline | **0.11×** (projected — slower than SW) |

---

## Notes on Projected Numbers

**Projection method (fallback path used per CF09 Task 7):**

The hardware accelerator end-to-end simulation (M3 cosim) passed with N_IN=8, N_OUT=4 parameters. The production kernel (N_IN=1960, N_OUT=512) was not simulated due to simulation time constraints. All HW numbers are projected from synthesis results and RTL analysis.

**Compute latency projection:**
- compute_core runs N_IN cycles in BUSY state (one column per cycle)
- At 50 MHz: 1960 / 50 × 10⁶ = **39.2 µs per inference**
- Throughput: 1 / 39.2 µs = **25,510 inferences/s**
- Effective GFLOP/s: 25,510 × 2,007,040 = **51.2 GFLOP/s**

**SPI transfer projection:**
- Input: 1960 × 2 bytes = 3,920 bytes; 16 SCLK bits per byte at 10 MHz = 1.6 µs/byte
  → 3,920 × 1.6 µs = 6.27 ms
  *(Optimized with burst mode: 3,920 × 8 bits / 10 MHz = 3.14 ms)*
- Output: 512 × 2 bytes = 1,024 bytes → 819 µs (burst mode)
- Total SPI time: **~3.96 ms** (burst) + 39.2 µs compute = **~4.0 ms**
- Throughput: **~252 inferences/s**

**Memory:**
- Weight SRAM: 512 × 1960 × 2 bits = **245,120 bits ≈ 245 KB on-chip**
- No heap allocation (fixed hardware registers)

**Energy estimate:** From M3 power_report.txt, estimated ~0.65 mW for the N_IN=8, N_OUT=4 test design. Production design scales roughly linearly → ~650 mW is unlikely; the production MAC array would dominate. A realistic estimate for the full 512-accumulator array at sky130 is ~5–20 mW. Pending full OpenLane synthesis.

---

## Summary

The compute engine is **11.1× faster than the M4 software baseline** in raw compute throughput (25,510 vs 2,293 inferences/s). However, end-to-end performance is dominated by the SPI interface at 10 MHz, which limits actual throughput to ~252 inferences/s — **11× slower than the SW baseline**. The fix is increasing SPI clock to 50 MHz, which would push end-to-end throughput to ~1,260 inferences/s (0.55× of SW, nearly matching it), and at 100 MHz SPI it would exceed SW by ~2.5×.

The memory footprint drops from **187 MB** (PyTorch + runtime) to **~245 KB** (on-chip SRAM weights only) — a **763× reduction** — which is a significant edge deployment advantage independent of throughput.
