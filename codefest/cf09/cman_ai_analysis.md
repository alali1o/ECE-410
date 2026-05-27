# CF09 CMAN — Arithmetic Intensity from First Principles
ECE 410/510 Spring 2026

---

## Task 1 — Dominant Kernel

**Kernel:** Ternary matrix-vector multiply (TMVM) — FC1 layer

**Dimensions:**
- Weight matrix W: 512 rows × 1960 columns
- Input vector x: 1960 elements
- Output vector y: 512 elements

**Data types:**
- W: ternary {−1, 0, +1}, stored as 2-bit encoding per weight (2'b00=0, 2'b01=+1, 2'b11=−1)
- x: signed INT16 (16-bit, 2 bytes per element)
- y: signed INT32 accumulator (32-bit, 4 bytes per element; top-1 truncated to INT16 for SPI output)

**Reuse pattern:** Dense matrix-vector multiply (GEMV). Each input element x[j] is reused by all 512 output neurons → 512× reuse per input element. Weights are used once per inference but can be held on-chip across inferences (weight-stationary reuse).

---

## Task 2 — FLOPs Count

Each output element y[i] requires 1960 multiply-accumulate operations (one per input element).

Total MACs = N_OUT × N_IN = 512 × 1960 = **1,003,520 MACs**

Total FLOPs = 2 × MACs = 2 × 1,003,520 = **2,007,040 FLOPs ≈ 2.01 MFLOPs**

(Note: ternary multiply is a conditional add/subtract/noop — no actual multiplier — but we count as MAC×2 for AI convention.)

---

## Task 3 — Bytes Transferred

### Lower Bound — No Data Reuse

Everything loaded from off-chip memory for each inference.

| Tensor | Size | Bytes |
|--------|------|-------|
| W (weights) | 512 × 1960 × 2 bits | 1,003,520 bits = **250,880 bytes** |
| x (input)   | 1960 × 2 bytes (INT16) | **3,920 bytes** |
| y (output)  | 512 × 2 bytes (INT16 after truncation) | **1,024 bytes** |
| **Total**   | | **255,824 bytes** |

```
B_low = (N_OUT × N_IN × 2 bits / 8) + N_IN × 2 + N_OUT × 2
      = 250,880 + 3,920 + 1,024
      = 255,824 bytes
```

### Upper Bound — Perfect On-Chip Weight Reuse

Weights are preloaded into on-chip SRAM and stay there. Only input and output cross the SPI interface each inference.

| Tensor | Bytes |
|--------|-------|
| x (input)  | 1960 × 2 = **3,920 bytes** |
| y (output) | 512 × 2 = **1,024 bytes** |
| **Total**  | **4,944 bytes** |

```
B_high = N_IN × 2 + N_OUT × 2
       = 3,920 + 1,024
       = 4,944 bytes
```

---

## Task 4 — Arithmetic Intensity

```
AI_lower = FLOPs / B_low  = 2,007,040 / 255,824 ≈ 7.8 FLOP/byte
AI_upper = FLOPs / B_high = 2,007,040 / 4,944   ≈ 406 FLOP/byte
```

### Sky130 ASIC Roofline Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Clock (after timing fix) | 50 MHz | M3 critical path analysis |
| MACs per cycle | 512 (all N_OUT accumulators update in parallel) | compute_core.sv RTL |
| Peak compute | 512 × 50 MHz × 2 = **51.2 GFLOP/s** | RTL × clock |
| Off-chip BW (SPI 10 MHz) | 10 Mbit/s = **1.25 MB/s = 0.00125 GB/s** | M1 interface selection |
| Ridge point (SPI) | 51.2 / 0.00125 = **40,960 FLOP/byte** | BW ceiling ÷ compute ceiling |

### Roofline Position

Both AI bounds lie far below the ridge point (40,960 FLOP/byte):

| Bound | AI (FLOP/byte) | Attainable perf | Region |
|-------|----------------|-----------------|--------|
| Lower (no reuse) | 7.8 | 7.8 × 1.25 MB/s = **9.75 KFLOP/s** | Memory-bound (SPI) |
| Upper (full reuse) | 406 | 406 × 1.25 MB/s = **508 KFLOP/s** | Memory-bound (SPI) |

See `cman_roofline_sketch.png` for the sketch.

---

## Task 5 — Bottleneck and Improvement

**Current bottleneck: Hardware interface bandwidth (SPI at 10 MHz = 1.25 MB/s)**

The compute engine processes one full inference in N_IN = 1960 cycles at 50 MHz = **39.2 µs**. But transferring the input and output through SPI takes 4,944 bytes × 8 bits / 10 MHz = **3.96 ms** — 100× longer than the compute. Even with perfect weight reuse (AI = 406), the design is still deeply SPI-limited. The ridge point sits at 40,960 FLOP/byte, and our best AI of 406 is two orders of magnitude below it.

**Single highest-leverage change: Increase SPI clock from 10 MHz to 50 MHz**

This raises interface bandwidth from 1.25 MB/s to 6.25 MB/s (5× improvement), reducing SPI transfer time from 3.96 ms to 794 µs per inference and boosting end-to-end throughput from ~252 to ~1,260 inferences/s — approaching the M1 SW baseline of 2,293 inferences/s. No RTL changes are required; only the SPI clock divider on the host MCU changes. The sky130 spi_slave_ext module is already specified for ≤ 4× oversampling, so a 50 MHz SPI with a 200+ MHz system clock would require re-evaluation of the synchronizer chain.
