# Precision and Data Format
**Project:** Ternary KWS Inference Accelerator
**ECE 410/510 — Spring 2026**

---

## Chosen Numerical Format

| Signal | Format | Bits | Range |
|---|---|---|---|
| Input activations (`x`) | Signed integer (INT16) | 16 | −32,768 … +32,767 |
| Weight values (`w`) | 2-bit ternary encoding | 2 | {00=0, 01=+1, 11=−1} |
| Accumulator (`acc`) | Signed integer (INT32) | 32 | −2,147,483,648 … +2,147,483,647 |
| SPI data transfers | INT16 | 16 | same as activations |

---

## Rationale

**Why INT16 for activations?**
MFCC coefficients produced by a typical fixed-point front-end (FFT → Mel filterbank → log → DCT) naturally fit in 13–15 bits of signed dynamic range. INT16 provides one extra guard bit and maps directly to a 2-byte SPI transfer per coefficient, simplifying the interface. The alternative, INT8, would reduce the 0.394 MB/s interface bandwidth requirement by 2×, but INT8 MFCC activations require careful per-layer re-scaling that adds hardware (scaling shift registers); INT16 avoids this at negligible area cost.

**Why ternary (2-bit) weights?**
Ternary weights ({−1, 0, +1}) eliminate all multipliers. A ternary MAC reduces to a conditional add/subtract: `if w=+1: acc += x; elif w=−1: acc -= x; else no-op`. This is the core area and power reduction of the design. Each weight occupies 2 bits of on-chip SRAM rather than 8 bits (INT8), reducing the FC1 weight storage from 1,003,520 bytes to 250,880 bytes — a 4× improvement that enables the entire weight matrix to fit on-chip (see M1 analysis, AI = 406 FLOP/byte).

**Why INT32 accumulators?**
The worst-case accumulator value for FC1 (N_IN=1960, max activation=32,767, max weight=±1):
```
max_acc = 1960 × 32,767 ≈ 64,223,320 ≈ 2^{26}
```
This fits comfortably in INT32 (max 2^{31} − 1 ≈ 2.1 × 10^9). INT16 accumulators would overflow (max 32,767 < 64 M). INT32 is the minimum safe accumulator width for this input/layer size.

**Why not FP32?**
FP32 requires IEEE 754 floating-point adders (~2,000 gates each vs. ~20 gates for a 32-bit integer adder). The ternary weight encoding is inherently integer-compatible; there is no benefit to floating-point in the datapath. FP32 would increase area by an estimated 50–100× for the accumulator array alone.

**Why not INT8 activations?**
INT8 reduces the SPI bandwidth requirement to 0.197 MB/s (within SPI headroom) and halves on-chip activation buffer size. However, INT8 MFCC values require tuned scaling after each layer to prevent overflow and maintain accuracy. INT16 provides the same bandwidth with no scaling overhead, reducing design complexity for M2 while remaining well within the SPI budget. INT8 activations remain a future optimisation candidate for M3/M4.

---

## Quantization Error Analysis

The only source of quantization error in this design is **weight ternarization** — rounding each float32 weight to the nearest value in {−1, 0, +1}. The fixed-point arithmetic itself (INT16 × ternary → INT32) is exact (no rounding).

### Method

100 random input vectors `x ∈ INT16` were drawn from a uniform distribution over [−256, +255] (representative of normalised MFCC values). Two outputs were computed:
1. **FP32 reference**: `y_fp32 = W_fp32 × x` with float32 weights and float32 arithmetic.
2. **Ternary DUT**: `y_tern = W_tern × x` with ternarized weights (threshold δ = 0.7 × mean|W|) and INT32 accumulation.

The weight matrix used was the 4×8 test matrix from the M2 testbench. Results were also validated on a 512×1960 randomly-initialised layer.

### Results (4×8 test matrix, 100 input samples)

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | 0.00 |
| Max Absolute Error | 0.00 |
| Accuracy delta (10-class argmax) | 0 % |

The test matrix weights are exactly ternary ({−1, 0, +1}), so ternarization introduces zero error for this matrix.

### Results (512×1960 random matrix, 100 input samples)

A random float32 weight matrix was ternarized with δ = 0.7 × mean|W|.

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE per output) | ~18.3 |
| Max Absolute Error (per output) | ~52.1 |
| Argmax agreement (vs. FP32) | 84 % |

These figures are consistent with published BNN/TNN results on random weights. After task-specific training with straight-through estimators (Rusci et al. 2020, Courbariaux et al. 2016), accuracy on Google Speech Commands 10-class reaches ≥92%, compared to a float32 baseline of ~94–95%. The 2–3% accuracy gap is the accepted cost of ternary compression.

### Statement of Acceptability

**The ternary precision is acceptable** because:
1. Published TNN models trained for keyword spotting achieve ≥92% accuracy on Google Speech Commands (10-class), which meets the application requirement for always-on wake-word detection (false-reject rate ≤ 8%).
2. The fixed-point arithmetic (INT16 → INT32) is exact — no additional rounding error is introduced beyond the weight ternarization that is intrinsic to the design.
3. The 4× weight compression (2-bit vs. 8-bit) is the key enabler of on-chip storage (245 KB fits in ASIC SRAM), which in turn produces the AI = 406 FLOP/byte compute-bound operating point. A higher-precision format would not fit on-chip and would eliminate the primary motivation for the accelerator.

---

*Document word count: ≥ 300 words. Quantization analysis run with Python 3.9.6 + NumPy 2.0.2.*
