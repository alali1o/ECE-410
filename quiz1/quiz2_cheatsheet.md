# ECE 410/510 — Quiz 2 Cheat Sheet
**Bao Nguyen | Portland State University | Spring 2026**
*Covers: Week 5+ material (TPUs, Transformers, Neuromorphic, CUDA MLP)*

---

## §1 — GPU Power & Data Center Trends

| GPU | TDP | Year |
|-----|-----|------|
| A100 | 400 W | 2020 |
| H100 | 700 W | 2022 |
| B200 | 1000 W | 2024 |
| B300 | 1400 W | 2025 |
| Vera Rubin | 2300 W | 2026 |

**PUE** (Power Usage Effectiveness) = Total Facility Power / IT Equipment Power
- Ideal PUE = 1.0
- Air-cooled = **1.80** → 44% overhead for cooling/power delivery
- Liquid-cooled = **1.15** → 13% overhead

**Scale**: hyperscale data centers = hundreds of MW; global AI infra = multi-GW

---

## §2 — CPU vs GPU vs TPU Comparison

| Dimension | CPU | GPU | TPU |
|-----------|-----|-----|-----|
| Design goal | Low latency, scalar | Throughput, SIMT parallel | Matrix multiply, systolic |
| Parallelism | Few cores, OOO | Thousands of CUDA cores | Systolic array (256×256) |
| MACs/cycle | Few | Tens of thousands | Up to 128K (MXU) |
| Memory | L1/L2/L3 cache | HBM + SMEM | HBM + on-chip SRAM |
| Predictions/sec | 5,482 | 13,194 | **225,000** |
| Perf/Watt | 1× | 2.9× | **83×** |

---

## §3 — Systolic Array (How TPU Computes)

A **grid of Processing Elements (PEs)** each doing one MAC. Data flows rhythmically — one operand per direction.

**TPU MXU**: 256×256 = **65,536 MACs per cycle**

**Timing for N×N matrix multiply**:
```
Fill pipeline:   N cycles       (first result comes out after N cycles)
Output phase:    N cycles       (N results stream out)
Drain pipeline:  N-2 cycles
Total:           3N - 2 cycles
Steady state:    1 result/cycle after fill
```

---

## §4 — Three Dataflow Types

| Dataflow | What stays fixed | Who moves | Used by |
|----------|-----------------|-----------|---------|
| **Weight Stationary** | Weights in PEs | Activations stream left→right, partial sums accumulate | Google TPU v1–v4 |
| **Output Stationary** | Partial sums in PEs | Weights + activations flow through | ShiDianNao |
| **Row Stationary** | A row of filter + its reuse | Maximizes all-data reuse simultaneously | MIT Eyeriss |

**Row Stationary wins on energy**: 10× fewer DRAM accesses than weight-stationary; minimizes all data movement simultaneously — but harder to implement.

---

## §5 — TPU Architecture & Roadmap

**TPU v1 key numbers**:
- 256×256 MXU = 65,536 MACs/cycle
- 700 MHz → 92 TOPS (INT8)
- Predictions/sec: **225,000** (vs CPU 5,482, GPU 13,194)
- Perf/Watt: **83× CPU**, 29× GPU

**TPU Roadmap**:
| Version | Year | MXU | BF16 TFLOPS | HBM BW |
|---------|------|-----|-------------|--------|
| v4 | 2022 | 128×128 | ~275 | ~1.2 TB/s |
| v7 Ironwood | 2025 | 256×256 | ~2300 | **7400 GB/s** |

**v7 Ironwood pod**: 9,216 chips; FP8 native; designed for inference at scale.

---

## §6 — BF16 (Brain Float 16)

```
FP32:  1 sign | 8 exponent | 23 mantissa  (32 bits)
FP16:  1 sign | 5 exponent | 10 mantissa  (16 bits)  ← RISK: narrow exponent range
BF16:  1 sign | 8 exponent | 7  mantissa  (16 bits)  ← SAFE: same range as FP32
```

**Key properties**:
- **Same dynamic range as FP32** — avoids overflow/underflow during training
- Developed by **Google Brain** for TPU training
- **Trivial conversion**: FP32 ↔ BF16 = just add/drop the last 16 mantissa bits
- Trade-off: less precision (7-bit mantissa) but rarely matters for training

---

## §7 — Transformers: NON-Recurrent (Critical Concept)

**"Attention Is All You Need"** (Vaswani et al., 2017)

### Why NOT RNN/LSTM?
| | RNN/LSTM | Transformer |
|--|----------|-------------|
| Processing | Sequential — token by token | **Parallel — all tokens at once** |
| Memory | Hidden state carries context forward | **No hidden state** — all context in attention |
| Training | Can't parallelize over time | Fully parallelizable |
| Long-range | Vanishing gradient problem | Direct attention to any token |

**The key insight**: Replace recurrence with **self-attention**. Positional encodings (sine/cosine) replace time ordering — the model doesn't need to process sequentially.

---

## §8 — Self-Attention: Q, K, V

```
Q = "What am I looking for?"    (Query)
K = "What do I offer?"          (Key)
V = "What do I contribute?"     (Value)

Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

- **√d_k scaling**: prevents dot products from growing too large (and softmax saturating)
- **Multi-Head Attention**: run H parallel attention heads, concatenate → project
  - Each head learns different relationship types
- **Masked** attention in decoder: prevent looking at future tokens during training

---

## §9 — Transformer Architecture

```
Input → [Embedding + Positional Encoding]
         ↓
    ┌── ENCODER (×N) ──────────────────────┐
    │  Multi-Head Self-Attention            │
    │  Add & Norm (residual connection)     │
    │  Feed-Forward Network (FFN)           │
    │  Add & Norm                           │
    └───────────────────────────────────────┘
         ↓
    ┌── DECODER (×N) ──────────────────────┐
    │  Masked Multi-Head Self-Attention     │
    │  Add & Norm                           │
    │  Cross-Attention (attends to encoder) │
    │  Add & Norm                           │
    │  FFN + Add & Norm                     │
    └───────────────────────────────────────┘
         ↓
    Linear + Softmax → Output probabilities
```

**Operations summary**:
| Operation | Notes |
|-----------|-------|
| MatMul | Dominant operation — maps to GEMM |
| Softmax | s(xᵢ) = eˣⁱ / Σeˣʲ |
| Layer Norm | Normalize over feature dim (not batch) |
| FFN | Two linear layers + ReLU |
| Residual/Skip | x + sublayer(x) — stabilizes training |
| Positional Encoding | sin/cos at different frequencies |

---

## §10 — NVIDIA Transformer Engine (H100)

- Library for running transformers efficiently on H100 tensor cores
- **Dynamically determines** optimal precision (FP16 or FP8) **per layer** each forward pass
- Uses **per-layer statistics** (adaptive range tracking) to choose without accuracy loss
- FP8 = 2× memory reduction vs FP16 for inference
- Tight integration with H100's 4th-gen tensor cores

---

## §11 — Blackwell B200 Architecture

| Spec | Value |
|------|-------|
| Transistors | 208 billion (dual-die package) |
| HBM3e | 192 GB |
| Memory BW | 8 TB/s |
| Peak FP4 (sparse) | 20 PFLOPS |
| NVLink | 5.0 — 1.8 TB/s |
| TDP | 1000 W |
| New features | MXFP4, 2nd-gen Transformer Engine, FP6 |

**MXFP4**: Micro-scaled FP4 — 16 FP4 values share one FP8 block scale (same idea as NVFP4 from earlier).

---

## §12 — Neuromorphic Transformer

Replaces **MAC** (multiply-accumulate) with **AAC** (AND-accumulate):
- Q/K/V matrices become **binary** (spikes: 0 or 1)
- AND replaces multiply; accumulate with integer addition
- Eliminates: softmax, scaling by √d_k, matrix transposition

**Impact**:
- 99.96% reduction in multiplications: **116M → 4,900** multiplications
- Based on **Spiking Neural Networks (SNN)** — event-driven, sparse activity
- Human brain analogy: ~20W, spike-based, temporal encoding

| | ANN | SNN |
|--|-----|-----|
| Activation | Continuous (float) | Binary spike (0 or 1) |
| Computation | Dense MACs | Sparse AND-accumulate |
| Energy | High | Very low |

---

## §13 — No Free Lunch Theorem

> No algorithm consistently outperforms all others across ALL problems.

**Hardware implication**: This is WHY specialized hardware excels. A GPU beats CPU for GEMM; a TPU beats GPU for matrix multiply; a neuromorphic chip beats both for sparse spiking workloads. No universal winner — match hardware to workload.

---

## §14 — CUDA MLP Mapping

**Parallelism types for MLP on GPU**:
| Type | What is parallelized | When to use |
|------|---------------------|-------------|
| Data parallelism | Multiple training examples simultaneously | Most common — batch dimension |
| Model parallelism | Partition network across GPUs | Large models that don't fit in one GPU |
| Layer-wise | Neurons in same layer are independent | Per-layer kernel launch |

**CUDA kernel design for MLP**:
- Separate kernel per layer → each layer gets optimal thread config
- Layer 1: `threadsPerBlock(8, 5)` — 8 batches × 5 hidden neurons
- Layer 2: `threadsPerBlock(16, 1)` — 16 batches × 1 output neuron
- Key trade-off: **kernel launch overhead** vs **register pressure** (too many threads per block = register spills)
- Operations with same data/parallelism pattern can be fused

**Memory flow**:
```
cudaMalloc      → allocate device memory
cudaMemcpy H→D → copy input/weights to GPU
kernel<<<M,T>>> → launch M blocks, T threads/block
cudaMemcpy D→H → copy results back to CPU
cudaFree        → release device memory
```

---

## §15 — Quick-Reference Numbers

| Fact | Value |
|------|-------|
| TPU MXU size | 256×256 = 65,536 MACs/cycle |
| TPU predictions/sec | 225,000 (vs GPU 13,194, CPU 5,482) |
| TPU perf/watt vs CPU | 83× |
| Systolic N×N total cycles | 3N − 2 |
| BF16 exponent bits | 8 (same as FP32) |
| BF16 mantissa bits | 7 (vs FP32's 23) |
| B200 peak FP4 sparse | 20 PFLOPS |
| B200 memory BW | 8 TB/s |
| Neuromorphic mult reduction | 99.96% (116M → 4,900) |
| Air-cooled data center PUE | 1.80 |
| Liquid-cooled PUE | 1.15 |
| H100 TDP | 700 W |
| B200 TDP | 1,000 W |

---

## §16 — Last-Minute Checklist

- [ ] Can I explain why transformers are NOT recurrent (and what replaced recurrence)?
- [ ] Can I draw the Q/K/V attention formula from memory?
- [ ] Can I explain weight-stationary vs row-stationary dataflow and which wins on energy?
- [ ] Do I know the systolic array timing: 3N−2 cycles for N×N multiply?
- [ ] Can I explain why BF16 is safer than FP16 for training?
- [ ] Do I know the TPU perf/watt advantage: 83× over CPU?
- [ ] Can I explain what the Neuromorphic Transformer replaces MACs with?
- [ ] Do I know what the No Free Lunch theorem means for hardware specialization?
- [ ] Can I set up thread dimensions for a CUDA MLP layer?
- [ ] Do I know B200's key specs (208B transistors, 8 TB/s BW, 20 PFLOPS FP4)?
