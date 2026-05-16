# CF07 CMAN — Sparsity Breakeven Analysis
**ECE 410/510 Spring 2026 | N = 512**

---

## Task 1 — Four Expressions

Let **N = 512** and **s** = fraction of zero weights (sparsity).

### (a) Dense MVM Compute — FLOPs

Each of the N² weight–activation pairs requires one multiply and one add (1 MAC = 2 FLOPs):

$$\text{Dense FLOPs} = 2N^2 = 2 \times 512^2 = \mathbf{524{,}288 \text{ FLOPs}}$$

### (b) Dense Memory Bytes

Every weight is FP32 (4 bytes). The full N×N matrix is loaded:

$$\text{Dense bytes} = 4N^2 = 4 \times 512^2 = \mathbf{1{,}048{,}576 \text{ bytes (1 MB)}}$$

### (c) Sparse Compute — FLOPs (function of s)

Only the (1 − s) fraction of weights are nonzero, so only those MACs execute:

$$\text{Sparse FLOPs} = 2N^2(1 - s)$$

At s = 0.9: Sparse FLOPs = 2 × 262,144 × 0.1 = **52,429 FLOPs**

### (d) Sparse Memory Bytes — CSR (function of s)

CSR uses three arrays:
| Array | Elements | Bytes each | Total |
|---|---|---|---|
| **values** (FP32 weights) | N²(1−s) | 4 | 4N²(1−s) |
| **col_idx** (INT32 column indices) | N²(1−s) | 4 | 4N²(1−s) |
| **row_ptr** (INT32 row pointers) | N+1 | 4 | 4(N+1) |

$$\boxed{\text{Sparse bytes} = 8N^2(1-s) + 4(N+1)}$$

At s = 0.9: 8 × 262,144 × 0.1 + 4 × 513 = 209,715 + 2,052 = **211,767 bytes**

---

## Task 2 — FLOPs Speedup and 2× Breakeven

$$\text{Speedup}_{\text{FLOPs}} = \frac{\text{Dense FLOPs}}{\text{Sparse FLOPs}} = \frac{2N^2}{2N^2(1-s)} = \frac{1}{1-s}$$

Set equal to 2:

$$\frac{1}{1-s} = 2 \implies 1-s = 0.5 \implies \boxed{s = 0.5 \text{ (50\% sparsity)}}$$

**Interpretation:** You need at least 50% of weights to be zero before sparse computation saves any FLOPs over dense computation. Modern pruned neural networks routinely exceed 90%, so the FLOPs argument is strong in practice.

---

## Task 3 — Memory Breakeven Sparsity

Set sparse bytes = dense bytes and solve for s:

$$8N^2(1-s) + 4(N+1) = 4N^2$$

$$8N^2 - 8N^2 s + 4N + 4 = 4N^2$$

$$8N^2 s = 8N^2 - 4N^2 + 4N + 4 = 4N^2 + 4N + 4$$

$$\boxed{s_{\text{mem}} = \frac{4N^2 + 4N + 4}{8N^2} = \frac{1}{2} + \frac{1}{2N} + \frac{1}{2N^2}}$$

**For N = 512:**

$$s_{\text{mem}} = 0.5 + \frac{1}{1024} + \frac{1}{524288} \approx 0.5 + 0.000977 + 0.0000019 \approx \mathbf{50.1\%}$$

**Interpretation:** The memory breakeven is almost exactly 50%, slightly above the FLOPs breakeven. The small gap (0.1%) comes from the row pointer array overhead — those 4(N+1) = 2,052 bytes of mandatory bookkeeping tilt the scale very slightly. Above ~50.1% sparsity, CSR uses less memory than a dense matrix.

---

## Task 4 — End-to-End Speedup at s = 0.9, N = 512

**System: memory-bandwidth-limited, BW = 320 GB/s**

For a memory-bound system, execution time is proportional to bytes transferred:

$$T = \frac{\text{Bytes}}{BW}$$

**Dense bytes transferred:**
$$B_{\text{dense}} = 4N^2 = 4 \times 262{,}144 = 1{,}048{,}576 \text{ bytes}$$

**Sparse bytes transferred at s = 0.9:**
$$B_{\text{sparse}} = 8N^2(1-s) + 4(N+1) = 8 \times 262{,}144 \times 0.1 + 4 \times 513$$
$$= 209{,}715.2 + 2{,}052 = 211{,}767.2 \text{ bytes}$$

**End-to-end speedup:**
$$\text{Speedup} = \frac{T_{\text{dense}}}{T_{\text{sparse}}} = \frac{B_{\text{dense}}}{B_{\text{sparse}}} = \frac{1{,}048{,}576}{211{,}767.2} \approx \mathbf{4.95\times}$$

**Interpretation:** At 90% sparsity, CSR almost hits 5× speedup in a memory-bound setting — not the full 10× you might expect from the FLOPs ratio, because the index arrays eat into the savings. The row pointer and column index arrays each consume as many bytes as the nonzero values themselves, cutting the bandwidth savings roughly in half compared to FLOPs savings. This is the key tax of CSR: you skip the compute but you still pay for the metadata.
