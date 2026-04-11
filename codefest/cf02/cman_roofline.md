# CMAN — Roofline Construction and Kernel Classification
**ECE 410/510 Spring 2026 — Codefest 2**

---

## Hardware Specification

| Parameter        | Value                              |
|------------------|------------------------------------|
| Peak Compute     | 10 TFLOPS = 10,000 GFLOPS (FP32)  |
| Peak DRAM BW     | 320 GB/s                           |
| Ridge Point      | 10,000 ÷ 320 = **31.25 FLOP/byte**|

---

## Task 1 — Roofline Diagram

See attached: `cman_roofline.png`

**How the roofline is built:**

The roofline has two segments on log-log axes:

- **Diagonal (memory-bound region):**
  Any kernel with AI < 31.25 FLOP/byte cannot feed the compute units fast enough.
  Its attainable performance is capped by bandwidth:
  ```
  Attainable performance = AI × 320 GB/s
  ```

- **Flat ceiling (compute-bound region):**
  Any kernel with AI > 31.25 FLOP/byte has enough data reuse to keep the
  compute units busy. Its attainable performance is capped by peak compute:
  ```
  Attainable performance = 10,000 GFLOPS
  ```

- **Ridge point:** where the two segments meet → **(31.25 FLOP/byte, 10,000 GFLOPS)**

---

## Task 2 — Kernel A: Dense GEMM (1024×1024, FP32)

**What it is:** Multiply two square FP32 matrices of size N×N, where N = 1024.

### Step 1 — Count FLOPs
Each output element C[i,j] requires N multiplications and N additions = 2N operations.
There are N² output elements total, so:
```
FLOPs = 2 × N³
      = 2 × 1024³
      = 2 × 1,073,741,824
      = 2,147,483,648 FLOPs  ≈ 2.15 GFLOPs
```

### Step 2 — Count Bytes Transferred
We load matrix A, load matrix B, and store matrix C — all from/to DRAM, no cache reuse.
Each element is FP32 = 4 bytes. Each matrix has N² = 1024² = 1,048,576 elements.
```
Bytes = (load A + load B + store C) × element size
      = 3 × N² × 4
      = 3 × 1,048,576 × 4
      = 12,582,912 bytes  ≈ 12.0 MB
```

### Step 3 — Arithmetic Intensity
```
AI = FLOPs ÷ Bytes
   = 2,147,483,648 ÷ 12,582,912
   = 170.67 FLOP/byte
```

### Step 4 — Classify and find attainable performance
```
AI = 170.67  >  Ridge point = 31.25  →  COMPUTE-BOUND

Attainable performance = min(AI × BW, Peak Compute)
                       = min(170.67 × 320, 10,000)
                       = min(54,614, 10,000)
                       = 10,000 GFLOPS  (hits the compute ceiling)
```

### Step 5 — Architectural Recommendation
GEMM is already at the compute ceiling — memory bandwidth is not the bottleneck.
**To improve performance: use lower precision (FP16 or INT8).**
This doubles or quadruples the effective TFLOPS on hardware that supports it
(e.g., tensor cores), allowing the same silicon area to do more work per cycle.

---

## Task 3 — Kernel B: Vector Addition (N = 4,194,304, FP32)

**What it is:** Add two FP32 vectors element-wise: C[i] = A[i] + B[i], for 4,194,304 elements.

### Step 1 — Count FLOPs
One addition per element:
```
FLOPs = N = 4,194,304 FLOPs  ≈ 4.19 MFLOPs
```

### Step 2 — Count Bytes Transferred
We load vector A, load vector B, and store result C — all from/to DRAM, no cache reuse.
Each element is FP32 = 4 bytes.
```
Bytes = (load A + load B + store C) × element size
      = 3 × N × 4
      = 3 × 4,194,304 × 4
      = 50,331,648 bytes  ≈ 48.0 MB
```

### Step 3 — Arithmetic Intensity
```
AI = FLOPs ÷ Bytes
   = 4,194,304 ÷ 50,331,648
   = 0.0833 FLOP/byte
```

### Step 4 — Classify and find attainable performance
```
AI = 0.0833  <  Ridge point = 31.25  →  MEMORY-BOUND

Attainable performance = min(AI × BW, Peak Compute)
                       = min(0.0833 × 320, 10,000)
                       = min(26.67, 10,000)
                       = 26.67 GFLOPS  (far below the compute ceiling)
```

### Step 5 — Architectural Recommendation
Vector addition does only 1 FLOP per 12 bytes of data — no amount of added compute
units will help. **To improve performance: increase memory bandwidth.**
This can be done by switching to HBM memory, or by fusing multiple element-wise
operations into a single kernel pass to reduce total DRAM traffic.

---

## Summary Table

| Kernel         | FLOPs         | Bytes      | AI (FLOP/byte) | Bound   | Attainable     |
|----------------|---------------|------------|----------------|---------|----------------|
| A — GEMM 1024² | 2,147,483,648 | 12,582,912 | **170.67**     | Compute | 10,000 GFLOPS  |
| B — Vec Add 4M | 4,194,304     | 50,331,648 | **0.0833**     | Memory  | 26.67 GFLOPS   |
