# ResNet-18 Profiling Analysis
**ECE 410/510 Spring 2026 — Codefest 1 (CLLM)**

Model: ResNet-18 | Batch size: 1 | Input: 3×224×224 | Precision: FP32

---

## Task 3 — Top-5 Layers by MAC Count

| Rank | Layer Name               | Input Shape         | Output Shape        | MACs        | Parameters |
|------|--------------------------|---------------------|---------------------|-------------|------------|
| 1    | conv1 (initial 7×7)      | [1, 3, 224, 224]    | [1, 64, 112, 112]   | 118,013,952 | 9,408      |
| 2    | layer1.0.conv1 (3×3)     | [1, 64, 56, 56]     | [1, 64, 56, 56]     | 115,605,504 | 36,864     |
| 3    | layer2.0.conv2 (3×3)     | [1, 128, 28, 28]    | [1, 128, 28, 28]    | 115,605,504 | 147,456    |
| 4    | layer3.1.conv2 (3×3)     | [1, 256, 14, 14]    | [1, 256, 14, 14]    | 115,605,504 | 589,824    |
| 5    | layer4.1.conv2 (3×3)     | [1, 512, 7, 7]      | [1, 512, 7, 7]      | 115,605,504 | 2,359,296  |

> Note: Many 3×3 conv layers share the same MAC count (115,605,504). Ranks 2–5 are a representative
> sample covering all four layer groups (layer1–layer4). The initial 7×7 `conv1` is the single
> highest-MAC layer.

---

## Task 4 — Arithmetic Intensity of the Most MAC-Intensive Layer

**Layer:** `conv1` — the initial 7×7 convolution
**Config:** kernel=7×7, in\_channels=3, out\_channels=64, stride=2, padding=3

### Step 1: Compute FLOPs
```
FLOPs = 2 × MACs = 2 × 118,013,952 = 236,027,904 FLOPs
```

### Step 2: Weight Memory
Weight tensor shape: [64, 3, 7, 7] = 64 × 3 × 7 × 7 = 9,408 elements
```
Weight bytes = 9,408 × 4 bytes/element (FP32) = 37,632 bytes
```

### Step 3: Activation Memory (input + output, no reuse assumed)
- Input activation:  [1, 3, 224, 224]  → 3 × 224 × 224 = 150,528 elements
- Output activation: [1, 64, 112, 112] → 64 × 112 × 112 = 802,816 elements

```
Input bytes  = 150,528 × 4 = 602,112 bytes
Output bytes = 802,816 × 4 = 3,211,264 bytes
Activation bytes = 602,112 + 3,211,264 = 3,813,376 bytes
```

### Step 4: Total Bytes Loaded from DRAM
```
Total bytes = Weight bytes + Activation bytes
            = 37,632 + 3,813,376
            = 3,851,008 bytes
```

### Step 5: Arithmetic Intensity
```
Arithmetic Intensity = FLOPs / Total bytes
                     = 236,027,904 / 3,851,008
                     ≈ 61.29 FLOP/byte
```

### Summary
| Quantity              | Value                  |
|-----------------------|------------------------|
| MACs                  | 118,013,952            |
| FLOPs (2 × MACs)      | 236,027,904            |
| Weight memory         | 37,632 bytes (36.75 KB)|
| Input activation      | 602,112 bytes (588 KB) |
| Output activation     | 3,211,264 bytes (3.06 MB)|
| Total DRAM bytes      | 3,851,008 bytes (3.67 MB)|
| **Arithmetic Intensity** | **≈ 61.29 FLOP/byte** |

> This intensity (~61 FLOP/byte) is relatively low compared to peak compute-to-bandwidth ratios
> on modern GPUs (~200–1000 FLOP/byte). The `conv1` layer is therefore **memory-bandwidth bound**
> under the no-reuse assumption, which motivates techniques like tiling and data reuse in hardware
> accelerator design.
