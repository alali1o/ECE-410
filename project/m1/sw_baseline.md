# Software Baseline Benchmark
**Project:** Ternary Neural Network Inference Accelerator — Keyword Spotting  
**ECE 410/510 Spring 2026**

---

## Platform and Configuration

| Parameter | Value |
|---|---|
| **Machine** | Apple MacBook Air (Mac16,12) |
| **Chip** | Apple M4 (ARM64, 10-core: 4P + 6E) |
| **Memory** | 16 GB unified (120 GB/s bandwidth, Apple spec) |
| **OS** | macOS 15 (Darwin 25.3.0, arm64) |
| **Python** | 3.9.6 |
| **PyTorch** | 2.8.0 |
| **CUDA** | N/A (CPU inference only) |
| **Batch size** | 1 (single 1-second audio window) |
| **Threads** | PyTorch default (all P-cores via OpenMP) |

---

## Model Under Test: TernaryKWS

A 3-layer fully-connected network simulating ternary {−1, 0, +1} weight inference:

```
Input:  1960 features  (49 frames × 40 MFCC coefficients, 1-second window)
FC1:    1960 → 512   TernaryLinear + BatchNorm + ReLU   ← dominant kernel
FC2:     512 → 128   TernaryLinear + BatchNorm + ReLU
FC3:     128 →  10   TernaryLinear  (10-class: Google Speech Commands)
```

Total parameters: **1,072,266**  
FP32 weight storage (simulation): **4,096 KB**  
Ternary weight storage (2 bits/weight, hardware target): **~245 KB** (FC1 only)

Ternarization is performed on-the-fly in software (`w.clamp(-δ, δ).sign()`, δ = 0.7·mean|w|). This is a simulation artifact; in hardware the weights are pre-quantized and stored in SRAM.

---

## Execution Time

Measured with `time.perf_counter()`, `torch.no_grad()`, 100 forward passes after 5 warm-up runs.

| Statistic | Value |
|---|---|
| **Median latency** | **0.4362 ms** |
| Mean latency | 0.4661 ms |
| Min latency | 0.3751 ms |
| Max latency | 0.8080 ms |

---

## Throughput

| Metric | Value |
|---|---|
| **Inferences / second** | **2,293 samples/sec** |
| Total FLOPs (full network) | 2,140,672 (~2.14 M) |
| Effective GFLOP/s | **4.91 GFLOP/s** |

FLOPs breakdown:

| Layer | FLOPs | % of total |
|---|---|---|
| FC1 (1960 × 512) | 2 × 1960 × 512 = 2,007,040 | **93.8%** |
| FC2 (512 × 128) | 2 × 512 × 128 = 131,072 | 6.1% |
| FC3 (128 × 10) | 2 × 128 × 10 = 2,560 | 0.1% |
| **Total** | **2,140,672** | 100% |

---

## Memory Usage

| Metric | Value |
|---|---|
| **Peak process RSS** | **187.3 MB** |
| Model weight memory (FP32) | 4,096 KB |
| Activation buffers (batch=1) | < 1 KB |

The large RSS is dominated by the PyTorch runtime and shared libraries, not the model weights.

---

## Reproducibility

To reproduce these numbers on any Apple Silicon Mac:

```bash
pip install torch==2.8.0
python3 - << 'EOF'
import torch, torch.nn as nn, time, statistics

class TernaryLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.randn(o, i))
        self.b = nn.Parameter(torch.zeros(o))
    def forward(self, x):
        d = 0.7 * self.w.abs().mean()
        return nn.functional.linear(x, self.w.clamp(-d, d).sign(), self.b)

class TernaryKWS(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = TernaryLinear(1960, 512)
        self.fc2 = TernaryLinear(512, 128)
        self.fc3 = TernaryLinear(128, 10)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

torch.manual_seed(0)
model = TernaryKWS().eval()
x = torch.randn(1, 1960)
with torch.no_grad():
    for _ in range(5): model(x)  # warm-up
    times = []; [times.append(time.perf_counter()) or model(x) for _ in range(100)]
print(f"Median: {statistics.median(times)*1000:.4f} ms")
EOF
```
