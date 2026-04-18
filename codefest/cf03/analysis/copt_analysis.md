# CF03 COPT — GPU Neural Network Forward Pass
ECE 410/510 Spring 2026 — Codefest 3 Optional Challenge

GPU: NVIDIA T4 | Peak FP32: 8,100 GFLOPS | Peak BW: 300 GB/s | Ridge: 27 FLOP/byte  
Network: FC(40→256) ReLU → FC(256→128) ReLU → FC(128→10) Softmax | Batch = 256, FP32

---

## Architecture

The forward pass mirrors the project's keyword-spotting ternary NN.
Each fully-connected layer is implemented as a single CUDA kernel (`fc_relu`) that:
1. Performs tiled GEMM (tile = 16) from shared memory — same strategy as CF03 CLLM
2. Adds the bias vector in the same pass
3. Applies ReLU in-place before writing the output tile back to global memory

Fusing bias + ReLU into the GEMM kernel eliminates a second global-memory round-trip per layer, saving 3 × BATCH × H bytes of DRAM traffic.  
A lightweight softmax kernel (one thread block per row, 10 elements) finishes the pass.

## Arithmetic Intensity

| Layer    | FLOPs (batch=256)     | Bytes (W + X_in + X_out) |
|----------|-----------------------|--------------------------|
| FC1      | 2 × 256 × 256 × 40  = 5,242,880  | (256×40 + 256×40 + 256×256) × 4 = 0.82 MB |
| FC2      | 2 × 256 × 128 × 256 = 16,777,216 | (128×256 + 256×256 + 256×128) × 4 = 0.79 MB |
| FC3      | 2 × 256 × 10  × 128 = 655,360    | (10×128 + 256×128 + 256×10) × 4 = 0.14 MB |
| **Total**| **~22.7 MFLOP**       | **~1.75 MB**             |

**AI ≈ 22.7M / 1.75M ≈ 13 FLOP/byte** — below the ridge point of 27, so this network is **memory-bound** on the T4 at batch=256.

This makes intuitive sense: the weight matrices are small (FC1 is only 40×256 = 40 KB, fits in L2 cache), so DRAM bandwidth rather than compute throughput is the bottleneck. Increasing batch size shifts the balance — at batch=4096 the AI rises above the ridge and the kernel becomes compute-bound.

## Measured Results (T4, Colab)

<!-- Fill in after running ./nn_forward on Colab -->

| Metric | Value |
|--------|-------|
| Time per forward pass | ___ ms |
| GPU throughput | ___ samples/s |
| Performance | ___ GFLOP/s |
| Bandwidth | ___ GB/s |
| CPU baseline (single-thread) | ___ ms |
| GPU speedup | ___× |

## Comparison to CLLM GEMM Kernels

The standalone GEMM benchmarks in CLLM used a single 1024×1024 matrix (AI = 170.67 FLOP/byte, well above ridge). The NN forward pass operates on much smaller matrices, driving the AI below the ridge. This is a key practical lesson: a real inference workload is often *more* memory-bound than isolated GEMM benchmarks suggest, because weight matrices fit in cache and activation tensors are small.

Increasing tile size (T=16 here vs T=8 in CLLM) helps recycle shared-memory data more aggressively, but the dominant bottleneck at small batch sizes remains DRAM bandwidth for loading weights on the first pass.

## How to Run

```bash
# On Colab T4 (after uploading nn_forward.cu):
nvcc -O2 -o nn_forward nn_forward.cu -lm
./nn_forward
```

Then fill in `gflops_gpu` and `ai_gpu` in `colab_nn.ipynb` Cell 4 and run to get `nn_roofline.png`.
