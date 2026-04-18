// gemm_naive.cu — Naive CUDA GEMM: one thread per output element
// ECE 410/510 Spring 2026 — Codefest 3 CLLM
// Compile: nvcc -O2 -o gemm_naive gemm_naive.cu
// Run:     ./gemm_naive

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// ---------------------------------------------------------------------------
// Kernel: each thread computes one element of C = A × B
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
int main() {
    size_t bytes = (size_t)N * N * sizeof(float);

    // Allocate and initialise host matrices
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) {
        hA[i] = (float)(i % 7) * 0.1f;
        hB[i] = (float)(i % 5) * 0.1f;
    }

    // Allocate device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    // Launch config: 16×16 thread blocks
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // Warm-up
    gemm_naive<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    // Timed run (10 iterations)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int REPS = 10;
    cudaEventRecord(start);
    for (int r = 0; r < REPS; r++)
        gemm_naive<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float ms_per_run = ms / REPS;

    double flops    = 2.0 * N * N * N;
    double gflops   = flops / (ms_per_run * 1e-3) / 1e9;
    double bytes_rw = 3.0 * N * N * sizeof(float);        // load A, B; store C
    double bw_gbs   = bytes_rw / (ms_per_run * 1e-3) / 1e9;

    printf("=== Naive GEMM (N=%d) ===\n", N);
    printf("  Time per run : %.3f ms\n",  ms_per_run);
    printf("  Performance  : %.2f GFLOP/s\n", gflops);
    printf("  Bandwidth    : %.2f GB/s\n",    bw_gbs);
    printf("  AI           : %.2f FLOP/byte\n", flops / bytes_rw);

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
