// gemm_tiled.cu — Tiled CUDA GEMM with shared memory (tile size = 8)
// ECE 410/510 Spring 2026 — Codefest 3 CLLM
// Compile: nvcc -O2 -o gemm_tiled gemm_tiled.cu
// Run:     ./gemm_tiled

#include <stdio.h>
#include <cuda_runtime.h>

#define N         1024
#define TILE_SIZE 8

// ---------------------------------------------------------------------------
// Kernel: each block computes one TILE_SIZE×TILE_SIZE output tile.
// Threads cooperatively load A and B tiles into shared memory, then
// compute the partial dot products — no DRAM traffic during the inner loop.
// ---------------------------------------------------------------------------
__global__ void gemm_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Slide the tile window along the k-dimension
    for (int t = 0; t < n / TILE_SIZE; t++) {
        // Each thread loads one element of the A and B tiles
        As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        __syncthreads();    // wait for tile to be fully loaded

        // Compute partial dot product using shared memory — no DRAM
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();    // wait before loading next tile
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
int main() {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) {
        hA[i] = (float)(i % 7) * 0.1f;
        hB[i] = (float)(i % 5) * 0.1f;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    // Each block is TILE_SIZE×TILE_SIZE threads
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(N / TILE_SIZE, N / TILE_SIZE);

    // Warm-up
    gemm_tiled<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    // Timed run (10 iterations)
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int REPS = 10;
    cudaEventRecord(start);
    for (int r = 0; r < REPS; r++)
        gemm_tiled<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float ms_per_run = ms / REPS;

    double flops    = 2.0 * N * N * N;
    double gflops   = flops / (ms_per_run * 1e-3) / 1e9;
    double bytes_rw = 3.0 * N * N * sizeof(float);
    double bw_gbs   = bytes_rw / (ms_per_run * 1e-3) / 1e9;

    printf("=== Tiled GEMM (N=%d, tile=%d) ===\n", N, TILE_SIZE);
    printf("  Time per run : %.3f ms\n",  ms_per_run);
    printf("  Performance  : %.2f GFLOP/s\n", gflops);
    printf("  Bandwidth    : %.2f GB/s\n",    bw_gbs);
    printf("  AI           : %.2f FLOP/byte\n", flops / bytes_rw);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
