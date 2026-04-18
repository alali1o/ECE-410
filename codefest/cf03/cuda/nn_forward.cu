// nn_forward.cu — GPU forward pass: 3-layer FC network (40→256→128→10)
// ECE 410/510 Spring 2026 — Codefest 3 COPT
// Network matches the project's keyword-spotting ternary NN architecture.
//
// Compile: nvcc -O2 -o nn_forward nn_forward.cu
// Run:     ./nn_forward

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ── Layer dimensions ────────────────────────────────────────────────────────
#define IN_DIM   40      // MFCC input features
#define H1_DIM   256     // hidden layer 1
#define H2_DIM   128     // hidden layer 2
#define OUT_DIM  10      // output classes (keywords)
#define BATCH    256     // inference batch size
#define TILE     16      // shared-memory tile size

// ── Error checking ───────────────────────────────────────────────────────────
#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ── Tiled GEMM + fused bias + optional ReLU ──────────────────────────────────
// C = A × W^T + b,  then optionally ReLU(C)
// A : [M × K]   (batch × in_features)
// W : [N × K]   (out_features × in_features, stored transposed for coalescing)
// b : [N]
// C : [M × N]   (batch × out_features)
__global__ void fc_relu(const float *A, const float *W, const float *b,
                        float *C, int M, int N, int K, bool relu)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // batch index
    int col = blockIdx.x * TILE + threadIdx.x;   // output neuron index

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int w_col = t * TILE + threadIdx.y;   // W is [N×K], access row=col, col=w_col

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Ws[threadIdx.y][threadIdx.x] = (col < N && w_col < K) ? W[col * K + w_col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Ws[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = sum + b[col];
        C[row * N + col] = (relu && val < 0.0f) ? 0.0f : val;
    }
}

// ── Softmax (one warp per row) ────────────────────────────────────────────────
__global__ void softmax(float *X, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    float *x = X + row * N;

    // max for numerical stability
    float maxv = x[0];
    for (int i = 1; i < N; i++) maxv = fmaxf(maxv, x[i]);

    float sumv = 0.0f;
    for (int i = 0; i < N; i++) { x[i] = expf(x[i] - maxv); sumv += x[i]; }
    for (int i = 0; i < N; i++) x[i] /= sumv;
}

// ── Helper: random init ──────────────────────────────────────────────────────
void rand_init(float *buf, int n, float scale) {
    for (int i = 0; i < n; i++)
        buf[i] = scale * ((float)rand() / RAND_MAX - 0.5f);
}

// ── Host ─────────────────────────────────────────────────────────────────────
int main() {
    printf("=== GPU NN Forward Pass (batch=%d, 40→256→128→10) ===\n\n", BATCH);

    // ── Allocate host buffers ─────────────────────────────────────────────────
    float *hX  = (float*)malloc(BATCH * IN_DIM  * sizeof(float));
    float *hW1 = (float*)malloc(H1_DIM * IN_DIM  * sizeof(float));
    float *hb1 = (float*)malloc(H1_DIM * sizeof(float));
    float *hW2 = (float*)malloc(H2_DIM * H1_DIM  * sizeof(float));
    float *hb2 = (float*)malloc(H2_DIM * sizeof(float));
    float *hW3 = (float*)malloc(OUT_DIM * H2_DIM  * sizeof(float));
    float *hb3 = (float*)malloc(OUT_DIM * sizeof(float));
    float *hOut= (float*)malloc(BATCH * OUT_DIM * sizeof(float));

    srand(42);
    rand_init(hX,  BATCH * IN_DIM,  1.0f);
    rand_init(hW1, H1_DIM * IN_DIM,  0.1f);
    rand_init(hb1, H1_DIM,           0.01f);
    rand_init(hW2, H2_DIM * H1_DIM,  0.1f);
    rand_init(hb2, H2_DIM,           0.01f);
    rand_init(hW3, OUT_DIM * H2_DIM, 0.1f);
    rand_init(hb3, OUT_DIM,          0.01f);

    // ── Allocate device buffers ───────────────────────────────────────────────
    float *dX, *dW1, *db1, *dH1;
    float *dW2, *db2, *dH2;
    float *dW3, *db3, *dOut;

    CUDA_CHECK(cudaMalloc(&dX,   BATCH  * IN_DIM  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1,  H1_DIM * IN_DIM  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1,  H1_DIM            * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dH1,  BATCH  * H1_DIM  * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dW2,  H2_DIM * H1_DIM  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2,  H2_DIM            * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dH2,  BATCH  * H2_DIM  * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dW3,  OUT_DIM * H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db3,  OUT_DIM            * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, BATCH  * OUT_DIM  * sizeof(float)));

    // ── Copy inputs and weights to device ────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(dX,  hX,  BATCH  * IN_DIM  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, hW1, H1_DIM * IN_DIM  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db1, hb1, H1_DIM            * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, hW2, H2_DIM * H1_DIM  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db2, hb2, H2_DIM            * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW3, hW3, OUT_DIM * H2_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db3, hb3, OUT_DIM            * sizeof(float), cudaMemcpyHostToDevice));

    // ── Launch config helper ──────────────────────────────────────────────────
    // fc_relu grid: (ceil(N/TILE), ceil(M/TILE)) blocks of TILE×TILE threads
    dim3 block(TILE, TILE);

    // ── Warm-up pass ─────────────────────────────────────────────────────────
    {
        dim3 g1((H1_DIM+TILE-1)/TILE, (BATCH+TILE-1)/TILE);
        dim3 g2((H2_DIM+TILE-1)/TILE, (BATCH+TILE-1)/TILE);
        dim3 g3((OUT_DIM+TILE-1)/TILE, (BATCH+TILE-1)/TILE);

        fc_relu<<<g1, block>>>(dX,  dW1, db1, dH1,  BATCH, H1_DIM,  IN_DIM, true);
        fc_relu<<<g2, block>>>(dH1, dW2, db2, dH2,  BATCH, H2_DIM,  H1_DIM, true);
        fc_relu<<<g3, block>>>(dH2, dW3, db3, dOut, BATCH, OUT_DIM, H2_DIM, false);
        softmax<<<BATCH, 1>>>(dOut, BATCH, OUT_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ── Timed run (100 iterations) ────────────────────────────────────────────
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int REPS = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; r++) {
        dim3 g1((H1_DIM+TILE-1)/TILE, (BATCH+TILE-1)/TILE);
        dim3 g2((H2_DIM+TILE-1)/TILE, (BATCH+TILE-1)/TILE);
        dim3 g3((OUT_DIM+TILE-1)/TILE, (BATCH+TILE-1)/TILE);

        fc_relu<<<g1, block>>>(dX,  dW1, db1, dH1,  BATCH, H1_DIM,  IN_DIM, true);
        fc_relu<<<g2, block>>>(dH1, dW2, db2, dH2,  BATCH, H2_DIM,  H1_DIM, true);
        fc_relu<<<g3, block>>>(dH2, dW3, db3, dOut, BATCH, OUT_DIM, H2_DIM, false);
        softmax<<<BATCH, 1>>>(dOut, BATCH, OUT_DIM);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    float ms = ms_total / REPS;

    // ── FLOPs and bandwidth (all three FC layers) ─────────────────────────────
    // Each FC: 2 × M × N × K multiply-adds
    double flops = 2.0 * BATCH * (
        (double)H1_DIM * IN_DIM +
        (double)H2_DIM * H1_DIM +
        (double)OUT_DIM * H2_DIM
    );
    // Bytes moved (load W, X activations; store Y activations)
    double bytes =
        (double)(H1_DIM * IN_DIM  + BATCH * IN_DIM  + BATCH * H1_DIM)  * sizeof(float) +
        (double)(H2_DIM * H1_DIM  + BATCH * H1_DIM  + BATCH * H2_DIM)  * sizeof(float) +
        (double)(OUT_DIM * H2_DIM + BATCH * H2_DIM  + BATCH * OUT_DIM) * sizeof(float);

    double gflops = flops / (ms * 1e-3) / 1e9;
    double bw_gbs = bytes / (ms * 1e-3) / 1e9;
    double ai     = flops / bytes;

    printf("  Time per forward pass : %.3f ms\n",      ms);
    printf("  Throughput            : %.1f samples/s\n", BATCH / (ms * 1e-3));
    printf("  Performance           : %.2f GFLOP/s\n",  gflops);
    printf("  Bandwidth             : %.2f GB/s\n",     bw_gbs);
    printf("  Arithmetic Intensity  : %.3f FLOP/byte\n", ai);
    printf("\n  T4 peak FP32 = 8100 GFLOP/s  →  %.1f%% utilization\n",
           100.0 * gflops / 8100.0);
    printf("  T4 peak BW   = 300  GB/s     →  %.1f%% BW utilization\n",
           100.0 * bw_gbs / 300.0);
    printf("  Ridge point  = 27 FLOP/byte  →  kernel is %s\n",
           ai > 27.0 ? "compute-bound" : "memory-bound");

    // ── Sanity check: copy output and verify softmax sums to 1 ───────────────
    CUDA_CHECK(cudaMemcpy(hOut, dOut, BATCH * OUT_DIM * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float row_sum = 0.0f;
    for (int j = 0; j < OUT_DIM; j++) row_sum += hOut[j];
    printf("\n  Softmax sanity (row 0 sum = %.6f, expected 1.0)\n", row_sum);

    // ── CPU baseline for speedup ──────────────────────────────────────────────
    // Simple single-threaded GEMM to measure GPU speedup
    printf("\n  Running CPU baseline (single-threaded, batch=%d)...\n", BATCH);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    float *cH1  = (float*)calloc(BATCH * H1_DIM, sizeof(float));
    float *cH2  = (float*)calloc(BATCH * H2_DIM, sizeof(float));
    float *cOut = (float*)calloc(BATCH * OUT_DIM, sizeof(float));

    // FC1 + ReLU
    for (int b = 0; b < BATCH; b++)
        for (int n = 0; n < H1_DIM; n++) {
            float s = hb1[n];
            for (int k = 0; k < IN_DIM; k++) s += hX[b*IN_DIM+k] * hW1[n*IN_DIM+k];
            cH1[b*H1_DIM+n] = s > 0 ? s : 0;
        }
    // FC2 + ReLU
    for (int b = 0; b < BATCH; b++)
        for (int n = 0; n < H2_DIM; n++) {
            float s = hb2[n];
            for (int k = 0; k < H1_DIM; k++) s += cH1[b*H1_DIM+k] * hW2[n*H1_DIM+k];
            cH2[b*H2_DIM+n] = s > 0 ? s : 0;
        }
    // FC3
    for (int b = 0; b < BATCH; b++)
        for (int n = 0; n < OUT_DIM; n++) {
            float s = hb3[n];
            for (int k = 0; k < H2_DIM; k++) s += cH2[b*H2_DIM+k] * hW3[n*H2_DIM+k];
            cOut[b*OUT_DIM+n] = s;
        }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = ((t1.tv_sec - t0.tv_sec) * 1e3) +
                    ((t1.tv_nsec - t0.tv_nsec) / 1e6);

    printf("  CPU time              : %.3f ms\n", cpu_ms);
    printf("  GPU speedup           : %.1f×\n",  cpu_ms / ms);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(dX);  cudaFree(dW1); cudaFree(db1); cudaFree(dH1);
    cudaFree(dW2); cudaFree(db2); cudaFree(dH2);
    cudaFree(dW3); cudaFree(db3); cudaFree(dOut);
    free(hX); free(hW1); free(hb1); free(hW2); free(hb2);
    free(hW3); free(hb3); free(hOut);
    free(cH1); free(cH2); free(cOut);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
