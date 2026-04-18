import numpy as np
import matplotlib.pyplot as plt

peak_compute = 8100   # GFLOPS FP32 (T4)
peak_bw      = 300    # GB/s
ridge        = peak_compute / peak_bw   # 27 FLOP/byte

# CLLM GEMM results
ai_gemm      = 170.67
gflops_naive = 316.09
gflops_tiled = 392.46

# COPT NN forward pass results (measured on T4)
ai_nn     = 22.345
gflops_nn = 190.32

ai_x     = np.logspace(-1, 4, 2000)
roofline = np.minimum(ai_x * peak_bw, peak_compute)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(ai_x, roofline, 'k-', linewidth=2.2, label='T4 Roofline')
ax.axvline(ridge, color='gray', linestyle='--', linewidth=1.2)
ax.text(ridge * 1.08, 12, f'Ridge\n{ridge:.0f} F/B', fontsize=8.5, color='gray')

# CLLM kernels (reference)
ax.plot(ai_gemm, gflops_naive, 'o', color='lightcoral', markersize=9, zorder=4,
        label=f'GEMM Naive — {gflops_naive:.0f} GFLOP/s')
ax.plot(ai_gemm, gflops_tiled, 's', color='steelblue', markersize=9, zorder=4,
        label=f'GEMM Tiled — {gflops_tiled:.0f} GFLOP/s')

# COPT: NN forward pass
ax.plot(ai_nn, gflops_nn, 'D', color='darkorange', markersize=12, zorder=5,
        label=f'NN Fwd Pass — {gflops_nn:.0f} GFLOP/s')
ax.annotate(f'NN Fwd Pass\n{gflops_nn:.0f} GFLOP/s\nAI={ai_nn:.1f} F/B\n111x CPU speedup',
            xy=(ai_nn, gflops_nn),
            xytext=(ai_nn * 0.13, gflops_nn * 5),
            fontsize=8.5, color='darkorange',
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.3))

ax.text(0.15, 40,  'Memory-Bound',  fontsize=9, color='gray', alpha=0.7)
ax.text(300,  40,  'Compute-Bound', fontsize=9, color='gray', alpha=0.7)

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=11)
ax.set_ylabel('Attainable Performance (GFLOPS)',   fontsize=11)
ax.set_title('CUDA NN Forward Pass Roofline — NVIDIA T4 GPU\n'
             'FC(40->256->128->10), Batch=256, FP32 | Measured on Google Colab',
             fontsize=11, fontweight='bold', pad=10)
ax.set_xlim(1e-1, 1e4)
ax.set_ylim(1e1,  1e5)
ax.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.5)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig('nn_roofline.png', dpi=150, bbox_inches='tight')
print("Saved nn_roofline.png")
print(f"\nNN Fwd Pass: {gflops_nn} GFLOP/s  (AI={ai_nn:.2f}, {100*gflops_nn/peak_compute:.1f}% of peak)")
print(f"Memory-bound: AI={ai_nn:.2f} < ridge={ridge:.0f} FLOP/byte")
print(f"GPU speedup: 111.2x over single-threaded CPU")
