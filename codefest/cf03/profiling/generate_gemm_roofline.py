import numpy as np
import matplotlib.pyplot as plt

# ── T4 GPU spec ────────────────────────────────────────────────────────────────
peak_compute = 8100   # GFLOPS FP32
peak_bw      = 300    # GB/s
ridge        = peak_compute / peak_bw   # 27 FLOP/byte

# ── Kernel stats (measured on Colab T4) ───────────────────────────────────────
N        = 1024
flops    = 2.0 * N**3          # 2,147,483,648
bytes_rw = 3.0 * N**2 * 4     # 12,582,912  (load A, B; store C)
ai       = flops / bytes_rw    # 170.67 FLOP/byte

gflops_naive = 316.09   # measured
gflops_tiled = 392.46   # measured

# ── Roofline curve ─────────────────────────────────────────────────────────────
ai_x     = np.logspace(-1, 4, 2000)
roofline = np.minimum(ai_x * peak_bw, peak_compute)

fig, ax = plt.subplots(figsize=(10, 6))

# Roofline
ax.loglog(ai_x, roofline, 'k-', linewidth=2.2, label='T4 Roofline')

# Ridge point
ax.axvline(ridge, color='gray', linestyle='--', linewidth=1.2)
ax.text(ridge * 1.08, 12, f'Ridge\n{ridge:.0f} F/B', fontsize=8.5, color='gray')

# Naive kernel
ax.plot(ai, gflops_naive, 'o', color='tomato', markersize=11, zorder=5,
        label=f'Naive GEMM — {gflops_naive:.0f} GFLOP/s')
ax.annotate(f'Naive\n{gflops_naive:.0f} GFLOP/s\n1.85 GB/s BW',
            xy=(ai, gflops_naive),
            xytext=(ai * 0.07, gflops_naive * 3.5),
            fontsize=8.5, color='tomato',
            arrowprops=dict(arrowstyle='->', color='tomato', lw=1.3))

# Tiled kernel
ax.plot(ai, gflops_tiled, 's', color='steelblue', markersize=11, zorder=5,
        label=f'Tiled GEMM (T=8) — {gflops_tiled:.0f} GFLOP/s')
ax.annotate(f'Tiled (T=8)\n{gflops_tiled:.0f} GFLOP/s\n2.30 GB/s BW',
            xy=(ai, gflops_tiled),
            xytext=(ai * 0.07, gflops_tiled * 6),
            fontsize=8.5, color='steelblue',
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.3))

# Speedup annotation
ax.annotate('', xy=(ai * 1.05, gflops_tiled),
            xytext=(ai * 1.05, gflops_naive),
            arrowprops=dict(arrowstyle='<->', color='green', lw=1.8))
ax.text(ai * 1.15, (gflops_naive + gflops_tiled) / 2,
        f'×{gflops_tiled/gflops_naive:.2f}', fontsize=9, color='green', va='center')

# Region labels
ax.text(0.15, 40,    'Memory-Bound',  fontsize=9, color='gray', alpha=0.7)
ax.text(300,  40,    'Compute-Bound', fontsize=9, color='gray', alpha=0.7)

# Formatting
ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=11)
ax.set_ylabel('Attainable Performance (GFLOPS)',   fontsize=11)
ax.set_title('CUDA GEMM Roofline — NVIDIA T4 GPU\n'
             'Naive vs. Tiled (tile=8), N=1024 FP32 | Measured on Google Colab',
             fontsize=11, fontweight='bold', pad=10)
ax.set_xlim(1e-1, 1e4)
ax.set_ylim(1e1,  1e5)
ax.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.5)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig('gemm_roofline.png', dpi=150, bbox_inches='tight')
print("Saved gemm_roofline.png")
print(f"\nNaive: {gflops_naive} GFLOP/s  (AI={ai:.1f}, {100*gflops_naive/peak_compute:.1f}% of peak)")
print(f"Tiled: {gflops_tiled} GFLOP/s  (AI={ai:.1f}, {100*gflops_tiled/peak_compute:.1f}% of peak)")
print(f"Speedup: {gflops_tiled/gflops_naive:.2f}x")
