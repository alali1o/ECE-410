"""
gen_plots.py — Generate CF09 roofline plots
ECE 410/510 Spring 2026
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Shared values ─────────────────────────────────────────────────────────────
AI_lower = 7.8      # FLOP/byte, no weight reuse
AI_upper = 406.0    # FLOP/byte, full weight reuse

# sky130 ASIC
sky_peak_gflops  = 51.2          # GFLOP/s  (512 MACs/cycle × 50 MHz × 2)
sky_bw_gbs       = 0.00125       # GB/s     (SPI 10 MHz = 1.25 MB/s)
sky_ridge        = sky_peak_gflops / sky_bw_gbs   # 40,960 FLOP/byte

# M4 platform
m4_peak_gflops   = 3600.0        # GFLOP/s FP32 (Apple spec)
m4_bw_gbs        = 120.0         # GB/s    (Apple spec)
m4_ridge         = m4_peak_gflops / m4_bw_gbs     # 30 FLOP/byte

# Measured / projected points
sw_ai       = 0.50     # measured M1 profiling (FLOP/byte, DRAM-bound)
sw_gflops   = 4.91     # measured M4 effective (GFLOP/s)

hw_ai       = AI_upper          # full reuse AI for hw point
hw_gflops   = hw_ai * sky_bw_gbs  # attainable = BW ceiling (memory-bound)
# hw projected compute-only (if we label separately)
hw_compute_gflops = sky_peak_gflops  # 51.2 GFLOP/s


# ── Helper: roofline y = min(peak, ai × bw) ──────────────────────────────────
def roofline(ai_arr, peak, bw):
    return np.minimum(peak, ai_arr * bw)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — CMAN hand-drawn style sketch (cman_roofline_sketch.png)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

ai_range = np.logspace(-1, 6, 1000)   # 0.1 to 1e6 FLOP/byte

# sky130 roofline
rl_sky = roofline(ai_range, sky_peak_gflops, sky_bw_gbs)
ax.loglog(ai_range, rl_sky, 'b-', linewidth=2.5, label='sky130 ASIC (50 MHz, SPI 10 MHz)')

# M4 roofline
rl_m4 = roofline(ai_range, m4_peak_gflops, m4_bw_gbs)
ax.loglog(ai_range, rl_m4, 'k--', linewidth=1.8, alpha=0.6, label='Apple M4 SW baseline')

# Ridge points
ax.axvline(sky_ridge, color='b', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(m4_ridge,  color='k', linestyle=':', linewidth=1, alpha=0.4)

ax.plot(sky_ridge, sky_peak_gflops, 'b^', markersize=9, zorder=5)
ax.plot(m4_ridge,  m4_peak_gflops,  'k^', markersize=9, zorder=5)

ax.annotate(f'sky130 ridge\n{sky_ridge:.0f} FLOP/B', xy=(sky_ridge, sky_peak_gflops),
            xytext=(sky_ridge*3, sky_peak_gflops*0.4),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.2),
            fontsize=8, color='blue')
ax.annotate(f'M4 ridge\n{m4_ridge:.0f} FLOP/B', xy=(m4_ridge, m4_peak_gflops),
            xytext=(m4_ridge*0.2, m4_peak_gflops*0.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
            fontsize=8, color='gray')

# AI bound vertical lines
ax.axvline(AI_lower, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax.axvline(AI_upper, color='green', linestyle='--', linewidth=1.5, alpha=0.8)

# AI bound labels
ymin_lim = 1e-5
ax.text(AI_lower * 1.15, ymin_lim * 3, f'AI_lower\n{AI_lower} FLOP/B\n(no reuse)',
        fontsize=8, color='red', va='bottom')
ax.text(AI_upper * 1.15, ymin_lim * 3, f'AI_upper\n{AI_upper} FLOP/B\n(full reuse)',
        fontsize=8, color='green', va='bottom')

# Attainable range band on sky130
att_lo = AI_lower * sky_bw_gbs   # 9.75e-3 GFLOP/s
att_hi = AI_upper * sky_bw_gbs   # 508e-3 GFLOP/s
ax.fill_betweenx([att_lo, att_hi],
                 [AI_lower, AI_lower], [AI_upper, AI_upper],
                 alpha=0.12, color='blue', label='Attainable range (sky130, SPI-limited)')

# Kernel performance range (vertical bar between lower and upper bound on sky130 rl)
perf_lo = att_lo   # memory-bound side
perf_hi = att_hi
ax.plot([AI_lower, AI_upper], [att_lo, att_hi], 'b-', linewidth=3, alpha=0.4)

# SW baseline measured point
ax.plot(sw_ai, sw_gflops, 'ks', markersize=10, zorder=6, label=f'SW baseline (M4 measured)\nAI={sw_ai}, {sw_gflops} GFLOP/s')
ax.annotate(f'SW baseline\n{sw_gflops} GFLOP/s', xy=(sw_ai, sw_gflops),
            xytext=(sw_ai*0.3, sw_gflops*5),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            fontsize=8)

# Labels, ceiling annotations
ax.axhline(sky_peak_gflops, color='b', linestyle=':', linewidth=1, alpha=0.4)
ax.axhline(m4_peak_gflops,  color='k', linestyle=':', linewidth=1, alpha=0.3)
ax.text(1e5, sky_peak_gflops * 1.2, f'sky130 compute ceiling\n{sky_peak_gflops} GFLOP/s',
        fontsize=8, color='blue', ha='right')
ax.text(1e5, m4_peak_gflops * 1.2, f'M4 compute ceiling\n{m4_peak_gflops} GFLOP/s',
        fontsize=8, color='gray', ha='right')

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=11)
ax.set_ylabel('Attainable Performance (GFLOP/s)', fontsize=11)
ax.set_title('CF09 CMAN Roofline Sketch\n'
             'Ternary KWS Accelerator (sky130 target) vs M4 SW Baseline\n'
             'FC1: 1960×512 ternary GEMV, N_IN=1960, N_OUT=512',
             fontsize=10)
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0.1, 1e6)
ax.set_ylim(1e-5, 1e5)
ax.grid(True, which='both', alpha=0.3, linestyle=':')

# Region labels
ax.text(1.5, 3e-4, 'Memory-bound\n(SPI-limited)', fontsize=9, color='blue', alpha=0.7)
ax.text(1e5, 30, 'Compute-bound\n(sky130)', fontsize=9, color='blue', alpha=0.7)

plt.tight_layout()
plt.savefig('cman_roofline_sketch.png', dpi=150, bbox_inches='tight')
print("Saved cman_roofline_sketch.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — CLLM benchmark roofline (benchmarks/roofline_plot.png)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

# sky130 ASIC roofline
ax.loglog(ai_range, rl_sky, 'b-', linewidth=2.5, label=f'sky130 ASIC roofline (50 MHz, SPI 10 MHz)')

# M4 roofline (reference)
ax.loglog(ai_range, rl_m4, 'k--', linewidth=1.5, alpha=0.5, label='Apple M4 roofline (reference)')

# Ridge lines
ax.axvline(sky_ridge, color='b', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(m4_ridge,  color='k', linestyle=':', linewidth=1, alpha=0.4)
ax.axhline(sky_peak_gflops, color='b', linestyle=':', linewidth=1, alpha=0.3)
ax.axhline(m4_peak_gflops,  color='k', linestyle=':', linewidth=1, alpha=0.2)

# AI bound vertical lines
ax.axvline(AI_lower, color='red',   linestyle='--', linewidth=1.2, alpha=0.7)
ax.axvline(AI_upper, color='green', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(AI_lower * 1.15, 2e-5, f'AI_low\n{AI_lower}', fontsize=7.5, color='red')
ax.text(AI_upper * 1.15, 2e-5, f'AI_high\n{AI_upper}', fontsize=7.5, color='green')

# SW baseline point
ax.plot(sw_ai, sw_gflops, 'ks', markersize=11, zorder=7,
        label=f'SW baseline — M4 measured\nAI={sw_ai} F/B, {sw_gflops} GFLOP/s, 2293 inf/s')

# HW projected point (SPI-limited, full reuse)
hw_spi_gflops = AI_upper * sky_bw_gbs   # ≈ 5.08e-4 GFLOP/s
ax.plot(AI_upper, hw_spi_gflops, 'b^', markersize=12, zorder=7,
        label=f'HW accelerator — PROJECTED (SPI-limited)\nAI={AI_upper} F/B, {hw_spi_gflops*1e6:.0f} KFLOP/s, ~252 inf/s')

# HW projected compute-only point (if SPI removed)
ax.plot(AI_upper, sky_peak_gflops, 'b*', markersize=14, zorder=7,
        label=f'HW accelerator — PROJECTED (compute ceiling)\n{sky_peak_gflops} GFLOP/s, 25,510 inf/s (no SPI overhead)')

# Annotations
ax.annotate('SW baseline\n(M4 measured)', xy=(sw_ai, sw_gflops),
            xytext=(0.2, 30), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='k', lw=1))
ax.annotate('HW projected\n(SPI-limited)', xy=(AI_upper, hw_spi_gflops),
            xytext=(500, hw_spi_gflops * 100), fontsize=8, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1))
ax.annotate('HW projected\n(compute ceiling)', xy=(AI_upper, sky_peak_gflops),
            xytext=(500, sky_peak_gflops * 0.15), fontsize=8, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1))

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=11)
ax.set_ylabel('Attainable Performance (GFLOP/s)', fontsize=11)
ax.set_title('CF09 CLLM Roofline — SW Baseline vs HW Accelerator (Projected)\n'
             'Ternary KWS Accelerator — FC1: 1960×512 GEMV on sky130',
             fontsize=10)
ax.legend(fontsize=7.5, loc='upper left')
ax.set_xlim(0.1, 1e6)
ax.set_ylim(1e-5, 1e5)
ax.grid(True, which='both', alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('benchmarks/roofline_plot.png', dpi=150, bbox_inches='tight')
print("Saved benchmarks/roofline_plot.png")
plt.close()
