"""
plot_waveform.py — Parse VCD and render annotated waveform PNG
M3 Co-Simulation Waveform: ECE 410/510 Spring 2026
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

VCD_FILE = "cosim_waveform.vcd"
OUT_FILE = "cosim_waveform.png"

# ── Minimal VCD parser ────────────────────────────────────────────────────────
def parse_vcd(path):
    signals = {}   # code -> name
    data    = {}   # name -> [(time_ns, value)]
    timescale_mult = 1e-9  # default 1ns

    with open(path) as f:
        lines = f.readlines()

    # Pass 1: find variable declarations and timescale
    scope = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('$timescale'):
            ts = ""
            while '$end' not in ts:
                i += 1
                ts += lines[i]
            if '1ps' in ts:  timescale_mult = 1e-12
            elif '1ns' in ts: timescale_mult = 1e-9
        elif line.startswith('$var'):
            parts = line.split()
            if len(parts) >= 5:
                code = parts[3]
                name = parts[4].rstrip(';')
                signals[code] = name
                data[name] = []
        i += 1

    # Pass 2: extract value changes
    cur_time = 0
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            cur_time = int(line[1:]) * timescale_mult * 1e9  # → ns
        elif line and line[0] in '01xz' and len(line) > 1:
            val  = line[0]
            code = line[1:]
            if code in signals:
                nm = signals[code]
                data[nm].append((cur_time, int(val) if val in '01' else 0))

    return data


def make_steps(pts, t_end):
    """Convert (t, v) list to step arrays for plotting."""
    if not pts:
        return np.array([0, t_end]), np.array([0, 0])
    ts = [p[0] for p in pts] + [t_end]
    vs = [p[1] for p in pts] + [pts[-1][1]]
    t_out, v_out = [], []
    for k in range(len(pts)):
        t_out += [ts[k], ts[k+1]]
        v_out += [vs[k], vs[k]]
    return np.array(t_out), np.array(v_out)


# ── Load data ─────────────────────────────────────────────────────────────────
data = parse_vcd(VCD_FILE)

# Signals to plot (adjust names to match VCD)
want = ['clk', 'rst_n', 'cs_n', 'sclk', 'mosi', 'miso',
        'dut.weight_init_done', 'dut.start_pulse',
        'dut.u_core.valid', 'dut.miso']

# Fall back: use whatever is available
available = list(data.keys())
plot_sigs = []
for w in ['clk', 'rst_n', 'dut.weight_init_done',
          'cs_n', 'sclk', 'mosi', 'miso',
          'dut.start_pulse', 'dut.u_core.valid']:
    # try exact then suffix match
    if w in data:
        plot_sigs.append(w)
    else:
        matches = [k for k in data if k.endswith(w.split('.')[-1])]
        if matches:
            plot_sigs.append(matches[0])

# Deduplicate preserving order
seen = set()
plot_sigs = [s for s in plot_sigs if not (s in seen or seen.add(s))]

t_end = max((pts[-1][0] if pts else 0) for pts in data.values()) if data else 1000

# ── Plot ──────────────────────────────────────────────────────────────────────
n = len(plot_sigs)
fig_h = max(5, n * 0.75 + 2)
fig, axes = plt.subplots(n, 1, figsize=(16, fig_h), sharex=True)
if n == 1:
    axes = [axes]

colors = {'clk': '#555', 'rst_n': '#d44', 'cs_n': '#d44',
          'sclk': '#55a', 'mosi': '#5a5', 'miso': '#a55',
          'default': '#3388cc'}

for ax, sig in zip(axes, plot_sigs):
    pts = data.get(sig, [])
    ts, vs = make_steps(pts, t_end)
    col = colors.get(sig.split('.')[-1], colors['default'])
    ax.fill_between(ts / 1000, vs, alpha=0.15, color=col, step='post')
    ax.step(ts / 1000, vs, where='post', color=col, linewidth=1.2)
    ax.set_ylim(-0.2, 1.4)
    ax.set_yticks([0, 1])
    label = sig.split('.')[-1]
    ax.set_ylabel(label, fontsize=7, rotation=0, ha='right', va='center')
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.tick_params(axis='y', labelsize=6)
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.5)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

axes[-1].set_xlabel('Time (µs)', fontsize=9)

# Annotation regions (approximate µs)
t_us_end = t_end / 1000

# Region 1: weight init (0 → ~1 µs)
axes[0].axvspan(0, 0.9, alpha=0.07, color='gray')
axes[0].text(0.45, 1.1, '① Weight\nInit', ha='center', fontsize=7, color='gray')

# Region 2: SPI feature writes (~1 → ~29 µs)
axes[0].axvspan(0.9, 28, alpha=0.07, color='blue')
axes[0].text(14, 1.1, '② SPI Feature Writes (16 bytes → reg[0])',
             ha='center', fontsize=7, color='navy')

# Region 3: start + compute + read
axes[0].axvspan(28, t_us_end, alpha=0.07, color='green')
axes[0].text(28 + (t_us_end - 28)/2, 1.1,
             '③ Start / Compute / Poll / Read Result',
             ha='center', fontsize=7, color='darkgreen')

fig.suptitle('M3 End-to-End Co-Simulation Waveform\n'
             'Ternary KWS Inference Accelerator — ECE 410/510 Spring 2026\n'
             'Input x=[10,−5,3,7,−2,8,0,4]  →  y=[12,12,6,−8]  →  argmax=0  →  PASS',
             fontsize=9, y=1.0)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
print(f"Saved {OUT_FILE}")
