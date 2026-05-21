# CF08 CMAN — AER Bandwidth Analysis
ECE 410/510 Spring 2026

**Given:** N = 1024 neurons, f = 50 Hz mean firing rate, 20 bits/packet (10-bit address + 6-bit timestamp + 4-bit framing).

---

## Task 1 — Mean Aggregate Spike Rate

R = N × f

R = 1024 × 50 = **51,200 spikes/s**

---

## Task 2 — Mean AER Bandwidth

B = R × 20 bits/packet

B = 51,200 × 20 = 1,024,000 bit/s = **1.024 Mbit/s**

---

## Task 3 — Interface Comparison

| Interface    | Max BW       | Mean B = 1.024 Mbit/s | Sustains? |
|--------------|--------------|------------------------|-----------|
| I²C          | 3.4 Mbit/s   | 1.024 < 3.4            | **Yes**   |
| SPI          | 50 Mbit/s    | 1.024 < 50             | **Yes**   |
| AXI4-Lite    | 100 Mbit/s   | 1.024 < 100            | **Yes**   |

All three can sustain the mean rate.

**Lowest-complexity interface that suffices: I²C**

I²C is the simplest protocol (2-wire, no chip-select, built-in addressing) and its 3.4 Mbit/s ceiling comfortably covers the 1.024 Mbit/s mean. SPI and AXI4-Lite are faster but more complex and unnecessary at this mean rate.

---

## Task 4 — Burst Analysis

**Setup:** 25% of 1024 neurons fire within a 1 ms window.

Number of spikes in burst: 0.25 × 1024 = **256 spikes**

Peak bandwidth = 256 spikes × 20 bits / 1 ms

Peak bandwidth = **5,120,000 bit/s = 5.12 Mbit/s**

**Burst-to-mean ratio = 5.12 / 1.024 = 5**

**Can I²C absorb the burst?**

I²C tops out at 3.4 Mbit/s. During the 1 ms burst window, I²C can transmit:

3,400,000 bit/s × 0.001 s = 3,400 bits → 3,400 / 20 = **170 packets drained**

But 256 packets arrive → **86 packets excess**

**I²C cannot absorb the burst without buffering.** A FIFO of ~128 entries (2,560 bits ≈ 320 bytes) would absorb the worst-case overshoot and drain at the 3.4 Mbit/s rate between bursts.

SPI (50 Mbit/s) can absorb the burst without any buffering and would be the better choice if bursts occur regularly.

---

## Task 5 — Frame-Based Comparison

**Frame-based bandwidth:**

Sample all 1024 neurons every 1 ms, 1 bit per neuron per sample.

B_frame = 1024 bits / 1 ms = 1,024,000 bit/s = **1.024 Mbit/s**

**AER-to-frame ratio at f = 50 Hz:**

AER/frame = 1.024 / 1.024 = **1.0**

At 50 Hz mean firing rate, AER and frame-based use exactly the same bandwidth.

**Crossover firing rate f_crossover:**

Set AER bandwidth equal to frame bandwidth and solve for f:

N × f_c × 20 = N × (1 / 1 ms)

f_c × 20 = 1000

**f_crossover = 50 Hz**

This confirms that 50 Hz is exactly the crossover point, which makes sense: the network is at the crossover by design.

**Implication:** AER saves bandwidth only when neurons fire at rates below 50 Hz (sparse activity); above 50 Hz the 20-bit packet overhead makes AER worse than a simple frame readout, so AER is the right choice for sparse, event-driven SNNs operating well below this threshold.
