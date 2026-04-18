# Interface Selection
**Project:** Ternary KWS Inference Accelerator  
**ECE 410/510 Spring 2026**

---

## Chosen Interface: SPI (Serial Peripheral Interface)

Selected from the project document table: SPI, I²C, AXI4-Lite, AXI4 Stream, PCIe, UCIe.

**Choice: SPI**

---

## Host Platform

**Assumed host:** ARM Cortex-M class microcontroller (e.g., STM32L4, nRF5340, or equivalent)  
Category: **MCU** (not FPGA SoC, not data-center host)

This matches the edge deployment context: an always-on battery-powered device where the chiplet acts as a keyword-spotting co-processor connected to a low-power MCU.

---

## Bandwidth Requirement Calculation

The kernel transfers MFCC feature vectors (input) and logit scores (output) over the interface at each inference.

**Operating point:** 100 inferences per second (one inference per 10 ms audio frame stride — real-time keyword spotting with a sliding 1-second window).

```
Input  per inference: 1960 MFCC coefficients × 2 bytes (INT16) = 3,920 bytes
Output per inference:   10 class logits       × 2 bytes (INT16) =    20 bytes
───────────────────────────────────────────────────────────────────────────────
Total  per inference:                                              3,940 bytes

Required bandwidth = 3,940 bytes/inference × 100 inferences/sec
                   = 394,000 bytes/sec
                   = 0.394 MB/s
                   = 3.15 Mb/s
```

---

## Interface Bandwidth vs. Required Bandwidth

| Interface | Clock | Rated bandwidth | Required | Headroom | Interface-bound? |
|---|---|---|---|---|---|
| **SPI** | **10 MHz** | **10 Mb/s (1.25 MB/s)** | **3.15 Mb/s** | **3.2×** | **No** |
| SPI | 50 MHz | 50 Mb/s (6.25 MB/s) | 3.15 Mb/s | 15.9× | No |
| I²C (Fast+) | 1 MHz | 1 Mb/s (0.125 MB/s) | 3.15 Mb/s | 0.32× | **Yes** (bottleneck) |

**Conclusion:** SPI at 10 MHz provides 3.2× headroom over the required bandwidth. The design is **not interface-bound** on the roofline. I²C at 1 MHz would be interface-bound and is therefore rejected.

---

## Roofline Position — Interface Check

From `codefest/cf02/analysis/ai_calculation.md`, the hardware accelerator operates at AI = 406 FLOP/byte (on-chip weights). At this operating point:

- Compute ceiling: 10 GOPS (target chiplet throughput)
- SPI bandwidth ceiling at 10 MHz: 1.25 MB/s → at AI = 406 FLOP/byte, the SPI-limited throughput would be: 1.25 × 10⁶ B/s × 406 FLOP/B = 508 GFLOP/s (far above the compute ceiling of 10 GOPS)

The SPI bandwidth is not the limiting factor. The compute ceiling (10 GOPS) is the binding constraint, as intended for a compute-bound design.

---

## Rationale for SPI over Other Interfaces

| Candidate | Rated BW | Verdict | Reason |
|---|---|---|---|
| **SPI** | 10–50 Mb/s | ✅ Chosen | Universal MCU peripheral; sufficient BW; simple protocol; < 4 pins |
| I²C | ≤ 1 Mb/s | ✗ Rejected | Insufficient bandwidth at standard speeds |
| AXI4-Lite | > 1 Gb/s | ✗ Overkill | Requires FPGA SoC or AMBA interconnect; not available on MCU class |
| AXI4 Stream | > 1 Gb/s | ✗ Overkill | Same as above |
| PCIe | >> 1 Gb/s | ✗ Overkill | Data-center interface; incompatible with edge MCU |
| UCIe | >> 1 Gb/s | ✗ Overkill | Die-to-die; incompatible with MCU deployment target |

SPI is the only option that satisfies the bandwidth requirement, is universally available on MCU platforms, and is synthesizable as a simple digital peripheral at negligible area overhead.
