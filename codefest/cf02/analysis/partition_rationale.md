# HW/SW Partition Rationale
**Project:** Ternary KWS Inference Accelerator  
**ECE 410/510 Spring 2026**

---

## Kernel(s) to Accelerate in Hardware

**Accelerated kernel: FC1 matrix-vector multiply (1960 × 512, ternary weights)**

Roofline justification: FC1 accounts for 93.8% of all MACs in the TernaryKWS network (2,007,040 of 2,140,672 FLOPs). In the software baseline on Apple M4 (AI = 0.50 FLOP/byte, float32), FC1 sits deep in the memory-bound region, 67× below the ridge point of 33 FLOP/byte. By moving the ternary weights into on-chip SRAM (250,880 bytes — feasible in ASIC) and accepting only input/output vectors over SPI, the effective arithmetic intensity rises to 406 FLOP/byte, crossing into the compute-bound region. This is the fundamental motivation for a hardware accelerator: on-chip weight storage is only practical when the model is small enough to fit (it is — 245 KB), and ternary arithmetic eliminates multipliers, making the compute array cheap enough to implement at the required scale.

**FC2 and FC3** (131,072 and 2,560 FLOPs respectively) will also be executed on the chiplet since their weight matrices (32 KB and 2.5 KB ternary) fit easily alongside FC1 in on-chip SRAM. They do not dominate performance but are cheaper to accelerate than to manage separately.

---

## What Remains in Software

The following stages remain on the host MCU in software:

1. **Audio capture and pre-emphasis** — PDM/PCM readout from an off-chip microphone, DC removal, pre-emphasis filter. These are simple DSP operations with no MAC intensity; they run comfortably on a Cortex-M0+ at < 1 mW.

2. **MFCC feature extraction** — FFT → Mel filterbank → log → DCT, producing 40 coefficients per 10 ms frame, 49 frames per 1-second window. This pipeline has higher arithmetic intensity than the NN layers but is well-served by fixed-point DSP units already present on MCU platforms (e.g., STM32 DSP extension). Adding it to the chiplet would require a general-purpose DSP block, significantly increasing area for marginal benefit.

3. **Post-processing and host logic** — Debouncing (e.g., requiring 3 consecutive confident detections before triggering), command routing, and SPI master management. These are control-flow dominated and run efficiently in software.

4. **Model updates / retraining** — Done offline on a workstation. The chiplet does not need to support training.

---

## Compute-Bound vs. Memory-Bound Analysis

| Scenario | AI (FLOP/byte) | M4 ridge point | Bound | Notes |
|---|---|---|---|---|
| SW baseline (FP32, DRAM) | 0.50 | 33 | **Memory-bound** | 67× below ridge |
| SW baseline (ternary 2-bit, DRAM) | 7.70 | 33 | **Memory-bound** | 4.3× below ridge |
| HW accelerator (ternary, on-chip W) | 406 | ~0.02* | **Compute-bound** | 20,000× above accelerator ridge |

*Accelerator ridge = peak ternary ops / on-chip SRAM bandwidth = 10 GOPS / 500 GB/s = 0.02 FLOP/byte.

The hardware design changes the kernel from memory-bound to compute-bound by eliminating off-chip weight traffic. This is only possible because:
- The ternary model is small (245 KB for FC1), fitting in on-chip SRAM.
- Ternary arithmetic replaces multipliers with adders, making the compute array area-efficient.

---

## Interface Bandwidth Requirement

The accelerator must receive MFCC input vectors and return output logits over SPI.

**Required bandwidth calculation:**

Target operating point: 100 inferences per second (10 ms frame stride, real-time keyword spotting).

```
Input  per inference: 1960 coefficients × 2 bytes/coeff (INT16) = 3,920 bytes
Output per inference:   10 class scores  × 2 bytes/score (INT16) =    20 bytes
Total  per inference:                                               3,940 bytes

Required bandwidth = 3,940 bytes × 100 inferences/sec = 394,000 bytes/sec
                   = 0.394 MB/s = 3.15 Mb/s
```

**Comparison to chosen interface:**

| Interface | Rated bandwidth | Required | Margin | Interface-bound? |
|---|---|---|---|---|
| SPI @ 10 MHz | 10 Mb/s = 1.25 MB/s | 3.15 Mb/s | 3.2× | No |
| SPI @ 50 MHz | 50 Mb/s = 6.25 MB/s | 3.15 Mb/s | 15.9× | No |

The design is **not interface-bound** at SPI 10 MHz or higher. The roofline operating point is determined by the on-chip ternary compute throughput, not by the SPI link. Weight traffic is zero at the interface (weights are resident in on-chip SRAM after one-time loading at boot).

---

## Summary

The single kernel to accelerate is the ternary FC layer group (FC1 + FC2 + FC3), dominated by FC1. Moving weights on-chip transforms the memory-bound software kernel (AI = 0.50) into a compute-bound hardware kernel (AI = 406). Ternary arithmetic makes the compute array small enough to fit the design in a sub-mm² standard-cell area. The SPI interface provides sufficient bandwidth (3.2× margin at 10 MHz) and is not the performance bottleneck.
