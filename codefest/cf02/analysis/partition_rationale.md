# HW/SW Partition Proposal
**ECE 410/510 Spring 2026**
**Project: Ternary NN Inference Accelerator for Keyword Spotting**

---

## (a) Which kernel to accelerate in hardware — and why the roofline supports it

The kernel selected for hardware acceleration is the **ternary fully-connected (FC) forward pass**
— the three matrix-vector multiplies that form the backbone of the keyword spotting inference
pipeline (layers: 40→256→128→10). This kernel accounts for over 90% of all arithmetic operations
in the forward pass, as confirmed by the cProfile output where `ternary_linear` (calling
`numpy.dot`) is the only function performing substantial arithmetic.

The roofline analysis strongly supports this choice. On the Apple M4 CPU (FP32, DRAM), the
kernel sits at AI = 0.495 FLOP/byte — well into the memory-bound region below the ridge point
of 1.92 FLOP/byte. This means the CPU's 230 GFLOPS of compute is severely underutilized; the
hardware is stalled waiting for data. A custom accelerator that stores weights in INT8 (1 byte per
ternary value instead of 4 bytes FP32) immediately raises the effective AI to 1.92 FLOP/byte,
pushing the kernel across the ridge point of the accelerator roofline (0.50 FLOP/byte) into the
compute-bound region — achieving 100 GFLOPS vs. 59.4 GFLOPS on the CPU, a **×1.7 speedup**
on this kernel alone.

## (b) What the software baseline will continue to handle

The host MCU (ARM Cortex-M class) will continue to handle all pre-processing steps that are
not arithmetic-intensive: audio sampling, framing, windowing, and MFCC feature extraction
(FFT + mel filterbank + log + DCT). These steps are sequential, control-flow heavy, and
produce only 40 output values — too little data to amortize accelerator invocation overhead.
The MCU also handles the SPI transaction protocol, result interpretation (argmax over 10 classes),
and any application-level logic (debouncing, wake word confirmation).

## (c) Interface bandwidth required to avoid becoming interface-bound

At inference time, the accelerator receives **40 FP32 MFCC coefficients** (input) and returns
**10 FP32 class scores** (output) per inference. Total data per invocation:

```
(40 + 10) × 4 bytes = 200 bytes per inference
```

Keyword spotting runs at ~10 ms per frame (100 inferences/sec), so the required interface
throughput is:

```
200 bytes × 100 inferences/sec = 20,000 bytes/sec = 0.16 Mbit/s
```

SPI at even 1 Mbit/s is more than 6× the required bandwidth — the interface is nowhere near
the bottleneck. This confirms that **SPI is the correct interface choice** for this application.

## (d) Bound classification and expected change with the accelerator

On the Apple M4 CPU with FP32 weights loaded from DRAM, the kernel is firmly
**memory-bound** (AI = 0.495 F/B vs. ridge = 1.92 F/B). The hardware accelerator changes
this in two ways: (1) INT8 ternary weight storage reduces memory traffic by 4×, raising AI to
1.92 F/B, and (2) weights are stored in on-chip SRAM (200 GB/s) rather than DRAM (120 GB/s),
further improving effective throughput. The result is that the kernel crosses the accelerator's
ridge point (0.50 F/B) and becomes **compute-bound** on the custom chip — which is the ideal
operating regime for a dedicated accelerator, as it means arithmetic units are fully utilized.
