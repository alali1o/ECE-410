# Heilmeier Catechism — Project Draft
**ECE 410/510 Spring 2026**
**Project: Binary/Ternary Neural Network Inference Accelerator for Keyword Spotting**

---

## Question 1: What are you trying to do?

I am building a custom hardware chip (chiplet) that can recognize spoken keywords (such as "yes", "no", or "stop") directly on a small, battery-powered device — without sending audio data to the cloud. The chip accelerates a highly compressed neural network that uses only -1, 0, and +1 as weight values (binary/ternary), replacing expensive multiply operations with simple additions. The chiplet connects to a microcontroller host via SPI and must classify a 1-second audio window in real time while consuming minimal power.

---

## Question 2: How is it done today, and what are the limits of current practice?

**How it is done today:**
Keyword spotting on edge devices is typically handled in one of two ways:
1. **On a general-purpose MCU (e.g., ARM Cortex-M4):** Small floating-point or INT8 models (e.g., DS-CNN, MobileNet) run on the MCU's CPU. Libraries like TensorFlow Lite for Microcontrollers (TFLite Micro) enable deployment, but the CPU executes all layers sequentially, with no parallelism.
2. **Dedicated DSP/NPU chips (e.g., Syntiant NDP, Arm Ethos-U):** Purpose-built chips with proprietary architectures. These achieve low power but are closed-source, expensive for low-volume use, and inflexible.

**Limits of current practice:**
- MCU-based inference is too slow and energy-hungry for always-on operation; a Cortex-M4 running INT8 inference consumes ~10–50 mW just for the neural network.
- Commercial NPUs are not openly synthesizable or modifiable — students and researchers cannot study or customize them.
- Standard INT8 quantization still requires hardware multipliers, which are area- and power-expensive in silicon.
- Binary/ternary networks (BNNs/TNNs) can replace multiplications with XNOR + popcount operations, dramatically reducing hardware cost, but no open, synthesizable, SPI-connected accelerator targeting OpenLane 2 exists for this use case.

---

## Question 3: What is new in your approach and why do you think it will be successful?

**What is new:**
I will design a fully synthesizable, open-source co-processor chiplet in SystemVerilog that:
1. **Executes ternary {-1, 0, +1} MAC operations** using only adders and subtractors — no multipliers — enabling a compact, low-power compute array.
2. **Targets the dominant kernel** of keyword spotting: the multiply-accumulate layer of a small fully-connected or 1D convolutional network operating on MFCC (Mel-frequency cepstral coefficient) audio features.
3. **Exposes an SPI slave interface** so any MCU-class host can stream feature vectors in and read back classification results — matching the real-world deployment context of always-on edge devices.
4. **Is synthesizable through OpenLane 2**, making the design reproducible and measurable in area, timing, and power.

**Why it will be successful:**
- Ternary arithmetic is well-matched to digital logic: weight storage drops from 8 bits to 2 bits, and MAC units shrink to adder trees, which are compact and fast in standard-cell libraries.
- The target kernel (a ~512–1024 neuron fully-connected layer on a ~40-feature MFCC vector) has a well-defined, fixed-size datapath — ideal for a first synthesizable accelerator.
- SPI at 10–50 Mbit/s is more than sufficient to transfer a 40-coefficient MFCC frame (40 × 16-bit = 640 bytes) in under 1 ms, well within the 10 ms frame stride of typical keyword spotting pipelines.
- The scope is deliberately narrow (one accelerated layer, one interface, one fixed model size), which makes synthesis and verification achievable within the term timeline.
