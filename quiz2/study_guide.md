# ECE 410/510 — Quiz 2 Study Guide
*Covers Week 5+ material: TPUs, Transformers, In-Memory Computing, Neuromorphic Chips, CUDA MLP*

---

## How to Use This Guide
Work each unit in order. For every key concept: read the explanation, close the guide, reproduce the formula or definition from memory, then check. For CMAN-style problems (systolic trace, sneak path KCL), always write the formulas before plugging in numbers — partial credit depends on it.

**Companion files** (read in this order):
1. `quiz_marked_slides.md` — the slides explicitly stamped with QUIZ. **Read this first.**
2. `cheatsheet.md` — one-page distilled facts
3. `study_guide.md` (this file) — full conceptual depth
4. `practice_questions.md` — test yourself

---

## Unit 1: TPU Architecture & Systolic Arrays

### Why TPUs Exist
CPUs are flexible but slow at matmul. GPUs are great but pay an energy tax on cache hierarchies and OOO logic. TPUs strip everything irrelevant and bet the whole chip on matrix multiply.

### TPU v1 Numbers (memorize)
- **MXU (Matrix Multiply Unit)**: 256 × 256 = **65,536 MACs per cycle**
- 700 MHz → 92 TOPS (INT8)
- Predictions/sec: **225,000** (vs GPU 13,194, CPU 5,482)
- Perf/Watt: **83×** over CPU, 29× over GPU
- INT8 math; FP32 accumulation in some variants
- PCIe Gen3 x16 to host

### TPU Architecture
```
Host ─PCIe─ Unified Buffer ──┐
                              ├─→ MXU (256×256 MAC array)
            Weight FIFO ──────┘     │
            (from DDR3)              ↓
                              Accumulators → Activation → Normalize/Pool
```

The **Weight FIFO** prefetches weights so the MXU never stalls. The **Unified Buffer** holds activations on-chip (~24 MB SRAM in v1). This is **weight-stationary**: weights stream into the array continuously, activations sit in the buffer.

### Systolic Array Mechanics
The name comes from the heart — data "pulses" rhythmically through Processing Elements (PEs). Each PE: register X, register Y → multiplier → accumulator → register Z.

**Why it saves energy**: a CPU reads multiple registers per MAC. A systolic array chains ALUs, so **one register read feeds many MACs**. Minimal data movement.

**Timing for N × N matmul**:
- Fill pipeline: N cycles (first result comes out)
- Output phase: N cycles (results stream out)
- Drain: N − 2 cycles
- **Total: 3N − 2 cycles, then 1 result/cycle in steady state**

### Three Dataflow Types
| Dataflow | What stays fixed | Movement | Example |
|----------|------------------|----------|---------|
| **Weight Stationary** | Weights in PEs | Activations stream L→R, partial sums accumulate down | TPU v1–v4 |
| **Output Stationary** | Partial sums in PEs | Both inputs flow through | ShiDianNao |
| **Row Stationary** | Filter row + its reuse | All-data reuse maximized simultaneously | MIT Eyeriss |

**Row-stationary wins on energy**: ~10× fewer DRAM accesses than weight-stationary. But it's harder to implement, so most production hardware uses weight-stationary.

### TPU Roadmap (Spring 2026 context)
| Version | Year | MXU | BF16 TFLOPS | HBM BW |
|---------|------|-----|-------------|--------|
| v4 | 2022 | 128×128 | ~275 | ~1.2 TB/s |
| v7 Ironwood | 2025 | 256×256 | ~2300 | **7400 GB/s** |
| **TPU 8t** (training) | 2026 | — | 12.6 PFLOPs FP4 | 6.5 TB/s |
| **TPU 8i** (inference) | 2026 | — | 10.1 PFLOPs FP4 | **8.6 TB/s, 384 MB SRAM** |

TPU 8t uses 3D torus (9,600 chips/pod); 8i uses Boardfly (max 7 hops, 1,152 chips, lower latency for autoregressive decoding).

---

## Unit 2: BF16 and Precision Formats

### Why BF16
FP16 has only 5 exponent bits → narrow dynamic range → training overflows. FP32 has 8 exponent + 23 mantissa = 32 bits = bandwidth/memory hog.

**BF16 = 1 sign | 8 exponent | 7 mantissa**:
- Same dynamic range as FP32 (8 exp bits)
- Half the memory of FP32 (16 bits)
- Trivial conversion: drop or zero-pad the last 16 mantissa bits
- Developed by **Google Brain** for TPU training

### Precision Hierarchy (memorize)
| Format | Bits | Use case |
|--------|------|----------|
| FP32 | 32 | Training, accumulation |
| FP16 | 16 (5 exp / 10 mant) | Inference, mixed-precision training |
| **BF16** | 16 (8 exp / 7 mant) | Training (safe range) |
| INT8 | 8 | Inference, post-training quantization |
| FP8 | 8 | Hopper+ tensor cores, mixed precision |
| FP4 (E2M1) | 4 (16 distinct values) | Blackwell, B200, NorthPole |
| NVFP4/MXFP4 | 4 + block scale | Micro-block scaled FP4 (16 values share an FP8 scale) |

**Lower precision = lower energy per op** (fewer bits to move and switch). FP4 uses ~8× less energy per multiply than FP32.

### NVIDIA Transformer Engine (H100)
- Dynamically picks **FP16 vs FP8 per layer** every forward pass
- Uses per-layer statistics for range tracking → no accuracy loss
- FP8 = 2× memory savings vs FP16 for inference
- 4th-gen tensor cores on H100

### Blackwell B200
- 208B transistors (dual-die package)
- HBM3e 192 GB, 8 TB/s
- **20 PFLOPS FP4 sparse**
- NVLink 5.0 at 1.8 TB/s
- 1000 W TDP
- New: **MXFP4** (micro-scaled FP4), FP6, 2nd-gen Transformer Engine

---

## Unit 3: Transformers — Non-Recurrent Architecture

### Why NOT RNN/LSTM
RNN/LSTM processes one token at a time. Hidden state passes step t to t+1:
- **Cannot parallelize** across time during training
- **Hidden state bottleneck** — all prior context compressed into one vector
- **Vanishing gradients** on long sequences

### Self-Attention Replaces Recurrence
- All tokens processed **in parallel** during training
- Every token connects **directly** to every other token
- **No hidden state** — order encoded via positional encoding (sin/cos at different frequencies)

### The Attention Formula
```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```
- `Q = X · W_Q`, `K = X · W_K`, `V = X · W_V` (learned linear projections)
- **Q**: "What am I looking for?"
- **K**: "What do I offer?"
- **V**: "What do I actually contribute if selected?"
- **√d_k scaling**: prevents dot products from blowing up; otherwise softmax saturates and gradients vanish

**Multi-Head Attention**: run H attention heads in parallel, concatenate outputs, project via W_O. Each head learns different relationship types.

### Transformer Block Structure
```
Input → Embedding + Positional Encoding
          ↓
ENCODER (×N):
   Multi-Head Self-Attention
   Add & Norm  (residual + layer norm)
   FFN (2 linear + ReLU/GELU)
   Add & Norm
          ↓
DECODER (×N):
   Masked Multi-Head Self-Attention
   Add & Norm
   Cross-Attention (attends to encoder output)
   Add & Norm
   FFN
   Add & Norm
          ↓
Linear + Softmax → token probabilities
```

### What's Learned vs What's Fixed
| Component | Status |
|-----------|--------|
| Token embedding (W_e) | **Learned** |
| Positional encoding | Fixed (sin/cos) |
| W_Q, W_K, W_V | **Learned** |
| Scaled dot product, softmax | Fixed |
| W_O (output projection) | **Learned** |
| Layer norm γ, β | **Learned** |
| FFN weights W_1, W_2 | **Learned** |
| Activation (GELU/SwiGLU) | Fixed |
| Unembedding W_u | **Learned** |

### RLHF (Reinforcement Learning from Human Feedback)
Three-stage alignment after pre-training:
1. **SFT** (Supervised Fine-Tuning): fine-tune on human-written prompt+response pairs
2. **Reward Model**: humans rank response pairs A vs B; train RM via cross-entropy → scalar score r(x,y)
3. **PPO RL**: policy LLM generates y; RM scores it; PPO updates policy to maximize:
   ```
   Max E[r(x,y)] − β · KL(policy, SFT)
   ```
   The **KL penalty** stops reward hacking — keeps policy near the SFT baseline.

> PPO = Proximal Policy Optimization · RM = Reward Model · KL = Kullback-Leibler divergence

---

## Unit 4: How to Accelerate an Algorithm (Quiz Bullseye)

The professor's full list of 10 levers:
1. Technology scaling (Moore's law is flattening)
2. Cache optimizations
3. Code optimization
4. Exploiting parallelism
5. GPUs
6. TPUs
7. Fancier CPUs (more pipelining, faster storage, better networking)
8. **HW/SW co-design**
9. Improve the algorithm (trade time for space, approximations)
10. **Emerging technology**

If a quiz asks "list 5 ways", pick from this list. If it asks "highest speedup but lowest applicability", say **FPGA/ASIC**.

### Traditional Tech Speedup Table (memorize ranges)
| Technique | Speedup |
|-----------|---------|
| Algorithm selection | 10×–1000×+ |
| Data structures (hash vs array) | 2×–100× |
| GPU acceleration | 10×–1000× |
| Multithreading | 2×–64× |
| Compiler flags | 1.2×–4× |
| Memory hierarchy | 2×–10× |
| Memoization | 2×–100× |
| CPU upgrade | 1.5×–4× |
| SSD vs HDD | 2×–100× |
| **FPGA/ASIC** | **10×–1000×** (Very High difficulty, Low applicability) |
| Code micro-opts | 1.1×–2× |
| Python → C++ | 3×–50× |

### Emerging Tech Master Table
| Technology | Speedup | Energy gain | Readiness |
|------------|---------|-------------|-----------|
| **Memristors** | 10×–100× | 100×–1000× | **Med–High** |
| Neuromorphic chips | 10×–1000× | 1000×–10,000× | Medium |
| Quantum | Exponential (specific) | Varies | Low |
| Spintronics | 5×–50× | 10×–100× | Medium |
| Photonic | 100×–1000× | 100×–1000× | Low–Med |
| Memcapacitors | 10×–100× | 50×–500× | Low |
| Reservoir computing | 50×–500× (temporal) | 100×–1000× | Low–Med |
| DNA | Massive (theoretical) | Ultra-low | Very Low |
| Phase-change | 5×–50× | 10×–100× | Med–High |
| Protein nanowires | Unknown | **10,000×–100,000×** | Very Low |

> **Memristors are the most mature emerging tech AND the key building block for IMC.** Know that.

### No Free Lunch Theorem (Wolpert & Macready, 1997)
> Averaged over all possible problems, every optimization algorithm performs equally well.

- Does NOT say algorithms are equal on your problem — it says there's no universal winner.
- **HW implication**: specialized chips (TPU, neuromorphic) win in their domain and fail outside it.
- **Any "X is better than Y" claim must specify the problem class.**
- This is why ASIC = highest possible speedup AND lowest applicability — it wins only when algorithm structure matches custom silicon.

---

## Unit 5: In-Memory Computing (IMC)

### The Energy Problem
At 45-nm (Verma et al., 2019):
- DRAM 64-bit access: **~2 nJ (2,000 pJ)**
- INT4 multiply: ~0.1 pJ
- **20,000× more expensive to move data than compute it**

The moment weights don't fit in SRAM, energy is dominated by data movement.

### Conventional vs Computational Memory
| Conventional (von Neumann) | In-Memory |
|----------------------------|-----------|
| Data D moves to processor | Command "perform f on D" sent to memory |
| Processor computes f(D) | Memory itself returns f(D) |
| Result moves back | Bulk transfer eliminated |

### Memory Types
| Category | Examples |
|----------|----------|
| **Charge-based** | SRAM, DRAM, Flash |
| **Resistance-based** | RRAM, PCM, STT-MRAM |

| Property | RRAM | PCM | STT-MRAM |
|----------|------|-----|----------|
| Mechanism | Filament formation | Phase change | Magnetic spin torque |
| Write speed | ~10 ns | 50–100 ns | 2–10 ns |
| Endurance | 10⁶–10¹² | 10⁶–10⁸ | >10¹² |
| Analog (Multi-Level Cell) | Strong | Strong | Limited |
| Maturity | Research/production | Research/early prod. | Production (embedded) |

### IMC Maturity (TRL 7–8 / 9)
- **Software IMC** (Mature): in-memory databases like SAP HANA, Redis, Oracle TimesTen. DRAM holds 66.7% of IMC spend.
- **Digital CIM** (Early Commercial): UPMEM PIM-DRAM shipping, Samsung HBM-PIM, Micron CXL DIMMs. INT4/INT8 macros at 22–25 TOPS/W.
- **Analog IMC** (Pre-commercial): IBM AIMC, Mythic, Aspinity. Challenges: analog noise, precision, scaling.

Market: $16.3B (2024) → $46.9B by 2033 at 12.5% CAGR; AI inference is the growth driver.

### IMC Macro-Categories
| Static IMC (low-voltage) | Dynamic IMC (switching) |
|--------------------------|--------------------------|
| Matrix-vector multiplication | Supervised training |
| Inverse MVM | Spike-based learning |
| Content-addressable memory | Stateful logic |
| Combinatorial optimization | True random number generation |
| Bayesian inference | |
| Physical unclonable function | |

### IMC Performance Payoff
Verma et al.: **10× to 1000× gains** in both throughput (GOP/s/mm²) and energy efficiency (TOP/s/W) vs non-IMC.

---

## Unit 6: The Crossbar Primitive

### How It Computes MVM in One Cycle
A resistive crossbar performs **I = G·V** in a single read using:
- **Ohm's law**: I(i,j) = G(i,j) · V(i)
- **Kirchhoff's current law**: I(j) = Σᵢ G(i,j) · V(i)

**One column current = one dot product. Whole MVM in a single read cycle.**

### Mapping an NN Layer
- Weights W → conductances G(i,j) programmed into resistive memory cells
- Input vector → voltages V(i) applied on rows
- Output current per column = Σᵢ G(i,j)·V(i) = the dot product for that output neuron
- Backward pass: read in the opposite direction → system computes Wᵀ·δ

### Capacitive Variant (1C cell)
Same idea with **capacitances C(i,j)** instead of conductances. Charge at bitline j: Q(j) = Σᵢ C(i,j)·V(i). Avoids static leakage during MVM.

### Crosspoint Cell Structures
| Cell | Structure | Property |
|------|-----------|----------|
| **1R** | 1 passive resistor | Simplest, dense, but sneak path problem |
| **1S1R** | 1 selector + 1 resistor | Nonlinear selector blocks sneak paths, keeps density |
| **1T1R** | 1 transistor + 1 resistor | Per-cell selectivity, lower density |
| **1C** | 1 capacitor | No static leakage during MVM |

### Negative Weights (week 8, page 10 red question)
The crossbar does I = G·V and conductance is **always positive**. There's no such thing as a negative resistor. But NN weights need to be signed for excitatory plus inhibitory synapses. Three standard fixes:

1. **Differential pair (most common).** Store each weight as **w = G⁺ − G⁻** across two memristor columns. Output current is I = (G⁺ − G⁻)·V. About **2× area**, clean signed result with the same crossbar primitive. TrueNorth, Mythic, and most published memristor accelerators use this.
2. **Offset subtraction.** Store w_shifted = w + w_max so all values are positive, then subtract a fixed offset at readout. Just **1× area**, but you lose dynamic range and ADC noise on the offset eats you.
3. **Sign-bit + magnitude in separate arrays.** Sign array + magnitude array, combined in periphery. Highest area cost, but clean separation for digital post-processing.

If the prof asks which to pick: **differential pair** is the safe answer.

---

## Unit 7: Sneak Paths (CF06 CMAN Topic)

### What They Are
**Unintended current pathways through unselected cells in a memory array.** When you try to read one cell:
- Current flows through neighboring cells via shared rows and columns
- The sense amp reads I_intended + I_sneak
- It can't distinguish them — the dot product gets corrupted

### CF06 Walkthrough (2×2 case)
Given R[0][0]=R[1][1]=1 kΩ (on), R[0][1]=R[1][0]=2 kΩ (off), with V_row0=1V, V_col0=0V virtual ground:

**Ideal** (row 1 and col 1 grounded):
- I_col0 = (1V − 0V)/1kΩ + (0V − 0V)/2kΩ = **1.000 mA** ✓

**Sneak path** (row 1 and col 1 floating): KCL at V_row1 and V_col1 →
- V_row1 = 0.4 V
- V_col1 = 0.6 V
- Sneak path: row0 → R[0][1] → col1 → R[1][1] → row1 → R[1][0] → col0
- Sneak current adds 0.2 mA → **I_col0_actual = 1.200 mA (+20% error)**

### Solutions
1. **Diodes** at each cell → unidirectional current only
2. **1T1R structures** → raise only the selected row's word line V_GX, turning on only that row's transistors. Transistors in every other row are off, no sneak current can flow.

---

## Unit 8: Sparse MVM and Sparse-on-Crossbar

### Sparse MVM
**y = A·x where A is mostly zeros.** Real workloads (GNNs, recommenders, scientific PDEs) are 90–99% zero.

Storing/multiplying zeros wastes memory, bandwidth, energy. Compressed formats (CSR, CSC, COO) skip them entirely.

### CSR (Compressed Sparse Row)
Three arrays:
- `values[]` — the non-zeros, in row-major order
- `col_idx[]` — column index of each non-zero
- `row_ptr[]` — length N+1; row i's non-zeros sit at `values[row_ptr[i] .. row_ptr[i+1]]`

Memory: ~2·nnz + N values instead of N². Walks rows in order → ideal for y = A·x.

**CSR vs COO**: COO stores (row, col, val) for every NZ (3·nnz entries). CSR run-length-encodes the row coordinate.

### Sparse-on-Crossbar Problem
Naïve mapping of a sparse matrix to an 8×8 crossbar uses ~13% of the cells. What breaks:
- **Area**: every zero still consumes a programmed cell + wire pitch
- **Energy**: bitlines charge through all cells (including zero conductances)
- **Accuracy**: many tiny currents sum on each bitline; ADC range wasted on noise
- **Tile size fixed** (e.g. 128×128): big matrices need many tiles

### Four Mitigation Techniques
1. **Permute & pack** (SW) — reorder rows/columns so non-zeros cluster into dense blocks. Example: 8 NZ in 64 cells (12%) → 8 NZ in 9 cells (89%) → **7× smaller**.
2. **Tile partitioning** (SW) — split A into k×k sub-matrices; skip all-zero tiles entirely.
3. **Row & bitline gating** (HW) — power-gate rows whose input is zero and columns with no non-zeros. Cuts dynamic + leakage energy with no accuracy cost.
4. **Format-aware drivers** (HW) — stream x in CSR/CSC order; drive only rows matching active non-zero patterns.

### Compression Overhead
- Indexing: decode `row_ptr`/`col_idx`, extra reads per NZ
- Routing: scheduler maps each NZ to a tile + address
- Padding: tiles must be regular; partial rows waste cells
- Sync: skipped rows shift y; need reorder logic
- Bandwidth: sparse access breaks memory locality

**Crossover: below ~70% sparsity, a dense crossbar usually wins** — decoder + scheduler + irregular access cost more than the zeros you skipped.

**Pays off when**:
- Sparsity ≥90%
- Structured (block / N:M) patterns simplify decoder
- Weight reuse amortizes overhead
- Large matrices (overhead O(NZ), savings O(N²))

---

## Unit 9: Neuromorphic Chips

### Definition
**Specialized hardware designed to mimic the structure and function of the human brain.** Distinct from CPUs/GPUs in that compute and memory are integrated, processing is event-driven, learning can happen on-chip, and the system is fault-tolerant by design.

### What is NOT neuromorphic (week 8 red label, page 2)
**Cerebras WSE-3** is 4 trillion transistors on a 300 mm wafer, TSMC 5 nm, 57× larger than an H100. Looks neuromorphic. It's not.
- All dense matrix math for LLM training/inference, no spikes anywhere
- Clock-driven and synchronous, not event-driven
- Von Neumann dataflow at the core/tile level, just replicated 850,000 times
- No SNN, no STDP, no biological neuron model

Brute scale, not brain inspiration. Same trick on **IBM NorthPole** (slide p.38): IBM markets it as brain-inspired, but it's a dense INT2/4/8 inference chip with no spikes, no learning, no neurons. Structurally inspired by the brain, fine. Functionally neuromorphic, no.

### Four Pillars
1. **Parallel processing** — compute and memory integrated in the same physical structures
2. **Event-driven** — operate on spikes, not clock cycles → power only consumed when processing
3. **Adaptability** — on-chip learning (synaptic plasticity)
4. **Distributed memory** — information stored across the network in connection strengths

### Traditional vs Neuromorphic Comparison
| Feature | Traditional | Neuromorphic |
|---------|------------|--------------|
| Architecture | Von Neumann (separate compute/memory) | Brain-inspired (integrated) |
| Processing | Sequential, clock-driven | **Parallel, event-driven** |
| Power | High | Low to very low |
| Learning | Separate algorithms | Often on-chip |
| Precision | High arithmetic | Variable/lower, stochastic |
| Fault tolerance | Limited | **High (distributed)** |

### The Trade-off Spectrum
```
SpiNNaker  ──  Akida/Loihi/BrainScaleS-2  ──  IBM TrueNorth/NorthPole
Flexibility ─────────────────────────────► Performance
More digital                              More analog (but needs DAC/ADC)
```

Other axes (per Borghi et al. review): routing-based weights vs crossbar, off-chip vs on-chip learning, compute-near-memory vs compute-in-memory, edge vs data-center focus.

### Why not do everything in software? (week 8 red question, page 5)
**2 nJ for one DRAM access**, **0.1 pJ for an INT4 multiply**. **20,000× gap on memory alone**. Now scale to a billion-neuron SNN in real-time. SW on a CPU can't get there. Four reasons it dies in software:

- **Energy**: every spike in SW costs instruction fetch + decode + memory access. The memory part alone is the 20,000× hit above. Brain-scale SNNs infeasible before you even get to the compute.
- **Latency**: real-time sensing needs microsecond response. CPU SW emulation runs about **1000× slower** than biological real-time. BrainScaleS hits **10,000× faster** than real-time, but only because it's analog hardware.
- **Parallelism**: brain is massively parallel and event-driven, CPU is sequential and clock-driven. Most of the time you're idle waiting on memory anyway.
- **Sparse activity wasted in SW**: about **1% of neurons spike** per timestep, but SW pays full instruction overhead per neuron. Hardware can gate everything off until a spike arrives. SW can't.

That's the whole reason this course exists. When memory access and parallelism are the bottleneck, specialized HW is the only way through.

---

## Unit 10: AER + Network-on-Chip (NoC)

### Address Event Representation (AER)
**Spike-event message-passing protocol** used by virtually all neuromorphic hardware and SW sim environments.

- When a neuron spikes, the system sends only the **spiking neuron's address** as an event packet
- Each spike encoded as packet: `| DEST_CORE | NEURON_ID | TIMESTAMP |`
- Only active neurons generate messages — sparse, event-driven, asynchronous
- **Timing information is implicit** in when the message is sent
- AER enables async sparse communication between NM cores

### Network-on-Chip (NoC)
Packet-switched interconnect replacing buses and point-to-point wires.

**Components**:
- **Router**: arbitration, buffering, switching
- **Link**: physical wires between routers
- **NI** (Network Interface): packs/unpacks data

**Why use it**:
- **Scalability**: bandwidth grows with router count
- **Parallelism**: multiple concurrent transactions
- **Modularity**: drop-in IP blocks

Common topology: **2D mesh**. Others: torus, ring, tree, butterfly.

### AER over NoC — End-to-End Flow
1. **Neuron spikes** — emits its own address only (neurons are "dumb")
2. **Routing lookup** — local SRAM table at source: src → [dest list], holds synaptic fan-out
3. **Packet fan-out** — network interface emits one packet per destination (1 spike → N packets)
4. **NoC delivers** — routers forward by DEST coordinates (XY routing)

**Key insights**:
- Neurons only know their own ID
- Connectivity lives in SRAM, not wires
- Rewriting the table = rewiring the network (programmable topology)
- Source-side fan-out enables on-line learning (STDP)

Real chips:
- **SpiNNaker**: multicast routers
- **Loihi**: axon table
- **TrueNorth**: per-neuron destination list

---

## Unit 11: Major Neuromorphic Chips

### SpiNNaker / SpiNNaker2 (Univ. of Manchester)
- Approach: **simulate neurons on general-purpose ARM cores**
- SpiNNaker1 (130 nm): ~1M ARM9 cores across 1,200 boards
- SpiNNaker2 (22 nm FDSOI): ~5.2M ARM M4F cores, 720 boards; aim: 10M cores
- Per node: 18 ARM968 cores (1 spare), 96 kB local + 128 MB SDRAM, packet router
- 200 MHz cores, **no FPU → fixed-point arithmetic**
- Real-time event-driven programming model (SC&MP + SARK)

### IBM TrueNorth (2015)
- **65 mW, 5.4B transistors**, real-time neurosynaptic processor
- **1 million digital neurons + 256 million synapses** across 4,096 cores
- Non-von Neumann, defect-tolerant, event-driven
- Async router; sync neuron + SRAM; QDI async circuits
- Per-tick computation (1 kHz sync trigger):
  1. Core receives spikes, stores in input buffers
  2. On tick, spikes distributed across horizontal axons
  3. Synaptic connections deliver spike to neuron via dendrite
  4. Neuron integrates incoming spikes, updates membrane potential
  5. Leak value subtracted from membrane potential
  6. If potential > threshold → spike, sent into network
- Neuron model: **augmented integrate-and-fire** (simpler than Hodgkin-Huxley or Izhikevich)

### IBM NorthPole (2023) — 10 Synergistic Axioms
1. Specialized for neural inference (no data-dependent branching)
2. Biological precision: optimized for **8-, 4-, and 2-bit** low-precision
3. Distributed core array: 16×16, each core does **8192 2-bit ops/cycle**
4. **Distributed memory near and intertwined with compute** (data locality)
5. Two dense NoCs for compute/memory (gray-matter and white-matter inspired)
6. Two more NoCs for reconfiguring weights and programs
7. **Data-independent branching** → fully pipelined, stall-free, deterministic; no memory misses
8. Co-optimized training (low-precision constraints in training)
9. Codesigned software (compiler, validator, runtime)
10. Frame-based usage: write input frame, read output frame; runs independently of host

**Performance**: ResNet-50 at INT8/4/2: **42,460 FPS at 74 W** on 12 nm with 22B transistors. Competitive with H100 (81,292 FPS at 700 W on 4 nm with 80B transistors) on energy/space metrics.

### BrainScaleS / BrainScaleS-2 (Heidelberg, HBP)
Analog mixed-signal, accelerated.

| | BrainScaleS-1 | BrainScaleS-2 |
|--|---------------|---------------|
| Integration | Wafer-scale | Single-chip ASIC |
| Neurons | 196,608/wafer | 512/chip |
| Synapses | ~44M/wafer | 212–217k plastic/chip |
| Speed | **1,000–10,000× biological real-time** | 1,000× real-time |
| Process | 180 nm | 65 nm |
| Plasticity | Fixed models | **Programmable via PPU (custom CPU)** |
| Neuron model | Adaptive spiking | **Adaptive Exponential Integrate-and-Fire (AdEx)** |

**AdEx equations** (BrainScaleS-2):
```
C_m · dV/dt = -g_l(V-E_l) + g_l·Δ_T·exp((V-V_T)/Δ_T) - w + I
τ_w · dw/dt = a(V-E_L) - w
```

Each quadrant: synaptic crossbar (256 rows × 128 columns), neuron circuits, analog parameter storage, PPU plasticity processor.

### Loihi / Loihi 2 (Intel)

| | Loihi (2017) | Loihi 2 (2021) |
|--|--------------|----------------|
| Process | 14 nm | Intel 4 |
| Die | 60 mm² | 31 mm² |
| Cores | 128 NM + 3 x86 | 128 NM + 6 Lakemont x86 |
| Neurons | ~130,000 | **Up to 1M (~8× more)** |
| Synapses | ~130M | Up to 120M |
| On-chip SRAM | >33 MB | ~25 MB |
| Neuron model | Fixed LIF | **Programmable via microcode** |
| Spike speed | Baseline | **10× faster** |
| Spike events | 1-bit | **Up to 32-bit graded** |
| Connectivity | FPGA-mediated | 10G Ethernet, GPIO, SPI |
| Framework | None at launch | **Lava (open-source)** |

**Loihi core microarch**: SYNAPSE → DENDRITE (updates compartment state u, v) → SOMA spike gen → AXON fan-out → LEARNING (reconfigurable engine, updates weights at epoch boundaries).

**Loihi computational primitives**:
- Stochastic noise (Markov chain Monte Carlo support)
- Configurable delays (polychronous dynamics)
- Configurable dendritic tree processing
- Threshold adaptation (homeostasis)
- Weight scaling/saturation (permanence levels beyond inference)

**LLM on Loihi 2** (Abreu et al., ICLR 2025 SCOPE workshop, arXiv:2503.18002v2): MatMul-free 370M-parameter LLM quantized with no accuracy loss. **3× higher throughput with 2× less energy** vs transformer LLMs on an edge GPU.

#### Why that gain is NOT impressive (week 8 red question, page 53)
Sounds good. It's actually really weak. Five things wrong with it:

- **3× throughput is tiny** for custom silicon vs general-purpose GPU. ASICs and NPUs routinely show 10× to 1000× over a CPU/GPU baseline (see emerging-tech table, Unit 4). 3× means Loihi 2's silicon advantage is barely showing up.
- **2× energy is also weak.** Loihi is supposed to be about **1000× more energy-efficient than GPUs** (academic chips table literally says this). Hitting only 2× means the workload isn't using what makes neuromorphic special.
- **The model was rewritten** to fit Loihi 2's constraints (MatMul-free, HGRN, RMSNorm, BitLinear from Zhu et al. 2024). Not a normal transformer, a paper-thin neuromorphic-friendly variant.
- **Baseline is an edge GPU**, not a datacenter GPU. Smaller win against a smaller comparison.
- **Only 370 M params**, well below where LLMs need to run (7B+). At usable scale the result might collapse entirely.

**Why LLMs don't fit on a spiking chip at all:** transformer attention needs matrix-matrix multiply (un-brain-like). Spikes can't train with backprop (no global gradient signal, only STDP, local-only, no convergence guarantee). SpikeGPT and BitLinear exist precisely because natural transformer math doesn't fit on a neuromorphic substrate.

**Punchline**: 3× / 2× is what you'd get from a moderately good software optimization, not from a fundamentally different chip architecture. Loihi 2's real win shows up on sparse event-driven workloads (image segmentation, keyword spotting, robot control, see Loihi EDP scatter on slide p.46). LLM inference is the wrong workload for a spiking chip.

> Loihi gives better EDP on LASSO, graph search, k-NN, constraint satisfaction, sequential MNIST. But **NOT in general competitive with general AI/ML workloads** (MobileNet, image segmentation).

### Akida (BrainChip, 2021)
- **First commercial neuromorphic processor**
- Event-based, sub-mW edge AI
- 1.2M neurons, 10B synapses
- Supports CNNs, RNNs, and **Temporal Event-based Neural Networks (TENNs)** — combine spatial + temporal convolutions
- Free chip emulator, TensorFlow-compatible MetaTF framework

---

## Unit 12: Neuromorphic Transformer + SNN Concepts

### Neuromorphic Transformer
Replaces **MAC (multiply-accumulate)** with **AAC (AND-accumulate)**:
- Q/K/V matrices become **binary spikes (0 or 1)**
- AND replaces multiply; accumulate with integer addition
- Eliminates: softmax, scaling by √d_k, matrix transposition

**Impact**: **99.96% reduction in multiplications** — 116M → 4,900 multiplications. Based on Spiking Neural Networks. Human brain analogy: ~20W, spike-based, temporal encoding.

### ANN vs SNN
| | ANN | SNN |
|--|-----|-----|
| Activation | Continuous (float) | Binary spike |
| Computation | Dense MACs | Sparse AND-accumulate |
| Time | Per-token, static | Continuous, spike timing matters |
| Energy | High | Very low |

### Current Limitations
- **Programming models** — lack of standardized frameworks (no CUDA equivalent)
- **Software ecosystem** — limited vs traditional
- **Scaling** — hard to reach human-brain scale efficiently
- **Algorithm mapping** — converting existing AI to SNN is non-trivial
- **Adoption** — early research
- **Cost** — SRAM is ~2 orders of magnitude more expensive than DRAM

### The "AlexNet Moment"
Kudithipudi et al. (2024) argue NM computing is approaching its AlexNet moment — the breakthrough that catalyzes the field. Like AlexNet's 2012 win was enabled by GPUs, an NM breakthrough may come from identifying the right small-scale HW configuration.

---

## Unit 13: CUDA MLP Mapping

### Parallelism Types for MLP on GPU
| Type | What's parallelized | When to use |
|------|---------------------|-------------|
| **Data parallelism** | Multiple training examples | Most common — batch dimension |
| **Model parallelism** | Partition network across GPUs | Models that don't fit in one GPU |
| **Layer-wise** | Neurons within a layer | Per-layer kernel launch |

### Kernel Design
- Separate kernel per layer → each layer gets optimal thread config
- Layer 1: `threadsPerBlock(8, 5)` — 8 batches × 5 hidden neurons
- Layer 2: `threadsPerBlock(16, 1)` — 16 batches × 1 output neuron
- Trade-off: **kernel launch overhead** vs **register pressure** (too many threads/block → register spills)
- Fuse ops with the same data/parallelism pattern

### Memory Flow
```
cudaMalloc      → allocate device memory
cudaMemcpy H→D  → copy input/weights to GPU
kernel<<<M,T>>> → launch M blocks, T threads/block
cudaMemcpy D→H  → copy results back
cudaFree        → release device memory
```

---

## Unit 14: Last-Minute Checklist

- [ ] Can I draw a systolic array and explain the dataflow?
- [ ] Can I derive the **3N − 2** cycle count for N×N matmul?
- [ ] Can I list the **three dataflows** and say which wins on energy (row-stationary)?
- [ ] Can I write the **TPU MXU = 256 × 256 = 65,536 MACs/cycle**?
- [ ] Can I explain BF16's bit layout and why it's safer than FP16 for training?
- [ ] Can I write **Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V** from memory?
- [ ] Can I explain why transformers are NOT recurrent?
- [ ] Can I list **5 ways to accelerate an algorithm**?
- [ ] Can I state the No Free Lunch theorem and one HW implication?
- [ ] Can I write **I(j) = Σᵢ G(i,j) · V(i)** (crossbar MVM)?
- [ ] Can I list the 4 cell structures (1R, 1S1R, 1T1R, 1C)?
- [ ] Can I explain sneak paths and the two solutions (diodes, 1T1R)?
- [ ] Can I solve a 2×2 KCL sneak path problem (CF06 style)?
- [ ] Can I list the 4 sparse-on-crossbar techniques and the **70% crossover**?
- [ ] Can I write CSR's three arrays (values, col_idx, row_ptr)?
- [ ] Can I describe the AER packet format and why neurons are "dumb"?
- [ ] Can I list the components of a NoC (router, link, NI)?
- [ ] Can I name the top 4 neuromorphic chips and one distinguishing fact about each?
- [ ] Can I explain what NorthPole's data-independent branching means?
- [ ] Can I explain the Neuromorphic Transformer (MAC → AAC, 99.96% reduction)?
- [ ] Can I write the LIF neuron equation (u[t] = λu[t-1] + a[t])?
- [ ] Can I set up CUDA thread dimensions for an MLP layer?

### Week 8 red-text questions (same priority as QUIZ-stickered material)

- [ ] Can I explain why **Cerebras WSE-3 and IBM NorthPole are NOT neuromorphic** (no spikes, clock-driven, Von Neumann at the tile level, no STDP)?
- [ ] Can I justify **why we don't do everything in software** (DRAM 2 nJ vs INT4 mult 0.1 pJ = 20,000× memory gap; sparse activity wasted on a CPU; 1000× slower than biological real-time)?
- [ ] Can I list **3 ways a crossbar handles negative weights** (differential pair w = G⁺−G⁻, offset subtraction, sign+magnitude) and pick differential pair as the standard?
- [ ] Can I **critique the LLM-on-Loihi-2 3×/2× result** (expected ~1000×, model rewritten, edge GPU baseline, only 370M params, wrong workload for a spiking chip)?
