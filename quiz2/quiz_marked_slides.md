# Quiz 2 — Ultimate Notesheet (QUIZ-Marked Slides Only)
*Source: every slide in weeks 5–7 stamped with a red "QUIZ" sticker.*

> **If you only have time to study one document, study this one.** These are the slides the professor explicitly flagged as quiz material.

---

## 🎯 Week 6 QUIZ Slides — `w6_mon_transformers_in_memory.pdf`

### Q6.1 — How can we accelerate an algorithm?
The 10 levers, in order. **Know all of them; quiz can ask "list 5 ways to speed up an algorithm."**

1. Technology scaling (Moore's law is flattening out)
2. Cache optimizations
3. Code optimization
4. Exploiting parallelism
5. GPUs
6. TPUs
7. Fancier CPUs (more pipelining, faster storage, better networking)
8. HW/SW co-design
9. Improve the algorithm (trade time for space, approximations)
10. Emerging technology

### Q6.2 — Traditional technology table (with values)
| Technique | Speedup | Difficulty | Applicability |
|-----------|---------|------------|---------------|
| Algorithm selection | 10×–1000×+ | Med–High | High |
| Data structure optim. | 2×–100× | Med | High |
| GPU acceleration | 10×–1000× | High | Med |
| Multithreading | 2×–64× | High | Med |
| Compiler flags | 1.2×–4× | Low | High |
| Memory hierarchy | 2×–10× | Med | High |
| Memoization | 2×–100× | Low–Med | Med |
| CPU hardware upgrade | 1.5×–4× | Low | High |
| SSD vs HDD | 2×–100× | Low | Med (I/O-bound) |
| **FPGA/ASIC** | **10×–1000×** | **Very High** | **Low** |
| Code micro-opts | 1.1×–2× | Med | High |
| Language change (Py→C++) | 3×–50× | High | Med |

### Q6.3 — No Free Lunch (NFL) theorem
> Averaged across all possible problems, **all optimization algorithms perform equally well** — no algorithm consistently outperforms random search.

- Formalized by **Wolpert & Macready, 1997**
- **HW implication**: specialized HW (TPU, neuromorphic) wins only in its domain
- Any superiority claim **must specify the problem class** — context matters
- Why FPGA/ASIC = highest speedup but lowest applicability

### Q6.4 — Systolic array
- Name from **heart analogy**: data "pulses" rhythmically through PEs
- CPUs/GPUs spend energy accessing many registers per op
- Systolic array **chains ALUs** — one register read feeds many MACs → minimal data movement
- PE: register X, register Y → multiplication → accumulator → register Z
- 2D grid: activations left→right, weights stationary, partial sums downward

### Q6.5 — Mapping a DNN onto a systolic array
- Weight Memory (on-chip SRAM) preloads weights w₁₁..wNN
- Activations stream from Activation Memory left to right
- Partial sums accumulate downward into Accumulators
- MAC unit: **8-bit a × 8-bit w → 16-bit product → 24-bit accumulator**

---

## 🎯 Week 7 QUIZ Slides — `w7_mon_neuromorphic_chips.pdf`

### Q7.1 — Emerging technology table (the "10 emerging tech" master table)
**Know which has highest speedup, which has highest energy efficiency, which is most mature.**

| Technology | Speedup | Energy gain | Readiness | Key advantage |
|------------|---------|-------------|-----------|---------------|
| **Memristors** | 10×–100× | 100×–1000× | **Med–High** | In-memory computing |
| Neuromorphic Chips | 10×–1000× | 1000×–10,000× | Medium | Spike-based, parallel |
| Quantum | Exponential (specific) | Varies | Low | Quantum parallelism |
| Spintronics | 5×–50× | 10×–100× | Medium | Non-volatile, minimal heat |
| Photonic | 100×–1000× | 100×–1000× | Low–Med | Ultra-high BW |
| Memcapacitors | 10×–100× | 50×–500× | Low | Complementary to memristors |
| Reservoir computing | 50×–500× (temporal) | 100×–1000× | Low–Med | Time-series |
| DNA | Massive (theoretical) | Ultra-low | Very Low | Molecular parallelism |
| Phase-change materials | 5×–50× | 10×–100× | Med–High | Non-volatile multi-level |
| Protein nanowires | Unknown | 10,000×–100,000× | Very Low | Ultra-low power |

### Q7.2 — Computing in processor vs. computing in memory
- **Conventional**: data D flows out to processor, result f(D) flows back
- **In-memory**: command "perform f on D" sent to memory; memory itself returns f(D)
- Memory types:
  - **Charge-based**: SRAM, DRAM, Flash
  - **Resistance-based**: RRAM, PCM, STT-MRAM

| Property | RRAM | PCM | STT-MRAM |
|----------|------|-----|----------|
| Mechanism | Filament | Phase change | Magnetic spin |
| Write speed | ~10 ns | 50–100 ns | 2–10 ns |
| Endurance | 10⁶–10¹² | 10⁶–10⁸ | >10¹² |
| Analog (MLC) | Strong | Strong | Limited |

### Q7.3 — MVM via crossbar (equation)
A crossbar performs MVM via universal circuit laws:

**Ohm's law:** I(i,j) = G(i,j) · V(i)
**Kirchhoff's current law:** I(j) = Σᵢ G(i,j) · V(i)
**Matrix form:** **I = G·V** — full MVM in **one read cycle**

### Q7.4 — Crossbar cell structures
Four cell types, know what each does:

| Cell | Components | Purpose |
|------|-----------|---------|
| **1R** | 1 resistor | Simplest, sneak path problem |
| **1S1R** | 1 selector + 1 resistor | Nonlinear selector blocks sneak paths, keeps density |
| **1T1R** | 1 transistor + 1 resistor | Best selectivity, lower density (rows turn on individually via gate voltage) |
| **1C** | 1 capacitor | No static leakage during MVM |

### Q7.5 — How to map an NN onto a crossbar
- **Weights W → conductances G(i,j)** programmed into resistive memory
- **Input vector → voltages V(i)** applied on rows
- **Output current per column** = Σᵢ G(i,j)·V(i) = dot product for that output neuron

**Why it works**:
- **Parallel multiplication** at each junction (one resistor per weight)
- **Current summation** via Kirchhoff on each bitline
- Forward = read in one direction; backward (Wᵀ·δ) = read in other direction

### Q7.6 — Sneak paths (4 slides)
**Sneak paths = unintended current pathways through unselected cells in a crossbar.**

- Floating row/column nodes let current loop through neighboring cells
- The sense amp reads I_intended + I_sneak — corruption you can't distinguish
- In a large array, every floating node adds more sneak loops → all dot products wrong
- **Solutions**: **diodes** (enforce unidirectional current) or **1T1R structures** (only the selected row's transistors turn on)

This is **exactly CF06 CMAN**. Walk through the 2×2 KCL solution: V_row1 = 0.4 V, V_col1 = 0.6 V, sneak adds +0.2 mA to a 1 mA signal (20% error).

### Q7.7 — Crossbar primitive (analog MVM in O(1) time)
- Program weights as conductances G(i,j) in ReRAM/PCM
- Apply input vector as row voltages V(i)
- Ohm + Kirchhoff → I(j) = Σᵢ G(i,j)·V(i)
- **One column current = one dot product**
- **Whole MVM in a single read cycle** (O(1) latency, not O(N))

### Q7.8 — Efficient Sparse Mapping on a Crossbar
Real matrices are 90–99% zeros. Naïve mapping = 13% utilization. Four techniques:

1. **Permute & pack** (SW) — reorder rows/columns to cluster non-zeros into dense blocks
   - Example: 8 NZ in 64 cells (12%) → 8 NZ in 9 cells (89%) = **7× smaller**
2. **Tile partitioning** (SW) — split A into k×k tiles, skip all-zero tiles entirely
3. **Row & bitline gating** (HW) — power-gate zero-input rows and empty columns
4. **Format-aware drivers** (HW) — drive in CSR/CSC order, hit only active non-zeros

**Crossover**: **below ~70% sparsity, dense crossbar usually wins** — overhead beats savings.

### Q7.9 — Neural network acceleration ecosystem
Four pillars of NN accelerators:

1. **GPU architectures** — parallel SIMT, HBM, inference + training (CNNs, transformers)
2. **Domain-specific (TPUs/NPUs)** — matrix units, specialized dataflow, inference + training (CNNs, LLMs)
3. **FPGA-based** — flexible datapath, custom precision, inference (RNNs, GNNs)
4. **ASIC & Emerging** — fixed function, in-memory compute, inference (specific models)

All connect through: dense linear algebra, sparsity support, attention kernels, mixed precision, optimized memory hierarchy.

### Q7.10 — Key building block: the crossbar (for NM chips)
Crossbar arrays provide the critical architecture for efficiently implementing synaptic connections.

**Why**:
- Efficient matrix-vector multiplication
- In-memory computing
- Implementation of synaptic weights
- Parallel processing
- Integration with emerging technologies

### Q7.11 — Address Event Representation (AER)
**AER = spike-event message-passing protocol for Network-on-Chip in neuromorphic chips.**

- Almost all NM hardware and SW sim environments use a variant of AER
- Simplest impl: a firing neuron sends its **unique ID** to all nodes holding any of its targets
- Each spike encoded as packet: `| DEST_CORE | NEURON_ID | TIMESTAMP |`
- Only active neurons generate messages — **sparse, event-driven, asynchronous**
- **Timing is implicit** in when the message is sent
- AER enables async sparse communication between NM cores
- NoC routes these spike event messages between processing elements

### Q7.12 — Network-on-Chip (NoC)
**Packet-switched interconnect** replacing buses and point-to-point wires.

**Components**:
- **Router**: arbitration, buffering, switching
- **Link**: physical wires between routers
- **NI** (Network Interface): packs/unpacks data

**Why use it**:
- **Scalability** — bandwidth grows with router count
- **Parallelism** — multiple concurrent transactions
- **Modularity** — drop-in IP blocks

**Default topology**: 2D mesh (also: torus, ring, tree, butterfly, hierarchical)

### Q7.13 — AER over NoC example (concrete packet)
Routing a spike from neuron (2,0) on Core A to neuron (5,3) on Core E:

```
AER packet:   | DEST_CORE | NEURON_ID | TIMESTAMP |
Spike A→E:    DEST = E (2,1)
              NID  = neuron #(5,3)
              T    = 1.247 ms
              Routed via XY: east → east → south
```

Sparse + asynchronous: routers fire only when a spike occurs → low average power, high peak BW. Used in **IBM TrueNorth, Intel Loihi, SpiNNaker**.

### Q7.15 — CSR (Compressed Sparse Row) — from CF7 lecture (Wed)

**Three arrays**:
- `values[]` — the non-zeros, in row-major order (one FP32 each)
- `col_idx[]` — column index of each non-zero (one INT32 each, length = nnz)
- `row_ptr[]` — bookmark into the k-indexed arrays; length **N+1**, with the last entry = nnz (sentinel)

**How to read row i**: slice `row_ptr[i] .. row_ptr[i+1]` gives row i's non-zeros.
- Row 1 example: `row_ptr[1..2] = [1,2]` → one NZ at k=1: `values[1]=8` at `col_idx[1]=2`

**Memory cost**: ~2·nnz + N values instead of N². Walk rows in order → ideal for y = A·x.

**vs COO** (Coordinate format): COO stores (row, col, val) for every NZ = 3·nnz entries. CSR keeps `col_idx` explicit but **run-length-encodes the row coordinate** into `row_ptr` (N+1 entries, not nnz).

### Q7.16 — CSR worked example: 4×4 matrix with 2 NZ per row

Matrix A (8 non-zeros total):
```
     j=0  j=1  j=2  j=3
i=0   5    0    0    3
i=1   0    8    2    0
i=2   6    0    0    9
i=3   0    1    0    7
```

CSR arrays:
- `values  = [5, 3, 8, 2, 6, 9, 1, 7]`  (k=0..7)
- `col_idx = [0, 3, 1, 2, 0, 3, 1, 3]`  (column of each NZ)
- `row_ptr = [0, 2, 4, 6, 8]`           (where each row starts in `values`, sentinel = 8)

**How to read row 2**: `row_ptr[2..3] = [4,6]` → NZs at k=4, 5. `values[4]=6` at `col_idx[4]=0`; `values[5]=9` at `col_idx[5]=3`.

**Storage**: 8 + 8 + 5 = 21 ints vs 16 for dense. **CSR wins as N grows** (overhead is O(N); savings are O(N²)).

> **`row_ptr[i]` answers: "what value of k does row i start at?"** It's a bookmark into the k-indexed arrays.

### Q7.17 — Reconstructing A from CSR

Algorithm:
```
for i in 0..N-1:
    for k in row_ptr[i] .. row_ptr[i+1]-1:
        A[i][col_idx[k]] = values[k]
```

Trace on the 4×4 example:
- **i=0**: `row_ptr[0..1] = [0,2]` → k=0,1 → `A[0][0]=5`, `A[0][3]=3`
- **i=1**: `row_ptr[1..2] = [2,4]` → k=2,3 → `A[1][1]=8`, `A[1][2]=2`
- **i=2**: `row_ptr[2..3] = [4,6]` → k=4,5 → `A[2][0]=6`, `A[2][3]=9`
- **i=3**: `row_ptr[3..4] = [6,8]` → k=6,7 → `A[3][1]=1`, `A[3][3]=7`

All 8 NZs placed; A fully reconstructed.

### Q7.14 — How does the source know the destination?
**It doesn't — a routing table at the source core looks up the fan-out.**

Four steps:
1. **Neuron spikes** — neuron (2,0) on Core A emits its own address only
2. **Routing lookup** — local SRAM table maps src → [dest list], stores synaptic fan-out
3. **Packet fan-out** — network interface emits one packet per destination (1 spike → N packets)
4. **NoC delivers** — routers forward by DEST coordinates via XY routing

**Key insights**:
- **Neurons are "dumb"** — only know their own ID
- **Connectivity = table** — synapses live in SRAM, not in wires
- **Programmable topology** — rewrite the table = rewire the network
- **Fan-out at source** — enables on-line learning (STDP)

Real chips: SpiNNaker (multicast routers), Loihi (axon table), TrueNorth (per-neuron dest list).

---

## 🎯 Week 8 — Red-Text Questions from `w8_mon_neuromorphic_chips.pdf`

These are the questions the professor wrote **in red font** on the Week 8 deck. Same status as a QUIZ sticker — if it's red, expect it to come up.

### Q8.1 — Why is Cerebras WSE-3 *not* considered a neuromorphic chip? (page 2)
Cerebras WSE-3 is **4 trillion transistors on a 300 mm wafer, TSMC 5 nm, 57× larger than an H100**. Looks like it should be neuromorphic. It's not, and the slide flags that in red.

**Answer.** Neuromorphic means brain-inspired *processing*, so event-driven spikes, integrated compute and memory, on-chip learning, distributed connectivity. Cerebras WSE-3 misses on every one.

- All **dense matrix math** for LLM training and inference, no spikes anywhere.
- **Clock-driven and synchronous**, not event-driven.
- **Von Neumann dataflow** at the core/tile level, just replicated 850,000 times.
- No SNN, no STDP, no biological neuron model.

So it's brute scale, not brain inspiration. Same trick the slides pull later on **IBM NorthPole** (p.38): IBM markets it as brain-inspired, but it's actually a dense INT2/4/8 inference chip with no spikes, no learning, no neurons. Structurally inspired by the brain, fine. Functionally neuromorphic, no.

### Q8.2 — Why not do everything in software? (page 5, "The usual trade-offs…!")
**2 nJ for one DRAM access**, **0.1 pJ for an INT4 multiply**. **20,000× gap on memory alone**. Now scale that to a billion-neuron SNN running in real-time. SW on a CPU can't get there.

**Answer.** Four reasons it dies in software.

- **Energy**: every spike in SW costs instruction fetch + decode + memory access. The memory part alone is the 20,000× hit above. Brain-scale SNNs become infeasible before you even get to the compute.
- **Latency**: real-time sensing needs microsecond response. CPU SW emulation runs about **1000× slower than biological real-time**. For example BrainScaleS hits **10,000× faster** than real-time, but only because it's analog hardware.
- **Parallelism**: brain is massively parallel and event-driven, CPU is sequential and clock-driven. Most of the time you're idle waiting on memory anyway.
- **Sparse activity is wasted in SW**: about **1% of neurons spike** per timestep, but SW still pays full instruction overhead per neuron. Hardware can gate everything off until a spike arrives. SW can't.

So that's the whole reason this course exists. When memory access and parallelism are the bottleneck, **specialized HW is the only way through.** The other trade-offs on the slide (power, reliability, programmability, on-chip learning, cost, scalability) push you toward more analog as you move right, more digital as you move left.

### Q8.3 — What to do with negative weights? (page 10, crossbar slide)
The crossbar does **I = G · V**, and conductance G is **always positive**. There's no such thing as a negative resistor. But NN weights need to be **signed** for excitatory plus inhibitory synapses. So how do you get a negative weight out of a non-negative device?

**Answer.** Three standard fixes.

1. **Differential pair encoding (most common).** Store each weight as **w = G⁺ − G⁻** across two memristor columns. Output current is I = (G⁺ − G⁻)·V. About **2× area**, but you get a clean signed result with the same crossbar primitive. This is what TrueNorth, Mythic, and most published memristor accelerators use.
2. **Offset subtraction.** Store w_shifted = w + w_max so all values are positive, then subtract a fixed offset at readout. Just **1× area**, but you lose dynamic range and the ADC noise on the offset eats you.
3. **Sign-bit + magnitude in separate arrays.** One array stores the sign, another stores the magnitude, combine in the periphery. **Highest area cost**, but clean separation for digital post-processing.

The slide hints at the differential pair scheme via the symmetric conductances drawn in the diagram (G_{i,j} on top, R̄_x on bottom, both feeding a subtracting summing amp). So if the prof asks you to pick one, **differential pair** is the safe answer.

### Q8.4 — Why is that gain performance not impressive? (page 53, "LLM on Loihi 2")
Abreu et al. (ICLR 2025 SCOPE workshop, arXiv:2503.18002v2) ran a **MatMul-free 370 M-parameter LLM on Intel Loihi 2** and reported **3× throughput, 2× less energy** vs transformer-based LLMs on an edge GPU. The slide highlights it in yellow. Sounds good. It's actually really weak.

**Answer.** Five things wrong with this number.

- **3× throughput is tiny** for a custom-silicon vs general-purpose-GPU comparison. ASICs and NPUs routinely show **10× to 1000×** over a CPU/GPU baseline (see Q7.1 emerging-tech table). 3× means Loihi 2's silicon advantage is barely showing up.
- **2× energy is also weak.** Loihi is supposed to be about **1000× more energy-efficient than GPUs** (the academic chip table literally says this). Hitting only 2× means the workload isn't using what makes neuromorphic special.
- **The model was rewritten to fit Loihi 2's constraints.** MatMul-free, HGRN, RMSNorm, BitLinear from Zhu et al. 2024. Not a normal transformer, a paper-thin neuromorphic-friendly variant.
- **The baseline is an edge GPU**, not a datacenter GPU. Smaller win against a smaller comparison.
- **Only 370 M parameters**, well below the scale where LLMs actually need to run (7 B and up). At usable scale the result might collapse entirely.

**Why it doesn't fit on a spiking chip at all.** Transformer attention needs matrix-matrix multiply (slide p.55), so it's fundamentally un-brain-like. Spikes can't train with backprop because there's no global gradient signal, just STDP, which is local-only with no convergence guarantee. So SpikeGPT, BitLinear, the whole "MatMul-free" line of work exists precisely because the natural transformer math **doesn't fit on a neuromorphic substrate.**

So the punchline: 3× / 2× is what you'd get from a moderately good software optimization, not from a fundamentally different chip architecture. Loihi 2's actual win shows up on **sparse, event-driven workloads** (image segmentation, keyword spotting, robot control, see Loihi EDP scatter on p.46). LLM inference is the wrong workload for a spiking chip.

---

## 🎯 Week 5 QUIZ Topics (inferred from quiz2 cheatsheet)

Week 5 slides aren't visually marked in my review, but the quiz2 cheatsheet flags these as core quiz material from `w5_mon_tpu_gpu_transformers.pdf`:

### Q5.1 — TPU MXU
- **256 × 256 = 65,536 MACs per cycle**
- 700 MHz → 92 TOPS (INT8)
- INT8 math; inputs from Unified Buffer + weights from Weight FIFO
- Output → Accumulators → Activation → Normalize/Pool
- **Weight-stationary dataflow**

### Q5.2 — TPU vs CPU vs GPU
| | CPU | GPU | TPU |
|--|-----|-----|-----|
| Predictions/sec | 5,482 | 13,194 | **225,000** |
| Perf/Watt | 1× | 2.9× | **83×** |

### Q5.3 — Three systolic dataflows
| Dataflow | What's fixed | Used by |
|----------|--------------|---------|
| **Weight Stationary** | Weights in PEs | Google TPU v1–v4 |
| **Output Stationary** | Partial sums in PEs | ShiDianNao |
| **Row Stationary** | Filter row + reuse | Eyeriss — **10× fewer DRAM accesses, wins on energy** |

### Q5.4 — Why transformers are NON-recurrent
| | RNN/LSTM | Transformer |
|--|----------|-------------|
| Processing | Sequential | **Parallel — all tokens at once** |
| Memory | Hidden state | **No hidden state — all context in attention** |
| Training | Can't parallelize over time | Fully parallelizable |
| Long-range | Vanishing gradients | Direct attention to any token |

Key: **self-attention replaces recurrence**; positional encoding (sin/cos) replaces time order.

### Q5.5 — Self-attention Q/K/V
```
Q = "What am I looking for?"
K = "What do I offer?"
V = "What do I actually contribute?"

Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
```
- **√d_k scaling**: stops softmax saturation
- Multi-head: H parallel heads, concatenate, project
- Masked attention in decoder: hides future tokens

### Q5.6 — BF16 (Brain Float 16)
```
FP32:  1 | 8 exp | 23 mantissa
FP16:  1 | 5 exp | 10 mantissa   ← narrow exp, training overflow risk
BF16:  1 | 8 exp | 7  mantissa   ← same range as FP32, safer
```
- Developed by **Google Brain** for TPU training
- Trivial conversion: FP32 ↔ BF16 = drop/add the last 16 mantissa bits

---

## ⚡ One-Page Quick Reference

| Topic | Killer fact |
|-------|-------------|
| TPU MXU | 256×256 = 65,536 MACs/cycle, 83× perf/Watt vs CPU |
| Systolic cycles | 3N−2 for N×N matmul |
| Best dataflow on energy | **Row-stationary** (Eyeriss) — 10× fewer DRAM accesses |
| BF16 exp/mantissa | 8/7 (vs FP32: 8/23, vs FP16: 5/10) |
| Transformer key replacement | **Recurrence → self-attention** |
| Attention formula | softmax(QKᵀ/√d_k)·V |
| NFL theorem | No algorithm wins universally — specify problem class |
| Crossbar MVM | **I = G·V** in **one read cycle** via Ohm + Kirchhoff |
| Sneak path | Floating nodes leak current; fixed with **diodes** or **1T1R** |
| Cell structures | 1R, 1S1R, 1T1R, 1C |
| Sparse crossbar crossover | **70% sparsity** (below: dense wins) |
| AER packet | DEST_CORE \| NEURON_ID \| TIMESTAMP |
| NoC default topology | 2D mesh |
| Routing | Source has SRAM table src→[dest list]; routers do XY |
| DRAM energy vs INT4 mult | ~2 nJ vs ~0.1 pJ → 20,000× more expensive |
| IMC speedup | 10×–1000× throughput AND energy |
| Highest energy efficiency emerging tech | Protein nanowires (10,000–100,000×, but Very Low TRL) |
| Most mature emerging tech | Memristors (Med–High TRL) |
| Memristor key advantage | In-memory computing, analog synaptic behavior |
| Top NM chips | Loihi, TrueNorth, NorthPole, BrainScaleS, SpiNNaker, Akida |
| Cerebras / NorthPole are NOT neuromorphic | Dense math, no spikes, no STDP — brain-*structured* not brain-*inspired* |
| Why not all-SW? | Memory access ~2 nJ vs INT4 mult 0.1 pJ → 20,000×; sparse activity wasted on a CPU |
| Negative weights on a crossbar | **Differential pair** w = G⁺ − G⁻ (2× area, clean), or offset subtraction, or sign+mag |
| LLM-on-Loihi-2 result | 3× throughput, 2× energy vs edge GPU — *weak* for a purpose-built chip (expected ~1000×) |

---

## ✅ "Could I answer this on the quiz?" checklist

- [ ] List 5 ways to accelerate an algorithm
- [ ] State the NFL theorem and one HW implication
- [ ] Draw a systolic array and explain MAC dataflow
- [ ] Compute output of a 2×2 systolic array for a given input (CF05 style)
- [ ] Identify the 3 dataflows and which wins on energy
- [ ] Write the crossbar MVM equation: I(j) = Σᵢ G(i,j)·V(i)
- [ ] Draw the 4 cell structures (1R, 1S1R, 1T1R, 1C) and explain sneak path fix
- [ ] Compute sneak path current in a 2×2 crossbar (CF06 CMAN style)
- [ ] Map a 3-input NN layer onto a 3×3 crossbar
- [ ] Explain why sparse mapping crosses over at 70% sparsity
- [ ] Write the self-attention formula from memory
- [ ] Explain why transformers are NOT recurrent
- [ ] State BF16's bit layout and why it's safer than FP16
- [ ] Describe AER packet format
- [ ] Draw a 2D mesh NoC and explain XY routing
- [ ] Explain "neurons are dumb" + how source knows destinations
- [ ] Explain why Cerebras WSE-3 and IBM NorthPole are NOT considered neuromorphic chips
- [ ] Justify why we don't do neuromorphic processing "all in software" (energy + parallelism arguments)
- [ ] Explain 3 ways a crossbar handles negative weights (differential pair, offset, sign+mag)
- [ ] Critique the "LLM on Loihi 2" 3× / 2× result — why is it disappointing?
