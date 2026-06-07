# Quiz 2 — Keyword Definitions
*One or two sentence definitions for every keyword from the quiz-marked slides.*

---

## Compute and Performance

**MAC (Multiply-Accumulate)** — One multiply plus one add, equals 2 FLOPs. The fundamental operation in every neural network layer.

**FLOP (Floating Point Operation)** — A single arithmetic operation on floating point numbers. Used to measure how much computation a kernel performs.

**GEMM (General Matrix-Matrix Multiply)** — The dominant operation in neural networks. Fully connected layers, convolutions, and attention all reduce to GEMM.

**Arithmetic Intensity (AI)** — FLOPs divided by bytes transferred to and from memory, measured in FLOP/byte. Tells you whether a kernel is bottlenecked by compute or memory bandwidth.

**Roofline Model** — A visual framework that plots achievable performance against arithmetic intensity. Shows two ceilings: peak compute and peak memory bandwidth.

**Ridge Point** — Where the memory bandwidth ceiling and compute ceiling meet on the roofline. Calculated as peak compute divided by peak bandwidth. Kernels left of the ridge are memory-bound, kernels right are compute-bound.

**Memory-bound** — A kernel limited by memory bandwidth, not compute throughput. Sits left of the ridge point. Fix by increasing data reuse or moving compute closer to memory.

**Compute-bound** — A kernel limited by peak compute throughput. Sits right of the ridge point. Fix by improving occupancy or Tensor core utilization.

**DRAM (Dynamic Random Access Memory)** — The large off-chip memory on a GPU or system. High capacity but slow and expensive to access energy-wise.

**SRAM (Static Random Access Memory)** — Fast on-chip memory. Used for caches, shared memory, and weight buffers. Much faster than DRAM but smaller and more expensive per bit.

**HBM (High Bandwidth Memory)** — High-bandwidth stacked DRAM used on modern GPUs and TPUs. Faster than standard DRAM but still off-chip.

**Bandwidth** — The rate at which data can be transferred between memory and compute, measured in GB/s. The bottleneck for memory-bound kernels.

**Throughput** — How many operations a system completes per unit time, measured in GFLOP/s or TOPS. The bottleneck for compute-bound kernels.

**Latency** — The time it takes to complete one operation from start to finish. CPUs minimize latency; GPUs hide it with warp switching.

**Occupancy** — Active warps divided by maximum warps per SM. Low occupancy means the scheduler has few warps to switch to and compute units sit idle.

**Tiled GEMM** — An optimization that uses shared memory to stage tiles of input matrices, so each value is loaded from HBM exactly once and reused many times. Increases arithmetic intensity from O(1) to O(N).

---

## Algorithm Acceleration

**Technology Scaling** — Improving performance by advancing the manufacturing process, such as shrinking transistor size. Historically driven by Moore's Law but now flattening out.

**Moore's Law** — The observation that transistor count on a chip doubles roughly every two years. No longer holding at the same pace, which is why algorithm and architecture improvements matter more.

**Cache Optimization** — Restructuring data access patterns to keep frequently used data in fast on-chip cache rather than going back to slow DRAM.

**Parallelism** — Running multiple operations at the same time. GPUs exploit massive parallelism across thousands of threads.

**HW/SW Co-design** — Designing hardware and software together concurrently so each can be optimized around the other. The opposite of designing one first and adapting the other.

**FPGA (Field-Programmable Gate Array)** — Reconfigurable hardware that can be programmed after manufacturing. High speedup for targeted workloads but complex to program. High difficulty, low applicability.

**ASIC (Application-Specific Integrated Circuit)** — Custom chip designed for one specific function. Highest possible speedup and efficiency but completely inflexible outside its target domain.

**No Free Lunch (NFL) Theorem** — Formalized by Wolpert and Macready, 1997. Averaged across all possible problems, all optimization algorithms perform equally well. Specialized hardware wins only in its specific domain.

**Wolpert and Macready** — Authors of the No Free Lunch theorem, 1997.

**PPAC (Performance, Power, Area, Cost)** — The four axes of every hardware design tradeoff. Improving one typically hurts another.

---

## Systolic Arrays

**Systolic Array** — A 2D grid of identical Processing Elements that rhythmically compute and pass data to their neighbors. Named for the analogy to a heartbeat.

**PE (Processing Element)** — The individual compute unit in a systolic array. Performs one MAC per clock cycle and forwards data to the next PE.

**Weight Stationary** — A systolic array dataflow where weights are held fixed in PEs while activations and partial sums stream through. Maximizes weight reuse. Used by Google TPU.

**Output Stationary** — A systolic array dataflow where each PE accumulates one output element while weights and activations both stream through. Minimizes accumulator writes.

**Row Stationary** — A systolic array dataflow where one filter row stays in each PE and all data types are reused as much as possible. Wins on energy with roughly ten times fewer DRAM accesses than other dataflows. Used by MIT Eyeriss.

**Activations** — The input values to a neural network layer, or the output values after an activation function. Stream through the systolic array in weight-stationary dataflow.

**Partial Sums** — Intermediate accumulated results during a matrix multiply. Built up across multiple MAC operations before producing the final output.

**Weight Memory** — On-chip SRAM that preloads weights before they are pushed into the systolic array.

**Activation Memory** — On-chip SRAM that holds input activations before they stream into the systolic array.

**Accumulators** — Registers that collect and sum partial products during matrix multiply. Need to be wide enough to avoid overflow.

**Pipeline Fill** — The initial cycles needed to load data into a systolic array before results start coming out. For an N by N array this takes 2N minus 1 cycles.

**Dataflow** — The strategy that determines which data type stays stationary and which streams through a systolic array. Determines energy efficiency.

---

## Precision Formats

**FP32 (32-bit Floating Point)** — Full precision with 1 sign bit, 8 exponent bits, and 23 mantissa bits. Standard for training and the baseline for comparison.

**FP16 (16-bit Floating Point)** — Half precision with 5 exponent bits and 10 mantissa bits. Narrower dynamic range than FP32 which causes overflow and underflow during training.

**BF16 (Brain Float 16)** — 16-bit format developed by Google Brain with 8 exponent bits and 7 mantissa bits. Same dynamic range as FP32, safer for training than FP16.

**INT8 (8-bit Integer)** — 8-bit integer format for inference only. Higher throughput than FP16 at the same die area.

**FP4 (4-bit Floating Point)** — Extreme low precision format with only 16 distinct representable values. Used with block scaling on Blackwell for LLM inference.

**Exponent Bits** — The bits in a floating point number that control dynamic range, meaning how large or small a value can be represented.

**Mantissa Bits** — The bits in a floating point number that control precision, meaning how many distinct values can be represented within a given range.

**Dynamic Range** — The span between the largest and smallest representable values. Controlled by exponent bits. BF16 and FP32 have the same dynamic range.

**Overflow** — When a value exceeds the maximum representable number. Common in FP16 training due to its narrow exponent range.

**Underflow** — When a value is smaller than the minimum representable number and rounds to zero. Common in FP16 training.

**Loss Scaling** — A workaround for FP16 training where gradients are multiplied by a large scale factor before the backward pass and divided back afterward to avoid underflow.

**Quantization** — Reducing the number of bits used to represent a value. Trades precision for throughput and memory efficiency. Binary weights are the extreme end of quantization.

**Block Scaling** — A technique used in FP4 (NVFP4) where groups of values share a common scale factor to recover accuracy lost from extreme precision reduction.

---

## Transformers

**Transformer** — A neural network architecture introduced in "Attention Is All You Need" (Vaswani et al., 2017). Replaces recurrence with self-attention and processes all tokens at the same time.

**Self-Attention** — The mechanism that lets every token attend directly to every other token in one operation. Computes dot products between Q, K, and V projections.

**Recurrence** — The sequential processing pattern in RNNs where each step depends on the previous hidden state. Cannot be parallelized over time. Replaced by self-attention in transformers.

**Hidden State** — The compressed memory vector passed from one time step to the next in an RNN or LSTM. Represents all prior context but has fixed size.

**RNN (Recurrent Neural Network)** — A sequential neural network that processes tokens one at a time, passing a hidden state forward. Cannot be parallelized during training.

**LSTM (Long Short-Term Memory)** — A gated variant of RNN designed to handle long-range dependencies better. Still sequential and cannot be parallelized.

**Q (Query)** — One of three projections in self-attention. Represents what a token is looking for.

**K (Key)** — One of three projections in self-attention. Represents what a token offers to other tokens.

**V (Value)** — One of three projections in self-attention. Represents what a token actually contributes if selected.

**Softmax** — A function that converts a vector of raw scores into a probability distribution that sums to one. Used in self-attention to weight the value vectors.

**Scaling Factor (√d_k)** — The square root of the key dimension used to divide the QK dot products before softmax. Prevents the dot products from getting too large and causing softmax to saturate.

**Multi-Head Attention** — Running multiple self-attention operations in parallel, each with different learned projections, then concatenating the results. Lets the model attend to different aspects of the input at once.

**Positional Encoding** — Sine and cosine signals added to token embeddings to give the transformer a sense of token order. Replaces the positional information that recurrence provided naturally.

**RLHF (Reinforcement Learning from Human Feedback)** — A three-stage process to align a pretrained LLM with human preferences: supervised fine-tuning, reward model training, and PPO reinforcement learning.

**SFT (Supervised Fine-Tuning)** — The first stage of RLHF. Fine-tune the pretrained model on human-written demonstrations to initialize good behavior.

**Reward Model** — A model trained on human preference rankings that outputs a scalar score for any prompt-response pair. Used to guide PPO training.

**PPO (Proximal Policy Optimization)** — The reinforcement learning algorithm used in RLHF to update the LLM policy to maximize reward while staying close to the SFT baseline via a KL penalty.

**KL Divergence** — A measure of how different two probability distributions are. Used in RLHF as a penalty to prevent the PPO-trained model from drifting too far from the SFT baseline.

---

## In-Memory Computing and Crossbar

**Crossbar** — A 2D grid of resistive memory cells where rows carry input voltages, columns collect output currents, and weights are programmed as conductances at each intersection. Performs MVM in one read cycle.

**MVM (Matrix-Vector Multiplication)** — Multiplying a matrix by a vector. The crossbar performs this natively via Ohm's law and Kirchhoff's current law.

**Conductance (G)** — How easily current flows through a resistive memory cell, measured in Siemens. The inverse of resistance. Each cell is programmed to a conductance value that represents a synaptic weight.

**Ohm's Law** — Current equals conductance times voltage, I equals G times V. The law that governs how each crossbar cell produces its contribution to the output current.

**Kirchhoff's Current Law** — The total current at a node equals the sum of all currents flowing into it. Applied down each column of a crossbar to sum all the individual cell currents into a dot product.

**Sense Amplifier** — The readout circuit at the bottom of each crossbar column that converts the total column current into a usable output value. Corrupted by sneak path current.

**Sneak Path** — An unintended current route through nearby unselected cells in a crossbar. Adds noise to the sense amplifier reading and corrupts the dot product result.

**Diode** — A device that allows current to flow in only one direction. Used in 1S1R crossbar cells to block sneak paths by preventing reverse current flow.

**Selector** — A nonlinear device (such as a diode) paired with a resistive memory cell in a 1S1R structure to suppress sneak paths while maintaining array density.

**1R** — The simplest crossbar cell with one resistor. Suffers from sneak paths.

**1S1R** — Crossbar cell with one selector and one resistor. The selector blocks sneak paths without sacrificing array density.

**1T1R** — Crossbar cell with one transistor and one resistor. Best selectivity since the transistor gate controls exactly which row is active. Lower density than 1S1R.

**1C** — Crossbar cell with one capacitor. No static leakage during MVM because capacitors do not conduct DC current.

**RRAM (Resistive RAM)** — A non-volatile resistive memory that stores data as different resistance states. Strong analog multilevel capability, good fit for crossbar synaptic weights.

**PCM (Phase-Change Memory)** — Non-volatile memory that stores data by switching between amorphous and crystalline states with different resistances. Strong analog behavior, slower write speed than RRAM.

**STT-MRAM (Spin-Transfer Torque Magnetic RAM)** — Non-volatile memory that stores data using magnetic spin states. Fast write speed and high endurance but limited analog multilevel capability.

**Memristor** — A resistive memory device whose resistance changes based on the history of current flow. The most mature emerging memory technology for in-memory computing.

**Differential Pair** — The most common approach for handling negative weights in a crossbar. Each weight is stored across two columns, G-plus and G-minus, and the signed result is G-plus minus G-minus.

**Offset Subtraction** — An approach for negative weights where all weights are shifted positive and a fixed offset is subtracted at readout. Simpler than differential pair but sacrifices dynamic range.

**Sign-Magnitude Encoding** — An approach for negative weights where sign and magnitude are stored in separate arrays and combined in the peripheral circuitry.

---

## Sparsity and CSR

**Sparse Matrix** — A matrix where most values are zero. Real neural network weight matrices after pruning are often 90 to 99 percent zeros.

**Sparsity** — The fraction of values in a matrix that are zero. Higher sparsity means more zeros and more potential savings from sparse formats.

**Permute and Pack** — A software technique for sparse crossbar mapping that reorders rows and columns to cluster non-zeros into dense blocks, reducing the number of tiles needed.

**Tile Partitioning** — Splitting a matrix into fixed-size tiles and skipping tiles that are entirely zero to avoid mapping empty crossbar cells.

**Bitline Gating** — A hardware technique that power-gates zero-input rows and empty columns to save energy during sparse MVM.

**CSR (Compressed Sparse Row)** — A sparse matrix format using three arrays: values (the non-zeros), col_idx (column index of each non-zero), and row_ptr (bookmark into the arrays for each row).

**COO (Coordinate Format)** — A sparse matrix format that stores every non-zero as a (row, col, value) tuple. Simpler than CSR but uses more memory since it stores the row index explicitly for every non-zero.

**values[]** — The array in CSR that holds all non-zero elements in row-major order.

**col_idx[]** — The array in CSR that holds the column index of each non-zero element.

**row_ptr[]** — The array in CSR of length N plus one where row_ptr[i] is the index into values where row i starts. The last entry equals the total number of non-zeros.

**Non-zeros (nnz)** — The count of non-zero elements in a sparse matrix. Determines the memory cost of CSR storage.

**70% Sparsity Crossover** — The point below which a dense crossbar usually outperforms sparse mapping because the decoder and scheduler overhead costs more than the savings from skipping zeros.

---

## Neuromorphic Chips and Communication

**AER (Address Event Representation)** — A spike event message-passing protocol for neuromorphic Network-on-Chip systems. A firing neuron sends only its unique ID as a packet rather than the full spike waveform.

**Spike** — A binary event fired by a neuron when its membrane potential crosses a threshold. The fundamental signal in spiking neural networks and neuromorphic chips.

**Neuron ID** — The unique identifier a neuron sends when it fires in an AER system. The only information the neuron itself emits.

**TIMESTAMP** — The time component of an AER packet indicating when the spike occurred. Timing is implicit in when the message is sent.

**DEST_CORE** — The destination core component of an AER packet. Tells the NoC where to route the spike.

**NoC (Network-on-Chip)** — A packet-switched interconnect that replaces buses in neuromorphic and other manycore chips. Scales in bandwidth as router count grows.

**Router** — A component of the NoC that handles arbitration, buffering, and switching of packets between links.

**Link** — The physical wires connecting routers in a NoC.

**NI (Network Interface)** — The component that packs data into packets for the NoC and unpacks incoming packets back into data.

**XY Routing** — A deterministic routing algorithm in a 2D mesh NoC where packets first travel horizontally to the correct column, then vertically to the correct row.

**2D Mesh** — The default NoC topology where each node connects to its four neighbors (up, down, left, right). Scalable and simple to implement.

**Torus** — A 2D mesh where the edges wrap around and connect to the opposite side, reducing maximum hop count.

**Fan-out** — The number of destination cores a single spike must be delivered to. One spike can generate multiple AER packets, one per destination.

**Routing Table** — An SRAM table at the source core that maps a neuron ID to its list of destination cores. Neurons are "dumb" and only know their own ID; the table holds the connectivity.

**Multicast** — Sending one packet to multiple destinations simultaneously. Used in SpiNNaker to efficiently deliver spikes to many target cores.

**SNN (Spiking Neural Network)** — A neural network where neurons communicate via discrete spike events rather than continuous-valued activations. More biologically realistic and energy efficient on sparse workloads.

**LIF (Leaky Integrate-and-Fire)** — The most common neuromorphic neuron model. Accumulates input current, leaks charge over time, and fires a spike when voltage crosses a threshold.

**STDP (Spike-Timing-Dependent Plasticity)** — A local on-chip learning rule where the strength of a synapse changes based on the relative timing of pre- and post-synaptic spikes. Enables on-chip learning without backpropagation.

**Event-driven** — Processing that only activates when a spike event occurs, rather than running every clock cycle. Enables low average power since most neurons are silent most of the time.

**Asynchronous** — Operating without a global clock. AER communication is asynchronous since spikes are sent only when they occur, not on a fixed schedule.

---

## Neuromorphic Hardware

**Loihi** — Intel's first neuromorphic research chip. Implements LIF neurons with on-chip STDP learning.

**Loihi 2** — Intel's second-generation neuromorphic chip. Supports up to a million neurons, programmable neuron models via microcode, graded spikes, and the Lava open-source framework.

**TrueNorth** — IBM's neuromorphic chip with one million neurons and 256 million synapses. Extremely low power (65 mW) but fixed neuron model and no on-chip learning.

**NorthPole** — IBM's dense inference chip optimized for INT2/4/8 operations. Not truly neuromorphic despite being marketed as brain-inspired. No spikes, no STDP, no learning.

**BrainScaleS** — A neuromorphic system from Heidelberg University using analog circuits to emulate neurons. Runs up to ten thousand times faster than biological real time.

**SpiNNaker** — A neuromorphic system from Manchester University using ARM cores and multicast routers for flexible spike routing. Designed for large-scale brain simulation.

**Akida** — A commercial neuromorphic chip from BrainChip using Temporal Event-based Neural Networks (TENNs) for edge inference.

**Cerebras WSE-3** — A wafer-scale dense matrix compute chip for LLM training. Not neuromorphic despite its size. Synchronous, clock-driven, no spikes or STDP.

**MatMul-free LLM** — A language model architecture rewritten to avoid matrix-matrix multiplication so it can run on neuromorphic hardware. Uses binary weights and AND-accumulate instead of multiply-accumulate.

**AAC (AND-Accumulate)** — Replacing MAC with a bitwise AND followed by integer addition. Used when weights are binary (0 or 1), making multiplication trivial.

**Binary Weights** — Weights constrained to plus-one or minus-one. The most aggressive form of quantization, 1-bit. Simplifies the MAC operation to add or subtract.

**TRL (Technology Readiness Level)** — A scale from 1 to 9 measuring how mature a technology is. Memristors are Med-High TRL. Protein nanowires are Very Low TRL.

**EDP (Energy-Delay Product)** — A combined metric of energy efficiency and speed. Lower EDP is better. Used to compare neuromorphic chips on different workloads.

---

## Emerging Technologies

**Memristors** — Resistive memory devices that change resistance based on current history. Most mature emerging technology for in-memory computing. 10 to 100 times speedup, 100 to 1000 times energy gain.

**Neuromorphic Chips** — Chips inspired by the brain's architecture using spike-based event-driven processing. 10 to 1000 times speedup, 1000 to 10,000 times energy gain over conventional processors on applicable workloads.

**Quantum Computing** — Computing using quantum mechanical phenomena like superposition and entanglement. Exponential speedup for specific problems but very low readiness and applicable only to certain algorithm classes.

**Spintronics** — Technology that uses electron spin rather than charge for computation and storage. Non-volatile, minimal heat. 5 to 50 times speedup, 10 to 100 times energy gain.

**Photonic Computing** — Using photons (light) instead of electrons for computation and interconnects. Ultra-high bandwidth potential. 100 to 1000 times speedup and energy gain but Low to Medium TRL.

**Memcapacitors** — Capacitive counterpart to memristors. Complementary technology for in-memory computing. 10 to 100 times speedup, 50 to 500 times energy gain. Low TRL.

**Reservoir Computing** — A computing paradigm using a fixed recurrent network (the reservoir) as a feature extractor for temporal data. 50 to 500 times speedup for time-series tasks. Low to Medium TRL.

**DNA Computing** — Using DNA molecules to perform computation. Massive theoretical parallelism at the molecular level. Ultra-low power but Very Low TRL with major manufacturing challenges.

**Phase-Change Materials** — Materials that switch between amorphous and crystalline states with different electrical properties. Used in PCM memory. 5 to 50 times speedup, 10 to 100 times energy gain. Med-High TRL.

**Protein Nanowires** — Biological nanowires with ultra-low power consumption. 10,000 to 100,000 times energy gain theoretically but Very Low TRL. The highest energy efficiency of any emerging technology on the table.

**IMC (In-Memory Computing)** — Performing computation inside the memory array itself rather than moving data to a separate processor. Eliminates the energy cost of DRAM access which is roughly 20,000 times more expensive than an INT4 multiply.

**PIM (Processing-in-Memory)** — A related term to IMC. Placing compute logic near or inside the memory to dramatically increase effective bandwidth. Used in my K-Means project to clear the ridge point for the memory-bound distance kernel.
