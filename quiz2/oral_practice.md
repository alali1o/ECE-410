# Quiz 2 — Oral Practice Questions and Answers
**Bao Nguyen | ECE 410/510 Spring 2026**
*Answers in oral exam style. 45-60 seconds spoken per answer. Based on QUIZ-marked slides weeks 5–8.*

---

## Q1. What is a crossbar and how does it perform matrix-vector multiplication?

A crossbar is a 2D grid of resistive memory cells where rows carry input voltages and columns collect output currents, and the weights are programmed as conductances at each intersection.

The way it does matrix-vector multiplication comes straight from Ohm's law and Kirchhoff's current law. Each cell multiplies the input voltage by its conductance (I = G × V), and all those currents sum down the column automatically via Kirchhoff, so each column output is the full dot product for that output neuron. The whole MVM happens in one read cycle.

In my K-Means project I found the distance kernel was memory-bound at 1.68 FLOP/byte, way below the ridge point, because I kept having to go back to DRAM for every access. A crossbar flips that by doing the computation inside the memory itself, which is exactly what I was trying to achieve by offloading to a near-memory PIM chiplet.

The point is the crossbar turns a matrix-vector multiply into a physics problem. Ohm and Kirchhoff do the math for you in one shot.

---

## Q2. What is a sneak path and why is it a problem in a crossbar?

A sneak path is an unintended current route through unselected cells in a crossbar array.

When you try to read one cell, floating row and column nodes let current loop through nearby cells instead of the one you selected. The sense amplifier, which is the readout circuit at the bottom of each column that converts current into an output value, picks up the intended current plus all the sneak current mixed together and you cannot tell them apart. In a large array every floating node adds more sneak loops, so all your dot products come out wrong.

The fix is either diodes to enforce one-directional current flow, or 1T1R cells where a transistor gate controls which row is active so only the selected row's current flows.

The point is sneak paths are a physical consequence of the crossbar topology and you have to handle them at the cell level before the array gives you correct MVM results.

---

## Q3. Why are transformers not recurrent and why does that matter?

A transformer replaces recurrence, which is the idea of processing one token at a time and passing a hidden state forward to the next step, with self-attention. Instead it processes all tokens at the same time in parallel and lets every token attend directly to every other token in one operation.

The reason that matters is parallelism. RNNs cannot parallelize over the time dimension because each step depends on the previous one. A transformer has no such dependency, so the entire sequence can be computed at once on a GPU, which is what makes training on large datasets feasible.

The tradeoff is that without positional recurrence you lose the natural sense of order, so positional encodings using sine and cosine functions are added to give the model a sense of where each token sits in the sequence.

The point is replacing recurrence with self-attention is what unlocked the scale of modern LLMs because it made the architecture fully parallelizable on GPU hardware.

---

## Q4. Explain the three systolic array dataflows and which one wins on energy.

The three dataflows are weight stationary, output stationary, and row stationary. Each one decides which data type stays fixed in the PEs (Processing Elements, the individual compute units in the array) and which streams through.

Weight stationary holds weights in the PEs while activations and partial sums stream through, maximizing weight reuse. That is what Google's TPU uses. Output stationary keeps each PE accumulating one output element. Row stationary keeps one filter row per PE and reuses all three data types as much as possible, which is why it wins on energy. Eyeriss showed roughly ten times fewer DRAM accesses using row stationary compared to the others.

The tradeoff is that each dataflow minimizes movement of one data type at the cost of moving the others more. There is no universally optimal choice, which connects to the No Free Lunch theorem (the idea that no single algorithm or architecture wins on every problem, you always have to match the solution to the specific workload).

The point is the dataflow strategy is a co-design decision and the right choice depends on which data type your hardware can least afford to move.

---

## Q5. What is AER and how does it work in a neuromorphic chip?

AER stands for Address Event Representation. It is a spike event message-passing protocol used for communication between cores in a neuromorphic chip.

When a neuron fires it does not send the actual spike waveform. It sends its own unique ID as a packet containing the destination core, the neuron ID, and a timestamp. Only active neurons generate messages so the communication is sparse and asynchronous. A routing table at the source core looks up which destinations that neuron connects to and the network interface sends out one packet per destination. The NoC, or Network-on-Chip, routes those packets using XY routing across a 2D mesh topology.

This is why neuromorphic chips are so energy efficient on sparse workloads. Routers only fire when a spike actually happens so average power stays very low even though peak bandwidth is high.

The point is AER lets a neuromorphic chip exploit sparsity at the communication level, which is where a huge fraction of the energy savings come from.

---

## Q6. Why is BF16 safer than FP16 for training?

BF16 and FP16 are both 16-bit formats but they distribute the bits differently. FP16 uses 5 bits for the exponent and 10 for the mantissa. BF16 keeps the full 8-bit exponent from FP32 and uses only 7 bits for the mantissa.

The exponent controls dynamic range, meaning how large or small a number you can represent. During training, gradients can spike to very large or very small values. FP16's narrow exponent causes overflow and underflow, which requires loss scaling to work around. BF16 has the same dynamic range as FP32 so those instabilities disappear. You give up some mantissa precision but for most training workloads that is an acceptable tradeoff.

Converting between BF16 and FP32 is also trivial. You just drop or add the last 16 mantissa bits with no reformatting needed.

The point is BF16 is a drop-in replacement for FP32 in training because it preserves the dynamic range that matters, and that is exactly why Google developed it for TPU workloads.

---

## Q7. Why can you not do neuromorphic processing all in software?

The core problem is energy and parallelism. A single DRAM access costs around 2 nanojoules while a single INT4 multiply costs around 0.1 picojoules. That is a gap of roughly twenty thousand times just for one memory access versus one multiply. Scale that to a brain-scale SNN (Spiking Neural Network, where neurons communicate via discrete spike events rather than continuous values) running in real time and the energy cost in software becomes not viable.

The second problem is parallelism. The brain is massively parallel and event-driven. A CPU is sequential and clock-driven. In software you are still paying full instruction fetch and decode overhead for every neuron every timestep, even though only about one percent of neurons are firing at any given moment. Hardware can gate everything off until a spike arrives. Software cannot.

The point is specialized hardware is the only way to close the gap between what the brain does and what silicon can do efficiently, which is exactly why neuromorphic chips exist.

---

## Q8. How does a crossbar handle negative weights?

A crossbar does MVM by having each cell multiply the input voltage by its conductance (I = G × V), and conductance is always positive. There is no such thing as a negative resistor. But neural network weights need to be signed to represent both positive and negative connections (excitatory and inhibitory synapses).

The most common fix is the differential pair approach. Each weight is stored across two memristor columns, one for the positive part and one for the negative part, and the final weight is the difference between them. The output current is the difference of the two column currents, which gives you a clean signed result using the same crossbar building block. The cost is roughly twice the area.

Two other options are offset subtraction, where you shift all weights positive and subtract a fixed offset at readout, and sign-magnitude encoding where separate arrays handle sign and magnitude.

The point is the crossbar's physics only gives you positive values so negative weights have to be handled through circuit design at the cell or column level.

---

## Q9. What is CSR format and how do you read a row from it?

CSR stands for Compressed Sparse Row. It is a format for storing sparse matrices using three arrays instead of the full N squared dense grid.

The three arrays are values, which holds all the non-zero elements in row-major order; col_idx, which holds the column index of each non-zero; and row_ptr, which is a bookmark array of length N plus one where row_ptr[i] tells you where row i starts in the values array. To read row i you slice from row_ptr[i] to row_ptr[i plus 1] in both arrays. That gives you exactly the non-zeros for that row and their column positions.

In my K-Means project I worked with dense distance matrices between pixels and centroids, so CSR would not help there. But for something like a sparse weight matrix in a pruned neural network, CSR is exactly how you would store it efficiently before mapping it onto a crossbar.

The point is CSR trades a small overhead of N plus one integers to eliminate all the zero storage, which pays off fast as the matrix gets large and sparse.

---

## Q10. Why is the Loihi 2 LLM result not impressive?

A recent paper ran a MatMul-free language model on Intel Loihi 2 and reported three times the throughput and two times better energy efficiency compared to an edge GPU.

That sounds good but it is actually weak. For a purpose-built neuromorphic chip versus a general GPU, you would expect something closer to ten to a thousand times improvement based on what the emerging technology table shows for neuromorphic chips. Three times throughput barely shows the silicon advantage. Two times energy is especially underwhelming because neuromorphic chips are supposed to be orders of magnitude more efficient on spike-based workloads.

The other problem is the model was completely rewritten to fit Loihi 2's constraints. No standard transformer attention, a special MatMul-free architecture that does not represent how LLMs actually run in practice.

The point is Loihi 2 is built for sparse event-driven workloads like robot control or keyword spotting, not LLM inference, and running a transformer-style model on it exposes that mismatch rather than showing what the chip is actually good at.
