# M3 Synthesis Plan — ternary_mac_array
**Synthesis path:** Option A (project compute core)

CF07 synthesis: 264 cells, estimated +8.5 ns slack at 100 MHz, ~609 μm² area. Not timing- or area-constrained, so M3 work focuses on correctness and completeness.

**Changes planned for M3:**

1. **Fix the valid_out bug.** Replace `valid_out <= (acc != '0)` with a cycle counter that pulses at exactly LENGTH cycles, so a legitimate zero dot product is not silently dropped.

2. **Parameterize LENGTH externally.** The same core must serve FC layers of length 256, 128, and 40 without per-layer re-synthesis. The parameter already exists; controller logic to drive it needs wiring.

3. **Run a real sky130 OpenLane 2 flow** (Docker) before May 24 to replace estimated timing/area with actual post-synthesis STA numbers.

The three-instance system (one core per FC layer) should stay under ~800 cells — well within the project's MCU-scale scope.
