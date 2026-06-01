# Critical Path Analysis — M3
ECE 410/510 Spring 2026

## Critical Path

**Start register:** `spi_slave_ext.sclk_s3` — the third stage of the 3-FF SCLK synchronizer chain.

**End register:** `spi_slave_ext.addr_latch` — the address latch captured after the 8th SPI bit.

**Logic stages between them:**

1. `sclk_rise` detection: `~sclk_s2 & sclk_s3` is computed combinationally from the synchronizer outputs (1 AND + 1 NOT gate).
2. Enable check: `sclk_rise && !csn_s2` — ORed with the CS_N synchronizer output (1 AND gate).
3. `bit_cnt == 7` comparator: 3-bit equality check on the bit counter drives the enable for addr_latch capture (4 XOR + 3-level AND tree ≈ 5 gate levels).
4. `addr_latch` MUX: selects between hold and new value `{rx_shift[AW-2:0], mosi_s2}`, then drives the D input of addr_latch (2 MUX + 1 AND gate).

**Total combinational depth estimate:** 10–12 gate levels.

## Why This Is the Critical Path

The SPI synchronizer chain is slow for two reasons. First, the 3-FF synchronizer for SCLK adds 3 registers of latency; after the third register, the edge detect is purely combinational and feeds directly into time-critical control logic. Second, the bit counter comparison (`bit_cnt == 7`) feeds an enable for the address latch, and this comparison spans 3 bits with an AND reduction tree — all of this combinational logic sits between two flip-flops on the same clock domain, leaving no pipeline register to absorb the delay.

The MAC accumulator path inside `compute_core` (32-bit signed add feeding back into the accumulator DFF) is the second-longest path (~8 levels), but it closes cleanly at 100 MHz on sky130.

## What Would Shorten It

The most effective fix is to **lower the target clock to 50 MHz (20 ns period)** for M4. This gives the SPI path roughly 4 ns of positive slack without any RTL changes and still provides 10× oversampling of a 5 MHz SCLK. Alternatively, the `bit_cnt == 7` comparison could be replaced with a terminal-count flag registered one cycle earlier, eliminating 3–4 gate levels from the path and allowing 80–100 MHz operation.
