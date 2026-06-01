# CF09 CLLM — Roofline Analysis
ECE 410/510 Spring 2026

## Gap Analysis (Projected Path)

The accelerator was plotted using the projected path because a full production-size simulation (N_IN=1960, N_OUT=512) was not run. The dominant uncertainty in the projection is the SPI transfer time: the calculation assumes burst-mode SPI (CS held low for all bytes), but the current spi_slave_ext RTL performs one 16-bit register transaction per byte, which is 2× slower and pushes end-to-end throughput below 130 inferences/s. The actual transaction overhead per byte depends on CS assertion and recovery cycles, which were not timed in simulation.

The second largest uncertainty is the system clock frequency. The M3 critical path analysis estimated 50 MHz as the safe operating point based on the SPI synchronizer chain (~16 ns estimated delay). If the actual synthesized critical path is shorter — possible since the estimate was conservative and generic cells were used, not sky130 standard cells — the clock could reach 80–100 MHz, improving compute throughput by up to 2×.

To convert the projection to a measurement, three things are needed: (1) run a full cocotb simulation of the production-size compute_core at N_IN=1960, N_OUT=512 to confirm the 1960-cycle inference latency; (2) profile the SPI byte-transfer time in M3 simulation to get the actual per-byte latency with CS overhead; (3) complete the OpenLane 2 flow on a Linux host to get a verified clock frequency from STA timing reports. The compute-only speedup (11.1×) is the most reliable projected number because it depends only on the well-verified cycle count and clock frequency, neither of which has significant uncertainty.
