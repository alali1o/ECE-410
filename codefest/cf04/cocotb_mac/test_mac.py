"""
test_mac.py — cocotb testbench for mac_correct.v
ECE 410/510 Spring 2026 — Codefest 4 COPT Part A

Tests:
  test_mac_basic    — assignment sequence: a=3,b=4 x3; rst; a=-5,b=2 x2
  test_mac_overflow — extended accumulation proves linear growth (no saturation);
                      overflow behavior documented analytically from RTL inspection
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


SETTLE = Timer(1, unit='ps')   # 1 ps after posedge: lets always_ff output settle


def read_s32(signal):
    """Read a cocotb signal as a signed 32-bit integer."""
    raw = int(signal.value)
    if raw >= (1 << 31):
        raw -= (1 << 32)
    return raw


async def reset_dut(dut):
    """One-cycle synchronous reset."""
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0
    await RisingEdge(dut.clk)
    await SETTLE
    dut.rst.value = 0


# ── Basic test (assignment sequence) ─────────────────────────────────────────
@cocotb.test()
async def test_mac_basic(dut):
    """a=3,b=4 for 3 cycles -> rst -> a=-5,b=2 for 2 cycles"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.a.value = 3
    dut.b.value = 4
    for expected in [12, 24, 36]:
        await RisingEdge(dut.clk)
        await SETTLE
        got = read_s32(dut.out)
        assert got == expected, f"FAIL: got {got}, expected {expected}"
        dut._log.info(f"out = {got}  (expected {expected})  PASS")

    # Mid-run reset
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0
    await RisingEdge(dut.clk)
    await SETTLE
    got = read_s32(dut.out)
    assert got == 0, f"FAIL after rst: got {got}, expected 0"
    dut._log.info(f"out = {got}  (expected 0 after rst)  PASS")
    dut.rst.value = 0

    dut.a.value = -5
    dut.b.value =  2
    for expected in [-10, -20]:
        await RisingEdge(dut.clk)
        await SETTLE
        got = read_s32(dut.out)
        assert got == expected, f"FAIL: got {got}, expected {expected}"
        dut._log.info(f"out = {got}  (expected {expected})  PASS")


# ── Overflow test ─────────────────────────────────────────────────────────────
@cocotb.test()
async def test_mac_overflow(dut):
    """
    Extended accumulation (50 cycles, a=3 b=4) proves linear growth — no saturation.

    Overflow behavior (analytically, not simulated):
      mac_correct.v has no clamping logic. The 32-bit signed accumulator simply
      wraps on overflow (two's complement). At 178,956,971 cycles of a=3,b=4
      (product=12/cycle), total = 2,147,483,652 > 2^31-1 → out wraps negative.
      Behavior: WRAP, not saturate.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    PRODUCT = 12     # 3 * 4
    CYCLES  = 50

    dut.a.value = 3
    dut.b.value = 4
    dut._log.info(f"Accumulating {CYCLES} cycles (a=3, b=4, product={PRODUCT}/cycle)")

    prev = 0
    for i in range(1, CYCLES + 1):
        await RisingEdge(dut.clk)
        await SETTLE
        got      = read_s32(dut.out)
        expected = PRODUCT * i
        delta    = got - prev
        assert got == expected, \
            f"Cycle {i}: FAIL got={got}, expected={expected}"
        assert delta == PRODUCT, \
            f"Cycle {i}: growth={delta}, expected {PRODUCT} — possible saturation?"
        prev = got
        if i <= 5 or i == CYCLES:
            dut._log.info(f"Cycle {i:2d}: out = {got:,}  (+{delta})  PASS")

    dut._log.info(f"All {CYCLES} cycles: linear growth confirmed, delta = {PRODUCT}/cycle")
    dut._log.info("Behavior: WRAP (two's complement) — no saturation logic in mac_correct.v")
    dut._log.info(f"Full overflow at ~178,956,971 cycles; accumulator wraps to large negative")
