"""
test_ternary_mac.py — cocotb testbench stub for ternary_mac_array.sv
ECE 410/510 Spring 2026 — Codefest 4 COPT Part B

Drives reset, loads a short dot-product (length=4), and checks the result.
Full assertion suite not required — goal is a working simulation harness.

Dot-product example:
  activations = [3, -2, 5, -1]   (INT8)
  weights     = [+1, -1, 0, +1]  (ternary)
  expected    = 3*1 + (-2)*(-1) + 5*0 + (-1)*1 = 3 + 2 + 0 - 1 = 4
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

# Ternary encoding: +1 -> 2'b01, 0 -> 2'b00, -1 -> 2'b11
def ternary_encode(w):
    if   w ==  1: return 0b01
    elif w == -1: return 0b11
    else:         return 0b00

@cocotb.test()
async def test_dot_product(dut):
    """Basic 4-element ternary dot product."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Reset
    dut.rst.value      = 1
    dut.valid_in.value = 0
    dut.act_in.value   = 0
    dut.wt_in.value    = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    activations = [3, -2, 5, -1]
    weights     = [+1, -1, 0, +1]
    expected    = sum(a * w for a, w in zip(activations, weights))  # = 4

    # Stream inputs
    dut.valid_in.value = 1
    for act, wt in zip(activations, weights):
        dut.act_in.value = act
        dut.wt_in.value  = ternary_encode(wt)
        await RisingEdge(dut.clk)

    dut.valid_in.value = 0
    await RisingEdge(dut.clk)

    result = dut.acc_out.value.signed_integer
    dut._log.info(f"Dot product result: {result}  (expected {expected})")
    assert result == expected, f"FAIL: got {result}, expected {expected}"
