# CF06 CMAN — Resistive Crossbar Sneak Paths
ECE 410/510 Spring 2026

**Given:**
- 2×2 crossbar, all resistors R = 1 kΩ
- Weight matrix W = [[+1, 0], [0, +1]] (only R[0][0] and R[1][1] are connected, R = 1 kΩ each)
- Input vector: row 0 driven to V = 1V, row 1 driven to V = 0V (ground)
- Output read on col 0

---

## (a) Ideal Read (No Sneak Path)

With row 1 = 0V, only the resistor at R[0][0] contributes to col 0.

```
I_col0_ideal = V_row0 / R[0][0] = 1V / 1kΩ = 1 mA
```

This represents the correct MVM result for that column.

---

## (b) Sneak Path Analysis

In a real crossbar, all resistors are always physically present (even "0-weight" cells have a finite, large resistance or a low-resistance path through adjacent cells when no selector is used). This creates parasitic current paths called **sneak paths**.

**Setup for sneak path scenario:**

Consider a 2×2 array where all four resistors are 1 kΩ (worst case — no selector diodes). Drive:
- Row 0: V_row0 = 1V
- Row 1: floating (or weakly pulled to 0V through the array)
- Col 1: connected to a sense amplifier (virtual ground ≈ 0V)
- Col 0: connected to a sense amplifier (virtual ground ≈ 0V)

Because col 1 is also at 0V, a sneak path exists:

```
Row 0 (1V) → R[0][1] (1kΩ) → Col 1 (0V)
Row 0 (1V) → R[0][0] (1kΩ) → Col 0 (0V)
```

But also if Row 1 is floating, KCL at the Row 1 node:

```
Sneak path:  Row 0 → R[0][1] → Col 1 node → R[1][1] → Row 1 → R[1][0] → Col 0
```

**KCL at the floating row 1 node (V_r1):**

Current in from col 1 side = current out to col 0 side:
```
(1 - V_r1) / 1k  =  V_r1 / 1k     [through R[0][1] and R[1][0]]
1 - V_r1 = V_r1
V_r1 = 0.5V
```

**Sneak current into col 0:**
```
I_sneak = V_r1 / R[1][0] = 0.5V / 1kΩ = 0.5 mA
```

**Total current at col 0:**
```
I_col0_actual = I_direct + I_sneak
              = (1V / 1kΩ) + 0.5 mA
              = 1.0 mA + 0.5 mA
              = 1.5 mA
```

**Error = 0.5 mA → 50% over-read on col 0.**

---

## (c) How Sneak Paths Corrupt MVM

The sense amplifier at col 0 reads 1.5 mA instead of the correct 1 mA. This excess current comes from an unintended path through unselected cells — it does not correspond to any weight × input product in the intended computation. In a large N×N crossbar this effect compounds: every unselected row that is not hard-driven to 0V contributes sneak currents through neighboring resistors. The result is that the analog dot product read out at each column column is corrupted, making the MVM output incorrect without mitigation (selector devices, active row drivers, or digital correction).

---

## Summary

| Condition | I_col0 | Error |
|-----------|--------|-------|
| Ideal (no sneak) | 1.0 mA | 0% |
| With sneak path | 1.5 mA | +50% |

Sneak paths are a fundamental challenge for passive resistive crossbar arrays used in analog in-memory computing.
