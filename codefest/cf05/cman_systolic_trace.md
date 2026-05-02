# CF05 CMAN — Systolic Array Trace (Weight-Stationary)
ECE 410/510 Spring 2026

**Given:**
- A = [[1, 2], [3, 4]]
- B = [[5, 6], [7, 8]]
- Expected C = A x B = [[19, 22], [43, 50]]

---

## (a) PE Diagram — 2x2 Array with Preloaded Weights

In weight-stationary dataflow, the weights (from B) are loaded into each PE before computation starts and never move. Inputs from A stream in from the left and pass rightward one PE per cycle. Partial sums flow downward one PE per cycle and accumulate as they go.

```
              Col 0              Col 1
          +-----------+      +-----------+
Row 0     | PE[0][0]  | ---> | PE[0][1]  |
          |  w = 5    |      |  w = 6    |
          +-----------+      +-----------+
                |                   |
                v                   v
          +-----------+      +-----------+
Row 1     | PE[1][0]  | ---> | PE[1][1]  |
          |  w = 7    |      |  w = 8    |
          +-----------+      +-----------+
                |                   |
                v                   v
            C[*][0]             C[*][1]
```

Weight assignments (pre-loaded from B, stay fixed):
- PE[0][0] = B[0][0] = 5
- PE[0][1] = B[0][1] = 6
- PE[1][0] = B[1][0] = 7
- PE[1][1] = B[1][1] = 8

---

## (b) Cycle-by-Cycle Trace

**How each PE works:**
Each cycle, a PE computes: `psum_out = psum_in (from above) + input_x * weight`
- `input_x` arrives from the left and is passed right to the next PE in the same row next cycle
- `psum_in` arrives from the PE directly above and is passed down after being accumulated

**Input scheduling (skewing):**
Because inputs travel right one step per cycle, and partial sums travel down one step per cycle, the inputs must be staggered (skewed) so each A element arrives at its correct PE at the correct time.

- Row 0 receives: A[0][0]=1 at cycle 1, A[1][0]=3 at cycle 2
- Row 1 receives: 0 (bubble) at cycle 1, A[0][1]=2 at cycle 2, A[1][1]=4 at cycle 3

This ensures, for example, that A[0][0]=1 (entering row 0 at cycle 1) and A[0][1]=2 (entering row 1 at cycle 2) both reach column 0 in time to be accumulated into C[0][0] at cycle 2.

**Cycle table:**

| Cycle | Input row 0 | Input row 1 | PE[0][0] psum | PE[0][1] psum | PE[1][0] psum | PE[1][1] psum | Output C |
|-------|-------------|-------------|---------------|---------------|---------------|---------------|----------|
| 1     | 1           | 0           | 5             | 0             | 0             | 0             | —        |
| 2     | 3           | 2           | 15            | 6             | 19            | 0             | C[0][0] = 19 |
| 3     | 0           | 4           | 0             | 18            | 43            | 22            | C[1][0] = 43, C[0][1] = 22 |
| 4     | 0           | 0           | 0             | 0             | 0             | 50            | C[1][1] = 50 |

**Step-by-step breakdown:**

**Cycle 1:**
- Row 0 input = 1, Row 1 input = 0 (bubble stall)
- PE[0][0]: psum_in=0 (nothing from above) + 1 x 5 = **5**
- PE[0][1]: psum_in=0, input not yet arrived (input 1 is still at PE[0][0]) = **0**
- PE[1][0]: psum_in=0, input=0 = **0**
- PE[1][1]: psum_in=0, input=0 = **0**

**Cycle 2:**
- Row 0 input = 3, Row 1 input = 2
- PE[0][0]: psum_in=0 + 3 x 5 = **15**
- PE[0][1]: psum_in=0 + 1 x 6 = **6**   (input 1 propagated right from PE[0][0] last cycle)
- PE[1][0]: psum_in=5 (from PE[0][0] last cycle) + 2 x 7 = 5 + 14 = **19** --> C[0][0] = 19
- PE[1][1]: psum_in=0 (PE[0][1] was 0 last cycle) + 0 x 8 = **0**   (row 1 bubble propagated right)

**Cycle 3:**
- Row 0 input = 0, Row 1 input = 4
- PE[0][0]: psum_in=0 + 0 x 5 = **0**
- PE[0][1]: psum_in=0 + 3 x 6 = **18**  (input 3 propagated right from PE[0][0] last cycle)
- PE[1][0]: psum_in=15 (from PE[0][0] last cycle) + 4 x 7 = 15 + 28 = **43** --> C[1][0] = 43
- PE[1][1]: psum_in=6 (from PE[0][1] last cycle) + 2 x 8 = 6 + 16 = **22**  --> C[0][1] = 22

**Cycle 4:**
- Row 0 input = 0, Row 1 input = 0 (drain)
- PE[0][0]: **0**
- PE[0][1]: **0**
- PE[1][0]: **0**
- PE[1][1]: psum_in=18 (from PE[0][1] last cycle) + 4 x 8 = 18 + 32 = **50** --> C[1][1] = 50

**Result: C = [[19, 22], [43, 50]] -- matches expected output.**

---

## (c) Counts

**(a) Total MAC operations: 8**

Each output element of C requires 2 multiplications (K=2 shared dimension). There are 4 output elements.
4 x 2 = 8 total MACs. This matches 4 PEs x 2 active cycles each = 8 MACs.

**(b) Input reuse count: each input value is reused 2 times**

Each element of A enters from the left and passes through 2 PEs (column 0 then column 1) in its row.
- A[0][0]=1 is used at PE[0][0] (cycle 1) and PE[0][1] (cycle 2)
- A[1][0]=3 is used at PE[0][0] (cycle 2) and PE[0][1] (cycle 3)
- A[0][1]=2 is used at PE[1][0] (cycle 2) and PE[1][1] (cycle 3)
- A[1][1]=4 is used at PE[1][0] (cycle 3) and PE[1][1] (cycle 4)

Every input element is loaded from off-chip once but used 2 times inside the array. Reuse factor = 2 (= N, the number of output columns).

**(c) Off-chip memory accesses:**

| Tensor | Accesses | Notes |
|--------|----------|-------|
| A (input) | 4 reads | Each of the 4 elements loaded once from off-chip, streamed in from left |
| B (weights) | 4 reads | Each of the 4 weights loaded once during pre-load before computation |
| C (output) | 4 writes | Each result written once when it drains out the bottom |

**Total off-chip accesses = 12** (4 reads for A + 4 reads for B + 4 writes for C).

---

## (d) Output-Stationary

In output-stationary dataflow, the **partial sums (the accumulating output values)** stay fixed in each PE, while both the A input values and the B weight values stream through the array from outside.
