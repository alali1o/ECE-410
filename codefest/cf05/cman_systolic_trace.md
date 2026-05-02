# CF05 CMAN — Systolic Array Trace (Weight-Stationary)
ECE 410/510 Spring 2026

A = [[1, 2], [3, 4]]   B = [[5, 6], [7, 8]]   Expected C = [[19, 22], [43, 50]]

---

## (a) PE Diagram — Preloaded Weights

```
Left inputs ──►  PE[0][0]  ──►  PE[0][1]
(row 0)          w = 5           w = 6
                    │               │
                    ▼               ▼
Left inputs ──►  PE[1][0]  ──►  PE[1][1]
(row 1)          w = 7           w = 8
                    │               │
                    ▼               ▼
                C[*][0]         C[*][1]   (outputs drain downward)
```

Weights are pre-loaded from B and stay fixed throughout computation.
Inputs (from A) stream in from the left and propagate right one PE per cycle.
Partial sums propagate downward one PE per cycle.

---

## (b) Cycle-by-Cycle Trace

**Input skewing** (required so each A element meets its B partner at the right PE at the right time):
- Row 0 receives: A[0][0]=1 at cycle 1, A[1][0]=3 at cycle 2
- Row 1 receives: 0 at cycle 1 (bubble), A[0][1]=2 at cycle 2, A[1][1]=4 at cycle 3

**PE rule each cycle:**  `psum_out = psum_in (from above) + in_x * weight`

| Cycle | Row 0 input | Row 1 input | PE[0][0] psum | PE[0][1] psum | PE[1][0] psum | PE[1][1] psum | Output C |
|-------|-------------|-------------|---------------|---------------|---------------|---------------|----------|
| 1     | 1           | 0           | 0+1×5 = **5** | 0+0×6 = 0    | 0+0×7 = 0    | 0+0×8 = 0    | —        |
| 2     | 3           | 2           | 0+3×5 = **15**| 0+1×6 = **6**| 5+2×7 = **19**| 0+0×8 = 0  | C[0][0]=**19** |
| 3     | 0           | 4           | 0+0×5 = 0    | 0+3×6 = **18**| 15+4×7 = **43**| 6+2×8 = **22** | C[1][0]=**43**, C[0][1]=**22** |
| 4     | 0           | 0           | 0            | 0             | 0             | 18+4×8 = **50** | C[1][1]=**50** |

**Verification:** C = [[19, 22], [43, 50]] ✓

**How the numbers flow (cycle 2 example):**
- PE[0][1] psum=6: input 1 (from cycle 1, propagated right) × weight 6 = 6
- PE[1][0] psum=19: psum_in=5 (from PE[0][0] cycle 1, propagated down) + input 2 × weight 7 = 5+14 = 19

---

## (c) Counts

**(a) Total MAC operations:** 8
Each of the 4 PEs performs 2 MACs (one per output row of A × the K=2 shared dimension).
4 PEs × 2 MACs = 8 total, matching the 2×2×2 GEMM count.

**(b) Input reuse:**
Each A element is reused **2 times** — once at each PE in its row (column 0 PE and column 1 PE).
Example: A[0][0]=1 enters PE[0][0] at cycle 1, then propagates to PE[0][1] at cycle 2.
The weight-stationary array gets its data-reuse from inputs passing through all N=2 column PEs.

**(c) Off-chip memory accesses:**

| Data | Accesses | Direction |
|------|----------|-----------|
| A    | 4 reads  | One load per element, streamed in from left |
| B    | 4 reads  | One load per weight during pre-load phase |
| C    | 4 writes | One store per output element as it drains out |

**Total: 12 off-chip accesses** (4+4+4).

---

## (d) Output-Stationary

In output-stationary dataflow, the **partial sums (accumulators) for each output element C[i][j]** stay fixed in their assigned PE, while both the A row elements and B weight values stream through the array.
