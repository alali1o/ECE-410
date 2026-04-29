# CF04 CMAN — INT8 Symmetric Quantization
ECE 410/510 Spring 2026

---

## (a) Scale Factor

Scanning all 16 values for the largest absolute value:
max|W| = 2.31  (row 3, col 4)

S = 2.31 / 127 = **0.01819**

---

## (b) INT8 Matrix W_q = round(W / S)

Dividing each element by S = 0.01819 and rounding:

```
W_q = [  47,  -66,   19,  115 ]
      [  -4,   50, -103,    7 ]
      [  85,    2,  -24, -127 ]
      [ -10,   57,   42,   30 ]
```

No clamping needed — all values fall within [−128, 127].

Sample checks:
- 0.85 / 0.01819 = 46.73 → 47
- 2.10 / 0.01819 = 115.45 → 115
- −2.31 / 0.01819 = −127.0 → −127

---

## (c) Dequantized Matrix W_deq = W_q × S

Multiplying each INT8 value back by S = 0.01819:

```
W_deq = [  0.8549, -1.2005,  0.3456,  2.0917 ]
        [ -0.0728,  0.9094, -1.8735,  0.1273 ]
        [  1.5461,  0.0364, -0.4365, -2.3100 ]
        [ -0.1819,  1.0368,  0.7639,  0.5457 ]
```

---

## (d) Error Analysis

Per-element absolute error |W − W_deq|:

```
error = [ 0.0049,  0.0005,  0.0056,  0.0083 ]
        [ 0.0028,  0.0006,  0.0065,  0.0073 ]
        [ 0.0039,  0.0064,  0.0035,  0.0000 ]
        [ 0.0019,  0.0068,  0.0061,  0.0043 ]
```

Largest error: **0.0083** at W[0,3] = 2.10 → W_deq = 2.0917

MAE = 0.0692 / 16 = **0.00433**

---

## (e) Bad Scale — S_bad = 0.01

W_q_bad = round(W / 0.01), clamped to [−128, 127]:

```
W_q_bad = [  85, -120,   34, 127 ]    ← 2.10/0.01 = 210, clamped to 127
          [  -7,   91, -128,  12 ]    ← −1.88/0.01 = −188, clamped to −128
          [ 127,    3,  -44,-128 ]    ← 1.55→155→127; −2.31→−231→−128
          [ -18,  103,   77,  55 ]
```

W_deq_bad = W_q_bad × 0.01 → four elements are badly wrong:

| Element | Original | Reconstructed | Error |
|---------|----------|---------------|-------|
| [0,3]   |  2.10    |  1.27         | 0.83  |
| [1,2]   | −1.88    | −1.28         | 0.60  |
| [2,0]   |  1.55    |  1.27         | 0.28  |
| [2,3]   | −2.31    | −1.28         | 1.03  |

MAE_bad = 2.74 / 16 = **0.171**

When S is too small, values that exceed ±127 get hard-clamped, causing large errors that cannot be recovered during dequantization.
