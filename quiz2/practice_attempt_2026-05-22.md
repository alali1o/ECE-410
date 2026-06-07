# Quiz 2 Practice Attempt — 2026-05-22

**Platform:** Shine! oral exam (Whisper transcription + Claude analysis with human oversight)
**Bao Nguyen | ECE 410/510 Spring 2026**

| Item | Value |
|------|-------|
| Score | **6.8 / 10** |
| Result | **Not Yet** (threshold 7.0 / 70%) |
| Delivery confidence | Medium |
| Practice window | 2026-05-18 → 2026-05-24 |
| Real Quiz 2 window | 2026-05-25 → 2026-05-31 (retakeable, 2 tokens) |

---

## Headline takeaway

Strong on **data-format math (BF16/FP16/FP32)** and **CSR sparse representation**, both Q2 and Q3 hit ~9/10 with confident follow-ups. Weak on **hands-on numerical traces** (Q1 systolic) and **circuit-specific node reasoning** (Q4 sneak path) — both questions where the answer needed a piece of paper or the original figure to point at, and Bao tried to reason about it out loud without a visual anchor.

Delivery confidence varied sharply by topic. On Q2/Q3, declarative sentences and few fillers ("BF16 is a sweet spot for training", "both arrays have exactly the same size"). On Q1/Q4, frequent self-interruptions ("Man, I need to write this on a piece of paper", "Ah, fuck", "Yeah, I have no idea"), restarted clauses, topic-hopping mid-sentence.

---

## Question-by-question breakdown

### Q1 — CF5 systolic reset between cycle 2 and cycle 3 — 5.0 / 10 ⚠
**Asked:** Why reset Row 0 partial sums between cycle 2 and cycle 3, and what specific value would C[1][0] take without the reset?

**Strong:** Identified the high-level reason (isolate output rows so accumulators don't carry leftover partial sums into the next output row). Stated that PE[0][0] would hold a leftover 5 from cycle 2.

**Weak:** Could not carry the arithmetic through. Attempted `5 + 3×5` and confused order of operations (`5+3=8, ×5=40`), then said "I have no idea." The concrete corrupted-vs-correct comparison was never delivered.

**The arithmetic I should have produced (memorize cold):**
- Correct C[1][0] = 43 = 3×5 + 4×7 = 15 + 28
- Without reset, PE[0][0] keeps the partial sum of 5 from cycle 2 instead of clearing to 0
- Cycle 3 PE[0][0] computes 5 + 3×5 = **20** (not 15) and sends 20 down to PE[1][0]
- Cycle 4 PE[1][0] computes 20 + 4×7 = **48** → corrupted C[1][0]
- Error = 48 − 43 = +5 = exactly the leftover partial sum that should have been cleared

**Fix in study materials:** added explicit no-reset trace to `cheatsheet.md` §3a and `practice_questions.md` §A5.

---

### Q2 — BF16 vs FP16 vs FP32 — 9.0 / 10 ✅
**Asked:** Pros and cons of BF16 vs FP16 and FP32.

**Strong:** Correct bit layouts cold (1/8/7, 1/5/10, 1/8/23). Explained BF16 dynamic-range advantage over FP16, precision tradeoff vs FP32, memory and throughput benefits. Follow-up on gradient swamping was excellent: 1 + 0.0001 example, 7-bit mantissa resolution limit, mixed-precision training with FP32 master weights as mitigation.

**Minor:** "128 distinct values between any powers of two" is approximate (actual = 2⁷ = 128, OK), and stumbled briefly when counting BF16 mantissa bits aloud.

**Keep doing:** Lead with bit-layout numbers, then explain dynamic range, then end with where it actually bites.

---

### Q3 — CSR `values` and `col_index` sizes — 9.0 / 10 ✅
**Asked:** Size of `values` and `col_index` arrays for the given matrix and why.

**Strong:** Identified both sizes = nnz = 8, enumerated non-zeros (3, 5, 2, 7, 1, 4, 6, 8) to justify the count, explained that the format saves memory because zeros are implied by what's not stored. Follow-up on row_pointer: correctly stated size n+1 = 7, explained bookmark/sentinel semantics, demonstrated slicing row i via pointer[i] to pointer[i+1].

**Minor:** Brief counting hiccup mid-enumeration, recovered cleanly.

**Keep doing:** Walk the array element by element when the question asks "what size and why" — the act of enumerating proves the count is right.

---

### Q4 — CF6 sneak path: why KCL at those two specific nodes? — 4.0 / 10 ⚠
**Asked:** Why write KCL at the two specific nodes that gave V_row1 = 0.4 and V_col1 = 0.6?

**Strong:** Acknowledged that some nodes were grounded and that grounding simplified analysis. Knew the answer numbers.

**Weak:** Sidestepped the actual question. Said "I literally redrew the entire circuit... actually didn't use KCL at all" and computed via Ohm's law on a redrawn network. That's a valid solving strategy but it didn't address *why those two nodes* are the KCL choice. Follow-up about grounded nodes was answered generically (ground = known 0 V, reduces unknowns) without pointing to a specific node in the actual circuit, partly because the original drawing wasn't in front of me.

**The answer I should have given:** Row 1 and col 1 are **floating** (undriven), so their voltages are unknown. Every other node in the 2×2 crossbar is pinned: V_row0 = 1 V (driven), V_col0 = 0 V (virtual ground at the sense amp). KCL at a floating node gives one equation per unknown, and there are exactly two unknowns (V_row1, V_col1), so two KCL equations is the minimum sufficient system. Writing KCL anywhere else either gives a redundant equation (pinned node, already known) or introduces no new information.

**Fix in study materials:** added explicit "which nodes to write KCL at and why" reasoning to `practice_questions.md` §D4 and `cheatsheet.md` §10a.

---

## What to drill before the real attempt (May 25 onward)

1. **CF5 trace under no-reset.** Redo the trace and write out C[1][0] = 48 (corrupted) vs 43 (correct), then explain it without notes. Five-minute drill.
2. **CF6 KCL node selection.** Be able to say out loud: "Row 1 and col 1 are floating, so their voltages are the two unknowns. KCL at each floating node gives the minimum sufficient system." That's the question being asked, every time.
3. **Visual aids.** For CF5 and CF6 questions, request paper before answering. The transcript shows my arithmetic falls apart without it. Asking for paper is not penalized; freezing is.
4. **Keep Q2/Q3 style.** Lead with the number, name the rule, end punchy. That cadence is what scored 9/10.

