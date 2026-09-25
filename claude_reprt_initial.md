# Confidence-Aware Rank-Blended Heuristic Search
### Formal Framework (v3 — Global Percentile Tracking, Rank-Space Blend)

---

## 1. Setup

Let $s$ denote a search state, $g(s)$ its accumulated path cost, and let there be two heuristic experts:

- **Classical expert** $h_{class}(s)$: an admissible geometric heuristic (e.g. Manhattan distance). Robust, monotone, but tie-heavy.
- **Learned expert** $\{h_m(s)\}_{m=1}^M$: an ensemble of $M$ neural networks trained with a pairwise ranking loss (following Chrestien et al., NeurIPS 2023). Each $h_m$ is an *arbitrary, uncalibrated scalar* — only its ordering across states is meaningful, not its magnitude.

The goal is a hybrid evaluation function $f(s) = g(s) + h_{blend}(s)$ for A\*, where $h_{blend}$ leans on the learned expert when it is confident and reliably decisive, and falls back toward the classical expert otherwise.

---

## 2. Global Percentile Tracking

For each of the $M$ learned seeds, maintain a sorted structure $\mathcal{H}_m$ (BST or sorted array) over **every** value $h_m(\cdot)$ produced so far in the search — not just the current expansion batch. When a new state $s$ is scored by seed $m$:

$$
p_m(s) = \frac{\text{rank of } h_m(s) \text{ in } \mathcal{H}_m}{|\mathcal{H}_m|} \in [0,1]
\tag{1}
$$

after which $h_m(s)$ is inserted into $\mathcal{H}_m$. Both the rank lookup and the insert are $O(\log V)$, where $V$ is the number of states scored so far — matching the $O(\log N)$ cost already paid for the A\* heap push, so this preserves the search's asymptotic per-step complexity.

The same machinery is applied to the classical heuristic: a sorted structure $\mathcal{H}_{class}$ gives a **global classical percentile**

$$
p_{class}(s) = \frac{\text{rank of } h_{class}(s) \text{ in } \mathcal{H}_{class}}{|\mathcal{H}_{class}|} \in [0,1]
\tag{2}
$$

This is a genuinely new object relative to the original spec: it puts *both* experts on the same footing (a global percentile) rather than converting only the learned side into the classical expert's raw units.

---

## 3. Epistemic Uncertainty via Rank Variance

Given the $M$ percentiles $p_1(s), \dots, p_M(s)$ from Eq. (1):

$$
\bar p(s) = \frac{1}{M}\sum_{m=1}^{M} p_m(s), \qquad
\sigma^2_{rank}(s) = \frac{1}{M}\sum_{m=1}^{M} \big(p_m(s) - \bar p(s)\big)^2
\tag{3}
$$

Since each $p_m(s) \in [0,1]$, $\sigma^2_{rank}(s)$ is maximized at $0.25$, achieved only when the ensemble is perfectly polarized (half the seeds rank $s$ best, half rank it worst).

**Confidence gate.** Convert rank variance into a confidence multiplier $\lambda(s) \in [\lambda_{min}, 1]$, with a safety floor $\lambda_{min}$ (e.g. $0.25$) that prevents the classical heuristic from being fully discarded:

$$
\lambda_{raw}(s) = \max\!\big(0,\ 1 - 4\,\sigma^2_{rank}(s)\big), \qquad
\lambda(s) = \lambda_{min} + (1-\lambda_{min})\,\lambda_{raw}(s)
\tag{4}
$$

$\lambda(s) \to 1$ when the ensemble agrees (low variance, in-distribution); $\lambda(s) \to \lambda_{min}$ when the ensemble is polarized (likely OOD).

---

## 4. Rank-Space Convex Blend (Option B)

**Why not project percentile onto $h_{class}$'s raw scale.** A percentile is a *relative* order statistic; it carries no information about absolute remaining cost. Projecting $\bar p(s)$ onto a running $[\mu_{min}, \mu_{max}]$ window of $h_{class}$ requires that window to be stable, but $\mu_{min}, \mu_{max}$ drift monotonically as the search approaches the goal — meaning nodes pushed to the open list at different times would be scored on different scales, silently violating the requirement that $f(s)$ stay fixed once pushed to the heap.

**Fix: blend in rank-space, convert to real units exactly once.** Both experts now already live in $[0,1]$ (Eqs. 1–2), so no per-node projection is needed. Blend directly:

$$
p_{blend}(s) = \lambda(s)\,\bar p(s) + \big(1-\lambda(s)\big)\,p_{class}(s)
\tag{5}
$$

Convert to real cost units with a **single fixed scale constant** $C$, set once (e.g. from a short calibration/warm-up pass, or as the map's diameter / an estimate of the initial $h_{class}$ at the start state) rather than updated online:

$$
h_{blend}(s) = p_{blend}(s) \cdot C
\tag{6}
$$

$$
f(s) = g(s) + h_{blend}(s)
\tag{7}
$$

Because $C$ is fixed once for the whole search, all nodes in the open list are compared on a consistent scale at every point in time — the static-priority requirement is satisfied by construction, not by approximation.

---

## 5. Why This Still Beats Both Baselines

- **vs. classical A\*:** $p_{blend}(s)$ inherits the learned ranker's fine-grained tie-breaking (percentiles are continuous, unlike raw Manhattan-distance plateaus), while $\lambda_{min}$ guarantees a bounded floor of classical-heuristic influence at all times — the search cannot drift arbitrarily far from an admissible, deadlock-aware signal.
- **vs. pure ranking-based search (Chrestien et al.):** For **GBFS** ($f = h$), a monotonic transform of $h_{nn}$ preserves expansion order, so global rank alone (no classical blend) is already sufficient — this is precisely Chrestien et al.'s own argument for ranking losses. For **A\*** specifically, $g(s)$ is in real cost units, so $h(s)$ must be too; Eq. (5)–(6) is the minimal fix that supplies real units to a rank-based signal instead of directly reusing $h_{class}$'s raw scale as done in the original draft, while adding online, per-state confidence gating that the base paper has no analogue of at all.
- **Failure mode is graceful, not catastrophic:** with $C$ fixed, the worst outcome is a fixed offset between $p_{blend}$-derived cost and true cost — this affects tie-breaking quality and slightly hurts sub-optimality bounds, but never breaks heap consistency or search correctness, since $\lambda_{min}$ anchors the blend to the admissible expert regardless of scale error.

---

## 6. Complexity (per step, unchanged from earlier analysis, now confirmed consistent with the actual design)

| Operation | Cost |
|---|---|
| Percentile lookup + insert, per seed $m$ (Eq. 1) | $O(\log V)$ |
| Classical percentile lookup + insert (Eq. 2) | $O(\log V)$ |
| Rank variance + confidence gate (Eqs. 3–4) | $O(M)$ |
| Rank-space blend (Eq. 5) + scale (Eq. 6, fixed $C$) | $O(1)$ |
| Heap push with $f(s)$ (Eq. 7) | $O(\log N)$ |

With $M$ a small constant, total per-node overhead is $O(\log V)$ on top of standard A\*'s $O(\log N)$ heap push — asymptotically unchanged, though the ensemble's $M$ forward passes remain the dominant **wall-clock** cost in practice and should be reported separately from asymptotic complexity in any empirical write-up.

---

## 7. Open Items to Resolve Next

1. **Choice of $C$.** Calibration-run estimate vs. fixed constant from map geometry — needs an empirical comparison.
2. **Warm-up / cold-start.** Percentiles are meaningless with very small $|\mathcal{H}_m|$; early search steps may need a fallback ($\lambda(s) = \lambda_{min}$ by default until $|\mathcal{H}_m|$ exceeds some threshold).
3. **Training side.** Ensemble ranking loss formulation ($M$ seeds, diversity mechanism — bagging vs. random init only) is not yet specified.
