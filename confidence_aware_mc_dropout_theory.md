# Confidence-Aware Rank-Blended Heuristic Search via Test-Time MC Dropout
### Formal Theoretical Framework & Experimental Protocol

---

## 1. Executive Summary & Core Concept

Standard learned heuristics (such as the pairwise ranking loss $\mathcal{L}^*$ in Chrestien et al., NeurIPS 2023) achieve remarkable search efficiency by ordering states effectively, but suffer from two major flaws:
1. **Arbitrary / Uncalibrated Scale:** The output $h_\theta(s)$ is an uncalibrated scalar with no guarantee of admissibility ($h(s) \le h^*(s)$) or uniform magnitude, making direct addition $g(s) + h_\theta(s)$ in A\* problematic.
2. **Overconfidence in Out-of-Distribution (OOD) / Deadlock States:** When encountering unfamiliar maze configurations, deterministic neural heuristics make confident, incorrect ranking decisions that lead the search into massive deadlocks.

**The Solution:** We implement a **Confidence-Aware Hybrid A\*** search that dynamically evaluates epistemic uncertainty online using **Monte Carlo Dropout (MC Dropout)** from a single pre-trained model. 
* When the neural ranker is **confident** (low variance across dropout passes), the search exploits the learned ranker to aggressively prune node expansions.
* When the neural ranker is **uncertain** (high variance across dropout passes), the search automatically shifts weight toward a reliable classical admissible heuristic $h_{class}(s)$, preventing deadlocks.

---

## 2. Mathematical Formalization

```
                          ┌───────────────────────────┐
                          │   Current State s         │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌───────────────────────┐                    ┌─────────────────────────┐
     │  Classical Heuristic  │                    │  Pre-Trained Net f_θ(s) │
     │     h_class(s)        │                    │  with Test-Time Dropout │
     └───────────┬───────────┘                    └────────────┬────────────┘
                 │                                             │
                 │ [Rank in H_class]                           │ [M Stochastic Passes]
                 ▼                                             ▼
        p_class(s) ∈ [0, 1]                      {h_1(s), ..., h_M(s)}
                 │                                             │
                 │                                             │ [Rank in {H_m}]
                 │                                             ▼
                 │                                    {p_1(s), ..., p_M(s)}
                 │                                             │
                 │                         ┌───────────────────┴───────────────────┐
                 │                         ▼                                       ▼
                 │                   Mean Rank p̄(s)                        Rank Variance σ²(s)
                 │                         │                                       │
                 │                         │                                       ▼
                 │                         │                               Confidence Gate
                 │                         │                             λ(s) ∈ [λ_min, 1]
                 │                         │                                       │
                 └─────────────────────────┼───────────────────────────────────────┘
                                           ▼
                            Rank-Space Convex Combination:
                     p_blend(s) = λ(s)·p̄(s) + (1 - λ(s))·p_class(s)
                                           │
                                           ▼ [Scale by Constant C]
                                      h_blend(s)
                                           │
                                           ▼
                                 f(s) = g(s) + h_blend(s)
                                           │
                                           ▼
                                    [Push to A* Heap]
```

### 2.1. Test-Time Monte Carlo Dropout (Sampling Virtual Seeds)
Given a single pre-trained network $f_\theta(s)$, we keep mild dropout ($p \in [0.05, 0.15]$) active at inference time. For any state $s$, we perform $M$ stochastic forward passes:
$$h_m(s) = f_{\theta, \text{mask}_m}(s), \quad m \in \{1, \dots, M\}$$
By Gal & Ghahramani (2016), this acts as variational sampling from the weight posterior $p(\theta | \mathcal{D})$ without requiring $M$ distinct networks to be trained.

### 2.2. Global Empirical Percentile Tracking ($\mathcal{H}_m$ and $\mathcal{H}_{class}$)
Because $h_m(s)$ and $h_{class}(s)$ live on completely different scales, we map both into a unified **rank-percentile space** $[0, 1]$.

For each virtual seed $m \in \{1, \dots, M\}$, we maintain a dynamic sorted structure $\mathcal{H}_m$ storing all values evaluated so far. When evaluating state $s$:
$$p_m(s) = \frac{\text{rank of } h_m(s) \text{ in } \mathcal{H}_m}{|\mathcal{H}_m|} \in [0, 1] \tag{1}$$

Similarly, for the classical heuristic $h_{class}(s)$ (e.g. Hungarian Manhattan matching), we maintain $\mathcal{H}_{class}$:
$$p_{class}(s) = \frac{\text{rank of } h_{class}(s) \text{ in } \mathcal{H}_{class}}{|\mathcal{H}_{class}|} \in [0, 1] \tag{2}$$

Both rank lookup and insertion take $O(\log V)$ using `bisect`, preserving the search's per-step asymptotic complexity.

### 2.3. Epistemic Uncertainty via Rank Variance
From the $M$ percentiles $\{p_1(s), \dots, p_M(s)\}$, compute the ensemble mean and sample rank variance:
$$\bar{p}(s) = \frac{1}{M}\sum_{m=1}^M p_m(s) \tag{3}$$
$$\sigma^2_{rank}(s) = \frac{1}{M}\sum_{m=1}^M \big(p_m(s) - \bar{p}(s)\big)^2 \tag{4}$$

Since each $p_m(s) \in [0, 1]$, the maximum theoretical variance is $\sigma^2_{\max} = 0.25$ (achieved only when half the dropout samples rank $s$ at $0$ and half rank it at $1$).

### 2.4. Confidence Gating Function
We convert $\sigma^2_{rank}(s)$ into a confidence multiplier $\lambda(s) \in [\lambda_{min}, 1]$:
$$\lambda_{raw}(s) = \max\big(0,\ 1 - 4\sigma^2_{rank}(s)\big) \tag{5}$$
$$\lambda(s) = \lambda_{min} + (1 - \lambda_{min}) \cdot \lambda_{raw}(s) \tag{6}$$

* **In-distribution / Clear path:** All dropout samples agree $\implies \sigma^2_{rank} \to 0 \implies \lambda(s) \to 1.0$ (Trust learned ranker).
* **Ambiguous / Deadlock state:** Dropout samples disagree $\implies \sigma^2_{rank} \to 0.25 \implies \lambda(s) \to \lambda_{min}$ (Fall back to classical heuristic).
* **Safety Floor $\lambda_{min}$ (e.g. $0.20$):** Guarantees that the classical deadlock-aware signal is never completely discarded.

### 2.5. Rank-Space Convex Blend & Cost Scaling
We compute the blended percentile:
$$p_{blend}(s) = \lambda(s)\bar{p}(s) + \big(1 - \lambda(s)\big) p_{class}(s) \in [0, 1] \tag{7}$$

To use this within A\*, we convert the percentile back to cost units using a **single static scale constant $C$** (e.g. map diameter $C \approx 50$):
$$h_{blend}(s) = p_{blend}(s) \cdot C \tag{8}$$
$$f(s) = g(s) + h_{blend}(s) \tag{9}$$

---

## 3. Search Mechanics, Open List, and Node Ordering

### 3.1. Open List Priority Queue
The Open List is maintained as a min-heap. Each entry is keyed by a tuple:
$$\text{Priority Key}(s) = \big(f(s),\ h_{blend}(s),\ \text{tie\_breaker}\big)$$
* **Primary Key ($f(s)$):** Minimizes estimated total path cost $g(s) + h_{blend}(s)$.
* **Secondary Key ($h_{blend}(s)$):** In case of equal $f$-values, prioritizes states closer to the goal.
* **Tertiary Key ($\text{tie\_breaker}$):** Strict FIFO counter to guarantee deterministic heap operations.

### 3.2. Static Scale Invariance (Heap Consistency)
Because $C$ is a fixed constant determined before the search begins:
* A node's $f(s)$ is computed once upon generation and remains static in the Open List.
* Unlike dynamic window normalization (which shifts dynamically and distorts existing heap keys), our static scale conversion guarantees that all nodes in the Open List are compared fairly on the exact same scale.

### 3.3. Closed List and Reopening Policy
* States are indexed in the Closed List by their compact byte-array representation.
* If a state is rediscovered with a strictly lower $g$-cost ($g_{\text{new}} < g_{\text{old}}$), it is updated and re-inserted into the Open List.

---

## 4. Experimental Steps & Evaluation Protocol

To empirically validate the framework for your coursework, we will structure the evaluation into 4 distinct experimental configurations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPERIMENT MATRIX:                                                          │
│                                                                             │
│ 1. Baseline A: Classical Admissible A* (Hungarian Manhattan)               │
│    • Theoretical benchmark for optimal expansions vs speed.                │
│                                                                             │
│ 2. Baseline B: Pure Learned A* / GBFS (Chrestien et al. L*)                 │
│    • Deterministic neural evaluation (no uncertainty, no classical blend).  │
│                                                                             │
│ 3. Ablation: Naive Fixed-Weight Hybrid (λ = 0.5 static)                     │
│    • Tests whether dynamic uncertainty gating is better than constant blend.│
│                                                                             │
│ 4. Proposed Method: Confidence-Aware MC-Dropout Hybrid A*                   │
│    • Dynamic λ(s) driven by MC Dropout rank variance.                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Metrics to Track & Report:
1. **Solve Rate (%)**: Percentage of maps solved within a fixed timeout (e.g. 10s).
2. **Node Expansions**: Mean, median, and standard deviation of expanded nodes.
3. **Solution Path Cost**: Path length (checking sub-optimality trade-offs).
4. **Search Time (ms)**: Wall-clock execution time per search.
5. **Uncertainty Calibration Plot**: Correlation curve between rank variance $\sigma^2_{rank}(s)$ and true sub-optimality / heuristic error.
