# Rule: Handling Ranking-Loss Neural Heuristics

**Description:** Guidelines for evaluating epistemic uncertainty and scaling neural predictions for models trained via shift-invariant pairwise ranking loss (e.g., $\mathcal{L}^*$).

## Context
When working with deep ensembles trained on pairwise ranking losses (where the model only learns that State A is closer than State B, not the actual distance), the neural networks will output values with arbitrary scalar offsets. This requires special mathematical handling when using them for A* search.

## Core Directives

1. **Never Use Raw Cross-Model Variance:**
   - Because ranking models settle on arbitrary static offsets (e.g., Model A outputs 100, Model B outputs -50), the raw standard deviation ($\sigma_{\text{raw}}$) between ensemble outputs will be a near-constant massive number. 
   - **Requirement:** You MUST use **zero-centered uncertainty** (subtracting each model's static mean offset over a batch) or map the outputs to a **Percentile Tracker space** before computing variance/std-dev.

2. **Understand the Meaning of $\sigma_h(s)$ (Topological Ambiguity):**
   - Epistemic uncertainty in this domain does not represent a standard Gaussian confidence interval over physical distance.
   - It represents *topological branching ambiguity*. Models agree tightly near the goal and disagree heavily far away or when faced with deceptive branches.

3. **Prefer Risk-Averse A* over Classical Fallbacks:**
   - Because high uncertainty directly correlates with high prediction error (the models are confused), do not use uncertainty to switch to a classical heuristic. 
   - **Requirement:** Use a Risk-Averse priority function: $f(s) = g(s) + \mu(s) + \kappa \cdot \sigma(s)$. This penalizes ambiguous branches and naturally pushes them to the back of the open list, letting the neural network resolve the topology safely.
