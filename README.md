# Resonant Phase-Locking: Stabilizing Long-Horizon Reasoning and Continual Adaptation via Phase-Synchronized Feature Routing

**TL;DR**: A tiny phase module added to transformers stabilizes long-horizon reasoning and reduces catastrophic forgetting by *phase-locking* features that should cohere over time.

## What is Resonant Phase-Locking (RPL)?
RPL augments each token with a learnable phase and base frequency. Attention logits and MLP routing are modulated by phase alignment (cos Δθ). A Kuramoto-style coupling step learns to synchronize phases for subroutines that should persist (plans, indentation cycles, loops) and desynchronize unrelated processes.

## Why it works
- **Temporal coherence**: tokens that belong to the same algorithmic loop share phase and reinforce each other.
- **Distraction-robustness**: irrelevant tokens are off-phase and get down-weighted.
- **Continual learning**: synchronized routines are less likely to be overwritten by new tasks.

## Repo contents
- `rpl_paper/main.tex` — full paper (LaTeX)
- `rpl_paper/figs/*.png` — figures (generated)
- `code/rpl_demo.py` — minimal reference implementation (pseudocode-level)
- `launch_assets/` — copy/paste launch text (tweet thread, HN, Reddit)

## Quickstart (reference pseudo-impl)
See [`code/rpl_demo.py`](code/rpl_demo.py). Integrate `rpl_attention` into your block, add phase heads, and include the regularizers:
- `L_coh`: encourage synchrony where attention is high
- `L_sep`: prevent trivial global locking by separating low-attention pairs

## Citing
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17606823.svg)](https://doi.org/10.5281/zenodo.17606823)
