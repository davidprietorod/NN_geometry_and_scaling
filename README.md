# NN geometry and scaling

From-scratch reimplementations of results from two papers on the geometry of neural networks: what
structure a network acquires when trained, and what governs how fast it improves as it
grows. Both reproduce the original analyses, including the checks the authors themselves
run, and report where the results match and where they do not.

---

## 1. Scaling laws and the dimension of the data manifold

**Paper:** Sharma & Kaplan, *A Neural Scaling Law from the Dimension of the Data Manifold*
([arXiv:2004.10802](https://arxiv.org/abs/2004.10802)) where loss falls off with parameter count
as `L(N) ≈ C₀ + C₁·N^(-α)`, with `α ≈ 4/d` for a data manifold of dimension `d`.

**Setup.** Teacher–student, with the data dimension fixed by construction: Gaussian latents
`z ∈ ℝ^d` embedded linearly into `ℝ²⁰`, a fixed untrained teacher generating the labels, and
20 students of width 8 to 168. Latent dimensions `d = 2, 4, 6, 8, 10, 12`. Alongside the
imposed latent dimension, the intrinsic dimension of the penultimate-layer activations is
estimated with the Levina–Bickel MLE estimator, as in the paper.

### Results

The power-law form reproduces and `α` falls monotonically with `d`, but the coefficient
comes out consistently above `4/d` by a factor of 1.3 to 1.9, widening with `d`. The
qualitative claim replicates while the quantitative shows an important deviation, at least at this training
budget.

![Scaling law fits](Error_vs_dimension/output/scaling_law_fits.pdf)

Intrinsic dimension tracks the exponent more closely than latent dimension, consistently
under both fitting procedures:

| Regression of … against 1/α | `curve_fit` | log-log |
|---|---|---|
| latent dimension `d` | 9.41 | 7.60 |
| intrinsic dimension | 7.81 | 5.80 |
| *theory* | *4* | *4* |

Intrinsic dimension also saturates below the latent one as `d` grows (≈10.9 against 12),
consistent with the network not resolving every latent direction.

`d = 2` is excluded from the last step in the regressions: training reaches the Bayes error at nearly every
width, so `C₀` absorbs the loss and the fit is unreliable.

| Main Files | Contents |
|---|---|
| `scaling_law_experiment.ipynb` | Full sweep — trains everything and saves results. Expensive. |
| `scaling_law_analysis.ipynb` | Fits and figures from the saved `results.npz`. **Runs independently.** |

---

## 2. Kolmogorov–Arnold geometry in shallow MLPs

**Paper:** Freedman & Mulligan, *Spontaneous Kolmogorov-Arnold Geometry in Shallow MLPs*
([arXiv:2509.12326](https://arxiv.org/abs/2509.12326)) where KA-type geometry emerges on its own
when training ordinary single-hidden-layer MLPs, readable in the Jacobian of the first-layer
map.

**Setup.** Target `f(x) = ∏ᵢ sin(π xᵢ)` on `[-1,1]³`, the *xor* function of Eq. (III.1).
GeLU MLPs of width 4 to 64; for `w = 64`, the size-`k` minors of the Jacobian (exterior
powers `k = 1, 2, 3`) sorted and plotted log-log against rank. The trained network is
compared against the same network untrained from an identical initialisation, the paper's
own control for separating learned structure from initialisation.

### Results

Training collapses the spectrum, and the effect grows with `k`. At `k = 3` the gap spans
several orders of magnitude: the trained network falls below `10⁻¹⁶` where the untrained one
sits around `10⁻¹¹`.

![Jacobian rank structure, k=3](Rank_analysis/output/jacobian_rank_w64_k3.pdf)

Low-rank structure is not present at initialisation — learning produces it. That the effect
sharpens with `k` is what one expects if the inner map is aligning with a small number of
directions. This replicates the paper's finding.

| Main File | Contents |
|---|---|
| `ka_geometry_experiment.ipynb` | Training, minor computation and figures. |
