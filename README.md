# kw-completion

[![CI](https://github.com/cgarryZA/kw-completion/actions/workflows/ci.yml/badge.svg)](https://github.com/cgarryZA/kw-completion/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kw-completion.svg)](https://pypi.org/project/kw-completion/)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/cgarryZA/kw-completion/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/cgarryZA/kw-completion/blob/main/LICENSE)

Kunita–Watanabe martingale completion for time-discrete stochastic solvers.

A discrete solver that outputs only the integrand channels leaves the orthogonal
martingale term of the Kunita–Watanabe decomposition at zero. This package builds
that missing channel: second-chaos template increments from the driving noise, a
per-node least-squares fit of a residual onto them, and the resulting martingale
path — with an admissibility diagnostic for checking that what was fitted really
is orthogonal to the drivers.

NumPy only. It knows nothing about any particular equation class: it takes
driving increments, a residual and a feature map, and returns coefficients and a
martingale path.

## Installation

```bash
pip install kw-completion
```

## Quick start

Fit on one sample, then freeze the coefficients and score on paths the fit never
saw. The admissibility diagnostic only carries its stated meaning on fresh paths.

```python
import numpy as np
import kw_completion as kw

# Fitting sample.
tmpl_fit = kw.templates(dW_fit, dNc_fit, nu1, tau)   # (n, N, 3) second-chaos increments
feat_fit = kw.features_trig(3)(X_fit, N)             # F_{t_i}-measurable coefficients
coef     = kw.fit(rhoY_fit, tmpl_fit, feat_fit, ridge=1e-8)   # per-node least squares

# Independent scoring sample; coef is frozen, apply does not refit.
tmpl = kw.templates(dW, dNc, nu1, tau)
feat = kw.features_trig(3)(X, N)
M    = kw.apply(coef, tmpl, feat)                    # (n, N+1), M[:, 0] == 0
z    = kw.admissibility_max_se(np.diff(M, axis=1), dW, dNc)
```

Reusing the fitting paths for scoring is permitted by the API and is useful for
reconstruction, but the resulting `z` is not an admissibility check. Sample
separation is the caller's responsibility.

### Array conventions

| Input | Shape | Meaning |
| --- | --- | --- |
| `dW`, `dNc` | `(n_paths, N)` | Brownian and compensated jump increments |
| `X` | `(n_paths, >= N)` | Conditioning state; only the first `N` nodes are read |
| `rhoY` | `(n_paths, N)` | Per-step residual the completion is fitted to |
| `tmpl` | `(n_paths, N, K)` | Template increments, `K = 3` for `templates` |
| `feat` | `(n_paths, N, F)` | Feature values, adapted to node `i` |
| `M` | `(n_paths, N + 1)` | Completed martingale path, `M[:, 0] == 0` |

`fit` solves one ridge-regularised least-squares problem per time node, so the
returned coefficients have shape `(N, K * F)` and are frozen thereafter: `apply`
does not refit.

## Marked jumps

`marked_templates` generalises the single-count case to `J` orthonormal mark modes
in L²(ν); ν may be σ-finite with infinite total mass. `marked_template_variances`
returns the per-template variances used to normalise them.

## Admissibility

`admissibility_max_se` reports the largest standardised deviation among the three
orthogonality identities `E[dM] = E[dM dW] = E[dM dNc] = 0`. It is a statistical
sanity check on a finite sample of scoring paths, not a proof: the identities hold
exactly in conditional expectation by construction, and a low score is evidence of
no *detectable* violation at that sample size rather than confirmation of the
theoretical identities. Values below roughly 6 are the usual pass band; the
statistic is a z-score, so it tightens as paths are added.

On a scoring sample containing no jumps the check false-fails at `z ~ sqrt(n/2)`
for any head with a nonzero `Psi^WN` or `Psi^N` coefficient, whatever its size,
because the jump intensity cancels between the mean and the standard error. Score
admissibility on samples where the drivers actually realise both channels.

## Scope

- Arrays are scalar-valued; the state is a rank-2 `(n_paths, nodes)` array.
- The templates are the second chaos of the two drivers. Richer dictionaries need
  their own template constructor.
- `fit` needs `N >= 1`; `admissibility_max_se` needs `n >= 2` paths for the
  standard error to exist.

## Tests

```bash
pip install -e ".[test]" && pytest -q
```

## Citation

If this package contributes to published work, please cite the software:

> Christian Garry. *kw-completion: Kunita–Watanabe martingale completion for
> time-discrete stochastic solvers*. Version 0.1.0, 2026.
> https://github.com/cgarryZA/kw-completion

Machine-readable metadata is in [`CITATION.cff`](https://github.com/cgarryZA/kw-completion/blob/main/CITATION.cff).

The methods implemented here are developed in the accompanying paper, Chunrong Feng
and Christian Garry, *Martingale Completion for Time-Discrete Stochastic Solvers*
(2026), which is listed as related work in `CITATION.cff` and is worth citing
alongside the software where the underlying theory is what matters. The record will
carry the paper's public identifier once one exists.

## Licence

Released under the [MIT License](https://github.com/cgarryZA/kw-completion/blob/main/LICENSE).
