"""Numerical, shape, orthogonality and packaging guards for kw_completion."""
import importlib.metadata as md
import subprocess
import sys

import numpy as np
import pytest

import kw_completion as kw

DIST = "kw-completion"
TAU, NU1 = 0.05, 1.3


def driving(n=256, N=16, seed=7):
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(TAU), size=(n, N))
    dNc = rng.poisson(NU1 * TAU, size=(n, N)) - NU1 * TAU
    X = np.cumsum(np.concatenate([np.zeros((n, 1)), dW], axis=1), axis=1)
    return dW, dNc, X


def test_templates_shape_is_n_by_N_by_K():
    dW, dNc, _ = driving()
    t = kw.templates(dW, dNc, NU1, TAU)
    assert t.shape == (dW.shape[0], dW.shape[1], kw.K_TEMPLATES)


def test_templates_are_mean_zero_second_chaos():
    dW, dNc, _ = driving(n=200_000, N=4)
    t = kw.templates(dW, dNc, NU1, TAU)
    se = t.std(axis=0, ddof=1) / np.sqrt(t.shape[0])
    assert np.all(np.abs(t.mean(axis=0)) < 6.0 * se)


def test_templates_are_orthogonal_to_the_driving_increments():
    """Second-chaos templates must be uncorrelated with dW and dNc: that is what
    makes a fitted head an orthogonal martingale rather than a drift."""
    dW, dNc, _ = driving(n=200_000, N=4)
    t = kw.templates(dW, dNc, NU1, TAU)
    for drive in (dW, dNc):
        prod = t * drive[:, :, None]
        se = prod.std(axis=0, ddof=1) / np.sqrt(prod.shape[0])
        assert np.all(np.abs(prod.mean(axis=0)) < 6.0 * se)


def test_fit_then_apply_recovers_a_planted_combination():
    dW, dNc, X = driving(n=20_000, N=8)
    t = kw.templates(dW, dNc, NU1, TAU)
    feat = kw.features_constant(X, dW.shape[1])
    rhoY = 0.7 * t[:, :, 0] - 0.4 * t[:, :, 1] + 0.25 * t[:, :, 2]
    M = kw.apply(kw.fit(rhoY, t, feat, 1e-12), t, feat)
    assert M.shape == (dW.shape[0], dW.shape[1] + 1)
    assert np.allclose(np.diff(M, axis=1), rhoY, atol=1e-8)


def test_apply_starts_the_martingale_at_zero():
    dW, dNc, X = driving()
    t = kw.templates(dW, dNc, NU1, TAU)
    feat = kw.features_constant(X, dW.shape[1])
    M = kw.apply(kw.fit(0.1 * t[:, :, 0], t, feat, 1e-10), t, feat)
    assert np.all(M[:, 0] == 0.0)


def test_a_fitted_head_is_admissible():
    dW, dNc, X = driving(n=60_000, N=8)
    t = kw.templates(dW, dNc, NU1, TAU)
    feat = kw.features_trig(2)(X, dW.shape[1])
    M = kw.apply(kw.fit(0.3 * t[:, :, 0] - 0.2 * t[:, :, 1], t, feat, 1e-10), t, feat)
    assert kw.admissibility_max_se(np.diff(M, axis=1), dW, dNc) < 6.0


def test_feature_builders_return_the_documented_ranks():
    dW, _, X = driving()
    N = dW.shape[1]
    assert kw.features_constant(X, N).shape == (X.shape[0], N, 1)
    assert kw.features_trig(3)(X, N).shape == (X.shape[0], N, 3)
    assert kw.features_trig_nodes(np.arange(1, N + 1))(X, N).shape == (X.shape[0], N, 3)


def test_marked_templates_column_count_and_variances():
    n, N, J = 128, 6, 3
    rng = np.random.default_rng(3)
    dW = rng.normal(0.0, np.sqrt(TAU), size=(n, N))
    modes = rng.normal(size=(n, N, J)) * np.sqrt(TAU)
    raw = rng.poisson(0.4, size=(n, N, J, J)).astype(float)
    raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    K = 1 + J + J * (J + 1) // 2
    assert kw.marked_templates(dW, modes, raw, TAU).shape == (n, N, K)
    assert kw.marked_template_variances(TAU, J).shape == (K,)


def test_non_finite_input_is_rejected_loudly():
    dW, dNc, _ = driving(n=8, N=4)
    dW[0, 0] = np.nan
    with pytest.raises(ValueError):
        kw.templates(dW, dNc, NU1, TAU)


def test_wrong_rank_is_rejected():
    dW, dNc, _ = driving(n=8, N=4)
    with pytest.raises(ValueError):
        kw.templates(dW[:, :, None], dNc, NU1, TAU)


def test_distribution_installs_under_the_new_name():
    assert md.version(DIST) == kw.__version__


def test_the_only_runtime_dependency_is_numpy():
    reqs = [r for r in (md.distribution(DIST).requires or [])
            if "extra ==" not in r]
    assert [r.split(";")[0].strip() for r in reqs] == ["numpy"]


def test_the_package_imports_nothing_from_the_solver_codebase():
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys, kw_completion; "
         "bad=[m for m in sys.modules if m.split('.')[0] in "
         "('mvfbsdej','kw_fbsdej','fbsdej_error_estimator','torch')]; "
         "print(bad)"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout


def test_public_surface_is_exactly_the_documented_names():
    assert set(kw.__all__) == {
        "K_TEMPLATES", "templates", "marked_templates", "marked_template_variances",
        "features_constant", "features_trig", "features_trig_nodes",
        "fit", "apply", "admissibility_max_se", "__version__"}


def test_empty_basis_is_rejected_by_contract_not_by_numpy():
    """An empty template or feature axis must fail the package's own check first."""
    dW, dNc, _ = driving(n=32, N=5)
    t = kw.templates(dW, dNc, NU1, TAU)
    with pytest.raises(ValueError, match="feature column"):
        kw.fit(np.zeros((32, 5)), t, np.zeros((32, 5, 0)))
    with pytest.raises(ValueError, match="template column"):
        kw.fit(np.zeros((32, 5)), t[:, :, :0], np.zeros((32, 5, 1)))


def test_asymmetric_pair_raw_is_rejected():
    """A_rs is a pair count and so symmetric; only the upper triangle is read."""
    n, N, J = 32, 5, 2
    dW, _, _ = driving(n=n, N=N)
    rng = np.random.default_rng(3)
    modes = rng.normal(size=(n, N, J))
    pair = np.zeros((n, N, J, J))
    pair[..., 0, 1] = 1.0
    pair[..., 1, 0] = 1.0
    ok = kw.marked_templates(dW, modes, pair, TAU)
    assert np.all(np.isfinite(ok))
    pair[..., 1, 0] = -123.0
    with pytest.raises(ValueError, match="symmetric"):
        kw.marked_templates(dW, modes, pair, TAU)
