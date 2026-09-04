r"""Martingale completion: a structurally admissible dictionary M-head.

The exact discrete solution of an MV-FBSDEJ is a five-component tuple
(X, Y, Z, U, M); a standard neural architecture outputs (Y, Z, U) and is scored
with the zero completion M-hat = 0, which faces the fixed-grid floor
E|M^pi_N|^2 (the zero-completion identity). This module supplies the completion
stage: a dictionary M-head whose increments are admissible under the contract
below, so the completed candidate stays in the class S_pi and the two-sided
estimator applies to it as stated.

Design: M-hat increments are F_{t_i}-measurable linear
combinations of fixed second-chaos template increments of the drivers,

    Psi^W_i  = dW_i^2 - tau                     (Brownian chaos-2)
    Psi^WN_i = dW_i * dNc_i                     (mixed chaos)
    Psi^N_i  = dNc_i^2 - dN_i                   (Poisson chaos-2, I_2)

with dNc = dN - nu1*tau the compensated increment.  Each template has exact
conditional mean zero and exact conditional orthogonality to dW_i and to the
compensated-Poisson first chaos; NOTE the Poisson element carries the
product-formula correction (I_1(1)^2 = I_2 + nu1*tau + I_1(1), so
I_2 = dNc^2 - nu1*tau - dNc = dNc^2 - dN): the naive dNc^2 - nu1*tau is NOT
orthogonal to dNc (compensated Poisson is skewed; E[dNc^3] = nu1*tau != 0).

Admissibility contract
----------------------
(i) Integrability.  Admissibility is preserved under F_{t_i}-measurable
    coefficients c with c in L^2 (bilinearity of the conditional
    covariances) -- and for these templates L^2 is SHARP: the conditional
    second moments E[(Psi^k_i)^2 | F_{t_i}] are the deterministic constants
    (2 tau^2, nu1 tau^2, 2 nu1^2 tau^2), so E[(c Psi)^2] = const * E[c^2]
    and c in L^2 is necessary and sufficient for the increment c*Psi to lie
    in L^2 (on active channels q > 0; at nu1 = 0 the two Poisson
    templates vanish identically and impose nothing).  Measurability alone is NOT enough (e.g. c = exp(X_0^2) is
    F_{t_0}-measurable but can have E[c^2] = infinity, giving a martingale
    outside S_pi that the mean-identity diagnostic cannot detect).  The
    shipped feature maps are bounded by 1 and the fitted coefficients are
    finite constants, so every head this module itself builds satisfies
    the contract.

(ii) Fit -> freeze -> fresh scoring paths.  The coefficients returned by
    fit() are functions of the whole fitting sample, so ON THE FITTING
    PATHS they are not F_{t_i}-measurable and admissibility does not apply
    there; in-sample scoring is data-snooping-optimistic.  The contract is

        fit data -> c-hat -> freeze c-hat -> fresh scoring paths:

    conditional on the fit data, c-hat is a fixed constant and
    phi_f(X_i) * c-hat is F_{t_i}-measurable on the scoring basis.  In
    dataset terms: D_train (neural training) and D_Mfit (this head's fit)
    may coincide, but D_score must be independent of everything fitted --
    neural weights and M-head coefficients alike.

Under this contract the head fits

    Delta M-hat_i = sum_{k,f} c_{i,k,f} * phi_f(X_i) * Psi^k_i

by per-node least squares of the candidate's one-step backward residual rho^Y_i
(engine convention: rho^Y = dY + f*tau - Z dW - U dNc - dM).  For the zero
completion of the otherwise EXACT discrete tuple, rho^Y_i = Delta M^pi_i
exactly; for a general candidate the residual is the unresolved backward
defect (Y/Z/U errors included), of which the head captures the part
correlated with the dictionary.  The features phi_f are F_{t_i}-measurable
by construction.

The head spans a three-element second-chaos dictionary, not the full marked
second chaos; the exact M^pi generally carries all higher chaos, so completion
reduces the floor to the projection distance E|M^pi_N - P_D M^pi_N|^2 and need
not remove it (the projection-floor theorem).

References
----------
Chunrong Feng and Christian Garry, "Martingale Completion for Time-Discrete
Stochastic Solvers" (2026).  The named results below -- the projection-floor
theorem, the order-elevation proposition of Appendix B, and the
dictionary-admissibility proposition -- are results of that paper.  See
CITATION.cff for the machine-readable record.
"""
import numpy as np

K_TEMPLATES = 3  # Psi^W, Psi^WN, Psi^N


def _arr(name, a, ndim):
    """asarray + rank check at the API boundary (house style: estimator.py)."""
    a = np.asarray(a, dtype=float)
    if a.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-d array, got shape {a.shape}")
    return a


def _finite(name, a):
    """Return ``a`` unchanged, raising if it holds any NaN or Inf."""
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values (NaN/Inf).")
    return a


def templates(dW, dNc, nu1, tau):
    """Second-chaos template increments, shape (n, N, 3).

    Columns: Psi^W = dW^2 - tau; Psi^WN = dW*dNc; Psi^N = dNc^2 - dN with
    dN = dNc + nu1*tau (the Poisson I_2; see module docstring for why the
    product-formula correction is mandatory)."""
    dW = _finite("dW", _arr("dW", dW, 2))
    dNc = _finite("dNc", _arr("dNc", dNc, 2))
    if dW.shape != dNc.shape:
        raise ValueError(f"templates: dW/dNc must be matching (n, N) arrays, "
                         f"got {dW.shape} and {dNc.shape}")
    if dW.shape[0] < 1:
        raise ValueError("templates: need at least one path (n >= 1)")
    if dW.shape[1] < 1:
        raise ValueError("templates: need at least one step (N >= 1)")
    tau = float(tau)
    if not (np.isfinite(tau) and tau > 0.0):
        raise ValueError(f"templates: tau must be finite and > 0 (got {tau!r})")
    nu1 = float(nu1)
    if not (np.isfinite(nu1) and nu1 >= 0.0):
        raise ValueError(f"templates: nu1 must be finite and >= 0 (got {nu1!r})")
    dN = dNc + nu1 * tau
    return np.stack([dW * dW - tau, dW * dNc, dNc * dNc - dN], axis=2)


def marked_templates(dW, jump_modes, pair_raw, tau):
    """Marked second-chaos templates, shape (n, N, 1 + J + J*(J+1)//2).

    Generalises :func:`templates` from a single Poisson count to J orthonormal
    mark modes zeta_1..zeta_J in L2(nu).  Inputs per step:

      jump_modes[..., r] = J_r  = int zeta_r(e)   dNtilde(de)   (compensated)
      pair_raw[..., r, s] = A_rs = int zeta_r zeta_s(e) dN(de)  (RAW count)

    Columns are dW^2 - tau, then dW*J_r, then J_r*J_s - A_rs for r <= s.

    The pair correction subtracts the quadratic variation and NOT the mean, for
    the same reason the single-count Psi^N does: by the Poisson product formula
    J_r J_s - A_rs = I_2(zeta_r sym zeta_s), which is pure second chaos and so
    orthogonal to int a dNtilde for every a in L2(nu), not merely for the J
    modes carried here.  Subtracting tau*delta_rs instead would leave a
    first-chaos component I_1(zeta_r zeta_s) behind and break admissibility.

    No finite-total-mass assumption is used, so nu may be sigma-finite with
    nu(E) = infinity; the modes need only satisfy zeta_r, zeta_r*zeta_s in
    L2(nu) for the displayed integrals to exist."""
    dW = _finite("dW", _arr("dW", dW, 2))
    jump_modes = _finite("jump_modes", _arr("jump_modes", jump_modes, 3))
    pair_raw = _finite("pair_raw", _arr("pair_raw", pair_raw, 4))
    n, N = dW.shape
    if n < 1:
        raise ValueError("marked_templates: need at least one path (n >= 1)")
    if N < 1:
        raise ValueError("marked_templates: need at least one step (N >= 1)")
    if jump_modes.shape[:2] != (n, N):
        raise ValueError(f"marked_templates: dW/jump_modes must share (n, N), "
                         f"got {dW.shape} and {jump_modes.shape}")
    J = jump_modes.shape[2]
    if J < 1:
        raise ValueError("marked_templates: need at least one mode (J >= 1)")
    if pair_raw.shape != (n, N, J, J):
        raise ValueError(f"marked_templates: pair_raw must be (n, N, J, J) = "
                         f"{(n, N, J, J)}, got {pair_raw.shape}")
    if J > 1 and not np.allclose(pair_raw, np.swapaxes(pair_raw, -1, -2),
                                 rtol=1e-9, atol=1e-12):
        raise ValueError(
            "marked_templates: pair_raw must be symmetric in its last two "
            "axes (A_rs = A_sr); only the upper triangle is read.")
    tau = float(tau)
    if not (np.isfinite(tau) and tau > 0.0):
        raise ValueError(f"marked_templates: tau must be finite and > 0 "
                         f"(got {tau!r})")
    cols = [(dW * dW - tau)[..., None], dW[..., None] * jump_modes]
    for r in range(J):
        for s in range(r, J):
            cols.append(
                (jump_modes[..., r] * jump_modes[..., s]
                 - pair_raw[..., r, s])[..., None])
    return np.concatenate(cols, axis=2)


def marked_template_variances(tau, n_modes):
    """Exact conditional Gram diagonal for :func:`marked_templates`.

    For orthonormal modes the Gram is diagonal with 2*tau^2 on dW^2-tau,
    tau^2 on each dW*J_r, and (1 + delta_rs)*tau^2 on each pair template.
    For J = 2 this is (2, 1, 1, 2, 1, 2) * tau^2."""
    tau = float(tau)
    if not (np.isfinite(tau) and tau > 0.0):
        raise ValueError(f"marked_template_variances: tau must be finite and "
                         f"> 0 (got {tau!r})")
    J = int(n_modes)
    if J < 1:
        raise ValueError("marked_template_variances: need J >= 1")
    q = [2.0 * tau ** 2] + [tau ** 2] * J
    for r in range(J):
        for s in range(r, J):
            q.append((2.0 if r == s else 1.0) * tau ** 2)
    return np.asarray(q)


def features_constant(X, N):
    """Coefficient features phi_f(X_i) = [1]: per-node constant coefficients.

    Applies the same state contract as :func:`features_trig` -- rank-2 finite X
    with at least N time nodes -- though the constant feature never reads X."""
    X = _finite("X", _arr("X", X, 2))
    if X.shape[0] < 1:
        raise ValueError(f"features_constant: X must carry at least one path, "
                         f"got shape {X.shape}")
    if N < 1:
        raise ValueError(f"features_constant: need N >= 1 (got {N})")
    if X.shape[1] < N:
        raise ValueError(f"features_constant: X has {X.shape[1]} time nodes, "
                         f"need at least N={N}")
    return np.ones((X.shape[0], N, 1))


def features_trig(k):
    """Feature map [1, cos(k X_i), sin(k X_i)] (F_{t_i}-measurable).

    Matches the sine benchmark's innovation structure: the chaos-2
    coefficients of the innovation are proportional to
    Re/Im(Lambda_i e^{ik X_i}) in the phi-normalised kernel convention of
    Appendix B of that paper (order-elevation proposition: this span makes the
    projection remove the entire second chaos)."""
    k = float(k)
    if not np.isfinite(k):
        raise ValueError(f"features_trig: k must be finite (got {k!r})")
    def phi(X, N):
        X = _finite("X", _arr("X", X, 2))
        if N < 1:
            raise ValueError(f"features_trig: need N >= 1 (got {N})")
        if X.shape[1] < N:
            raise ValueError(
                f"features_trig: X has {X.shape[1]} nodes, need >= N={N}")
        Xi = X[:, :N]
        return np.stack([np.ones_like(Xi), np.cos(k * Xi), np.sin(k * Xi)],
                        axis=2)
    return phi


def features_trig_nodes(kvec):
    """Feature map [1, cos(k_i X_i), sin(k_i X_i)] with a PER-NODE wavenumber
    vector k_i (F_{t_i}-measurable).

    Generalises :func:`features_trig` to a node-dependent frequency: at a
    CONSTANT k-vector the two produce identical feature arrays.  The
    node-dependent span is what a benchmark with a coupled forward state
    needs -- a state-affine drift
    rescales the frequency down the grid (k_i = rho_i k_{i+1}, ``det["k"]``),
    so the analytic trig features of the design note are {1, cos(k_i x),
    sin(k_i x)}, not a single-frequency transplant.  Each feature is a bounded
    (by 1) deterministic function of X_i, hence F_{t_i}-measurable, so heads
    built on this map are admissible by the dictionary-admissibility
    proposition exactly as for
    :func:`features_trig` (the projection-floor theorem is stated for this
    dictionary)."""
    kvec = _finite("k", _arr("k", kvec, 1))

    def phi(X, N):
        X = _finite("X", _arr("X", X, 2))
        if N < 1:
            raise ValueError(f"features_trig_nodes: need N >= 1 (got {N})")
        if X.shape[1] < N:
            raise ValueError(
                f"features_trig_nodes: X has {X.shape[1]} nodes, need >= N={N}")
        if kvec.shape[0] < N:
            raise ValueError(
                f"features_trig_nodes: {kvec.shape[0]} frequencies for N={N} "
                f"nodes; the head was built on a different grid")
        kx = kvec[None, :N] * X[:, :N]
        return np.stack([np.ones_like(kx), np.cos(kx), np.sin(kx)], axis=2)

    return phi


def fit(rhoY, tmpl, feat, ridge=1e-10):
    """Per-node least squares of the residual on {phi_f(X_i) * Psi^k_i}.

    rhoY: (n, N) backward residuals of the zero-completed candidate;
    tmpl: (n, N, K) from templates(); feat: (n, N, F) F_{t_i}-measurable.
    Returns coeffs (N, K*F), to be FROZEN and applied on fresh paths
    (contract (ii) in the module docstring): the coefficients depend on the
    whole fitting sample, so admissibility holds on scoring paths
    independent of it, and in-sample scoring is optimistic.

    Numerics: the raw design columns have chaos variances
    (2 tau^2, nu1 tau^2, 2 nu1^2 tau^2), a spread of order nu1^2 that makes
    raw normal equations near-singular away from nu1 ~ 1 (measured
    cond(Gram) = 3e16 at nu1 = 1e-3).  So each node solves an RMS-normalised
    AUGMENTED least-squares system in beta = coeffs * s,

        [D / s; diag(sqrt(n*ridge) * sbar / s)] beta ~ [rhoY_i; 0],

    via lstsq/SVD -- never forming D^T D (which squares the condition
    number).  The normalisation is PURE PRECONDITIONING: the penalty is in
    original coordinates with the global scale sbar^2 = mean(s^2)
    (= trace(Gram)/(K*F)), i.e. algebraically the classical objective
    (1/n)||D c - rho||^2 + ridge * sbar^2 * ||c||^2.  Deliberately NOT the
    standardised-ridge convention (penalty on beta): that inflates a
    rare-event column -- empirical scale estimated from ~1 sampled jump --
    to unit sample norm, lets it absorb residual by chance, and explodes
    the unscaled coefficient (measured max|c| = 8e3 under heavy ridge).
    Under the original-coordinate penalty, degenerate and rare columns are
    pinned toward 0 and ridge shrinks coefficients uniformly.

    Two rare-event caveats (both conservative-direction).  (1) On a fitting
    sample with no jumps the Poisson templates degenerate to first-chaos
    directions at scale nu1*tau (Psi^WN = -nu1*tau*dW exactly), which can sit
    above the s-floor and absorb noise into a large spurious jump
    coefficient at the default ridge; on scoring paths where a jump does
    fire, this inflates E_pi (never deflates it).  Fit jump coefficients
    only on samples that actually contain jumps.  (2) The penalty scale
    sbar^2 is global, so a TRUE column whose scale is below ~sqrt(ridge)
    of the dominant column is pinned toward 0 along with the degenerate
    ones; with the shipped templates the relevant ratio is ~nu1, so the
    cliff coincides with the no-jump regime, but external feature maps with
    a large dynamic range should lower ridge or rescale."""
    rhoY = _finite("rhoY", _arr("rhoY", rhoY, 2))
    tmpl = _finite("tmpl", _arr("tmpl", tmpl, 3))
    feat = _finite("feat", _arr("feat", feat, 3))
    n, N = rhoY.shape
    if n < 1:
        raise ValueError("fit: need at least one path (n >= 1)")
    if N < 1:
        raise ValueError("fit: need at least one step (N >= 1)")
    if tmpl.shape[:2] != (n, N) or feat.shape[:2] != (n, N):
        raise ValueError(
            f"fit: shape mismatch: rhoY {rhoY.shape}, tmpl {tmpl.shape}, "
            f"feat {feat.shape}")
    ridge = float(ridge)
    if not (np.isfinite(ridge) and ridge >= 0.0):
        raise ValueError(f"fit: ridge must be finite and >= 0 (got {ridge!r})")
    K, F = tmpl.shape[2], feat.shape[2]
    if K < 1 or F < 1:
        raise ValueError(
            f"fit: need at least one template column and one feature column "
            f"(got K={K}, F={F})")
    coeffs = np.empty((N, K * F))
    ridge_scale = np.sqrt(n * ridge)
    zeros = np.zeros(K * F)
    for i in range(N):
        D = (tmpl[:, i, :, None] * feat[:, i, None, :]).reshape(n, K * F)
        s = np.sqrt(np.mean(D * D, axis=0))    # per-column RMS scale
        if not np.isfinite(s).all():
            raise ValueError(
                f"fit: design column scales overflow double precision at node {i} "
                "(|entries| >~ 1e154 square to inf); rescale the inputs")
        s = np.maximum(s, max(1e-12 * float(s.max()), 1e-300))
        sbar = np.sqrt(np.mean(s * s))         # global scale, = sqrt(tr(G)/(K*F))
        A = np.vstack([D / s, np.diag(ridge_scale * sbar / s)])
        b = np.concatenate([rhoY[:, i], zeros])
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        coeffs[i] = beta / s
    return _finite("fitted coeffs", coeffs)


def apply(coeffs, tmpl, feat):
    """Build M-hat (n, N+1) from fitted coefficients: cumsum of the fitted
    increments, M-hat_0 = 0.  Admissible under the module contract: on paths
    independent of the fit of `coeffs`, each increment is a frozen
    F_{t_i}-measurable linear combination (bounded features, finite
    coefficients, hence L^2) of admissible templates."""
    coeffs = _finite("coeffs", _arr("coeffs", coeffs, 2))
    tmpl = _finite("tmpl", _arr("tmpl", tmpl, 3))
    feat = _finite("feat", _arr("feat", feat, 3))
    n, N = tmpl.shape[:2]
    if N < 1:
        raise ValueError("apply: need at least one step (N >= 1)")
    if feat.shape[:2] != (n, N):
        raise ValueError(
            f"apply: tmpl/feat path-node mismatch: {tmpl.shape} vs {feat.shape}")
    K, F = tmpl.shape[2], feat.shape[2]
    if coeffs.shape != (N, K * F):
        raise ValueError(f"apply: coeffs shape {coeffs.shape} != {(N, K * F)}")
    design = (tmpl[:, :, :, None] * feat[:, :, None, :]).reshape(n, N, K * F)
    dM = np.einsum('nik,ik->ni', design, coeffs)
    M = np.zeros((n, N + 1))
    M[:, 1:] = np.cumsum(dM, axis=1)
    return M


def admissibility_max_se(dM, dW, dNc):
    """Max |mean|/SE over the three per-node admissibility identities
    E[dM] = E[dM dW] = E[dM dNc] = 0 (statistical check on sampled paths;
    the identities hold exactly in conditional expectation by construction).

    Non-finite input raises; the standard error needs n >= 2 paths.

    Caveat (rare-jump regime): on a scoring sample containing no jumps, any
    head with a nonzero Psi^WN or Psi^N coefficient false-fails at
    z ~ sqrt(n/2) whatever the coefficient size, because the intensity cancels
    between mean and SE. Admissibility itself holds by construction."""
    dM = _finite("dM", _arr("dM", dM, 2))
    dW = _finite("dW", _arr("dW", dW, 2))
    dNc = _finite("dNc", _arr("dNc", dNc, 2))
    if not (dM.shape == dW.shape == dNc.shape):
        raise ValueError(
            f"admissibility_max_se: shapes differ: {dM.shape}, {dW.shape}, "
            f"{dNc.shape}")
    if dM.shape[1] < 1:
        raise ValueError("admissibility_max_se: need at least one step (N >= 1)")
    n = dM.shape[0]
    if n < 2:
        raise ValueError(
            f"admissibility_max_se: need n >= 2 paths for a standard error "
            f"(got n={n}; with one path the score is meaningless)")
    out = 0.0
    for v in (dM, dM * dW, dM * dNc):
        se = np.abs(v.mean(axis=0)) / (v.std(axis=0) / np.sqrt(n) + 1e-300)
        out = max(out, float(se.max()))
    return out
