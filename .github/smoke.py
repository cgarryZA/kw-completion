import numpy as np
import kw_completion as kw

n, N, tau, nu1 = 512, 8, 0.125, 1.0

def sample(seed):
    r = np.random.default_rng(seed)
    dW = r.normal(0.0, np.sqrt(tau), (n, N))
    dNc = r.poisson(nu1 * tau, (n, N)) - nu1 * tau
    return dW, dNc, np.cumsum(dW, axis=1)

# Fit on one sample, freeze, and score on an independent one.
dW_f, dNc_f, X_f = sample(0)
t_f = kw.templates(dW_f, dNc_f, nu1, tau)
coef = kw.fit(0.3 * t_f[:, :, 0] - 0.2 * t_f[:, :, 1], t_f,
              kw.features_constant(X_f, N), ridge=1e-10)

dW, dNc, X = sample(1)
tmpl = kw.templates(dW, dNc, nu1, tau)
feat = kw.features_constant(X, N)
M = kw.apply(coef, tmpl, feat)
assert M.shape == (n, N + 1) and np.all(M[:, 0] == 0.0)
z = kw.admissibility_max_se(np.diff(M, axis=1), dW, dNc)
assert z < 6.0, z
assert np.isfinite(z), z
print("kw-completion smoke OK: M%s admissibility_max_se=%.3f" % (M.shape, z))
