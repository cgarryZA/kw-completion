"""Kunita-Watanabe martingale completion for time-discrete stochastic solvers."""
from kw_completion._core import (
    K_TEMPLATES,
    templates,
    marked_templates,
    marked_template_variances,
    features_constant,
    features_trig,
    features_trig_nodes,
    fit,
    apply,
    admissibility_max_se,
)

__version__ = "0.1.0"

__all__ = [
    "K_TEMPLATES",
    "templates",
    "marked_templates",
    "marked_template_variances",
    "features_constant",
    "features_trig",
    "features_trig_nodes",
    "fit",
    "apply",
    "admissibility_max_se",
    "__version__",
]
