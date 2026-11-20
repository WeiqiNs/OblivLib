import math

import pytest

from oblivlib.dependency.balls_into_bins import lambert_w


def test_lambert_w_satisfies_defining_equation():
    for x in (0.0, 0.5, 1.0, 2.0, 10.0, 100.0):
        w = lambert_w(x)
        assert abs(w * math.exp(w) - x) < 1e-6


def test_lambert_w_rejects_below_lower_bound_and_reports_non_convergence():
    with pytest.raises(ValueError):
        lambert_w(-1.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        lambert_w(10.0, max_iter=0)
