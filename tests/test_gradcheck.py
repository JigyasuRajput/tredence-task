"""The consolidated gradient-checking suite: every op, finite differences, honest tolerance."""

import pytest

from engine.gradcheck import DEFAULT_TOL, grad_check, standard_cases

CASES = standard_cases()


@pytest.mark.parametrize("case", CASES, ids=[name for name, _, _ in CASES])
def test_op_gradient_matches_finite_differences(case):
    name, build, params = case
    err = grad_check(build, params)
    assert err < DEFAULT_TOL, f"{name}: relative error {err:.2e} exceeds {DEFAULT_TOL:.0e}"
