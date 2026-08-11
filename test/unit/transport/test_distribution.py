import math

import numpy as np
import pytest
from numba import njit

import mcdc.transport.distribution as dist

# ======================================================================================
# Regression coverage for sample_white_direction(nx, ny, nz, rng_state)
#
# The function samples a direction isotropically about a reference direction
# (nx, ny, nz). It has two branches, split on whether the reference direction is
# aligned with +/-z:
#   - the general case (abs(nz) != 1.0), and
#   - the polar special case (abs(nz) == 1.0), where x/y are swapped with z in the
#     formula to avoid a 0/0 in the general-case algebra.
#
# Before the fix landed (mcdc-project/mcdc PR #470), the branch guard was
# `if nz != 1.0`, so a reference direction of exactly (0, 0, -1) incorrectly took the
# general-case branch. That branch computes B = sqrt(1 - nz**2), which is
# sqrt(1 - 1) == 0 for nz == -1.0, and then divides by it (C = Ac / B), producing
# inf/nan direction components instead of a valid unit vector. The fix tightened the
# guard to `abs(nz) != 1.0`, sending both poles through the special-case branch.
#
# These tests were not present anywhere before (grepped test/unit for
# "sample_white_direction": zero hits at the time of writing), so they previously
# would not have caught a regression to the old `nz != 1.0` guard even though the bug
# itself is already fixed in current mcdc/transport/distribution.py.
# ======================================================================================

MOCK_RNG_STATE_DTYPE = np.dtype(
    [
        ("idx", np.int64),
        ("n_values", np.int64),
        ("values", np.float64, (4,)),
    ]
)


@njit
def _mock_lcg_known_sequence(rng_state):
    # Force rng.lcg to return pre-set values in order, mirroring the pattern used in
    # test/unit/distributions/conftest.py's mock_rng_sequence fixture.
    state = rng_state[0]
    i = state["idx"]
    value = state["values"][i]
    state["idx"] = i + 1
    return value


def _mock_rng_state(*values):
    state = np.zeros(1, dtype=MOCK_RNG_STATE_DTYPE)
    state[0]["idx"] = 0
    state[0]["n_values"] = len(values)
    state[0]["values"][: len(values)] = np.asarray(values, dtype=np.float64)
    return state


@pytest.fixture
def mock_rng(monkeypatch):
    monkeypatch.setattr(dist.rng, "lcg", _mock_lcg_known_sequence)


# A generic, non-degenerate draw: xi1 -> mu = sqrt(0.36) = 0.6, xi2 -> azi = 0.7 * 2*pi.
GENERIC_XI1 = 0.36
GENERIC_XI2 = 0.7


def _unit_reference_direction(nz):
    # Build a unit vector (nx, ny, nz) with nx = sqrt(1 - nz**2), ny = 0, matching how
    # RAVEN/mcdc physics code always calls sample_white_direction with a normalized
    # reference direction (see e.g. tally.py's polar_reference normalization).
    nx = math.sqrt(max(0.0, 1.0 - nz**2))
    return nx, 0.0, nz


@pytest.mark.parametrize("nz", [1.0, -1.0, 0.999, -0.999, 0.0])
def test_sampled_direction_is_unit_length_and_finite(nz, mock_rng):
    nx, ny, _ = _unit_reference_direction(nz)
    rng_state = _mock_rng_state(GENERIC_XI1, GENERIC_XI2)

    x, y, z = dist.sample_white_direction(nx, ny, nz, rng_state)

    assert math.isfinite(x)
    assert math.isfinite(y)
    assert math.isfinite(z)
    norm = math.sqrt(x**2 + y**2 + z**2)
    assert norm == pytest.approx(1.0, rel=0.0, abs=1e-10)


def test_negative_z_pole_does_not_divide_by_zero(mock_rng):
    # The exact regression case for the `nz != 1.0` -> `abs(nz) != 1.0` fix: a
    # reference direction of precisely (0, 0, -1). Pre-fix, this took the general
    # branch, computed B = sqrt(1 - (-1.0)**2) == 0.0, and divided by it.
    rng_state = _mock_rng_state(GENERIC_XI1, GENERIC_XI2)

    x, y, z = dist.sample_white_direction(0.0, 0.0, -1.0, rng_state)

    assert math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
    assert not any(math.isnan(v) for v in (x, y, z))
    norm = math.sqrt(x**2 + y**2 + z**2)
    assert norm == pytest.approx(1.0, rel=0.0, abs=1e-10)


def test_positive_and_negative_z_pole_agree_up_to_reflection(mock_rng):
    # Both poles should use the same (correct) special-case branch and therefore
    # produce a consistent, finite, unit-length result for the same rng draw -- not
    # one finite (nz=+1, already worked pre-fix) and one broken (nz=-1, the bug).
    rng_state_pos = _mock_rng_state(GENERIC_XI1, GENERIC_XI2)
    rng_state_neg = _mock_rng_state(GENERIC_XI1, GENERIC_XI2)

    pos = dist.sample_white_direction(0.0, 0.0, 1.0, rng_state_pos)
    neg = dist.sample_white_direction(0.0, 0.0, -1.0, rng_state_neg)

    for v in (*pos, *neg):
        assert math.isfinite(v)

    norm_pos = math.sqrt(sum(v**2 for v in pos))
    norm_neg = math.sqrt(sum(v**2 for v in neg))
    assert norm_pos == pytest.approx(1.0, rel=0.0, abs=1e-10)
    assert norm_neg == pytest.approx(1.0, rel=0.0, abs=1e-10)
