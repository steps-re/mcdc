import numpy as np
import pytest

import mcdc

# ======================================================================================
# Regression coverage for Tally's polar_reference normalization
#
# mcdc/object_/tally.py normalizes a user-provided polar_reference vector with:
#
#     self.polar_reference = polar_reference_arr / np.linalg.norm(polar_reference_arr)
#
# Before the fix landed (mcdc-project/mcdc PR #470), this was an in-place divide:
#
#     self.polar_reference /= polar_reference / np.linalg.norm(polar_reference)
#
# self.polar_reference started as np.array([0.0, 0.0, 1.0]) (the default), so dividing
# it *in place* by the (different-shaped/valued) normalized user vector corrupted it
# component-wise instead of replacing it, producing values like [0, nan, inf] for some
# inputs rather than the normalized input vector.
#
# grepping test/unit/tally for "polar_reference" turned up zero hits before this file
# was added, so this behavior had no regression coverage even though the underlying
# bug is already fixed in current tally.py.
# ======================================================================================


def test_default_polar_reference_is_unit_z():
    tally = mcdc.Tally(scores=["flux"])
    np.testing.assert_allclose(tally.polar_reference, [0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "polar_reference, expected_normalized",
    [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 2.0, 0.0], [0.0, 1.0, 0.0]),
        ([3.0, 4.0, 0.0], [0.6, 0.8, 0.0]),
        ([1.0, 1.0, 1.0], [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]),
    ],
)
def test_polar_reference_is_normalized_not_corrupted(
    polar_reference, expected_normalized
):
    tally = mcdc.Tally(scores=["flux"], polar_reference=polar_reference)

    # The historical bug produced values like [0, nan, inf] here instead of the
    # normalized input vector, so explicitly guard against non-finite output as well
    # as checking the normalized value itself.
    assert np.all(np.isfinite(tally.polar_reference))
    np.testing.assert_allclose(
        tally.polar_reference, expected_normalized, rtol=0.0, atol=1e-12
    )
    assert np.linalg.norm(tally.polar_reference) == pytest.approx(1.0, abs=1e-12)


def test_polar_reference_input_array_not_mutated():
    # Guard against a regression that reintroduces in-place mutation of the caller's
    # own array via aliasing.
    polar_reference_input = np.array([2.0, 0.0, 0.0])
    original = polar_reference_input.copy()

    tally = mcdc.Tally(scores=["flux"], polar_reference=polar_reference_input)

    np.testing.assert_array_equal(polar_reference_input, original)
    np.testing.assert_allclose(tally.polar_reference, [1.0, 0.0, 0.0])
