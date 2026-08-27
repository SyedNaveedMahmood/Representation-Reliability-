import numpy as np

from representation_reliability.runners.e00c_followup import signed_cosine


def test_signed_cosine_preserves_orientation():
    a = np.array([1.0, 0.0, 0.0])
    assert signed_cosine(a, a) == 1.0
    assert signed_cosine(a, -a) == -1.0


def test_rms_positive_scaling_does_not_preserve_cross_example_ranking():
    # A positive, sample-specific denominator preserves each score's sign but
    # can reverse ranking across examples. This is why raw-residual cosine is
    # not an exact readout-geometry diagnostic under RMSNorm.
    numerators = np.array([2.0, 3.0])
    rms = np.array([1.0, 3.0])
    normalized_scores = numerators / rms
    assert numerators[1] > numerators[0]
    assert normalized_scores[1] < normalized_scores[0]


def test_signed_cosine_rejects_zero_direction():
    try:
        signed_cosine(np.zeros(3), np.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("zero direction should be rejected")
