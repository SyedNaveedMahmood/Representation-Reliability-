import numpy as np

from representation_reliability.interventions.truth_coordinate import (
    coordinate_transfer_delta,
    full_residual_patch_delta,
    normalized_direction,
    random_unit_direction,
)


def test_coordinate_transfer_changes_only_probe_coordinate():
    base = np.array([1.0, 2.0, 3.0])
    source = np.array([4.0, -1.0, 5.0])
    u = normalized_direction(np.array([1.0, 1.0, 0.0]))
    delta = coordinate_transfer_delta(base, source, u, alpha=1.0)
    edited = base + delta

    assert np.isclose(np.dot(edited, u), np.dot(source, u))
    residual = edited - source
    assert np.isclose(np.dot(residual, u), 0.0)

    base_orth = base - np.dot(base, u) * u
    edited_orth = edited - np.dot(edited, u) * u
    np.testing.assert_allclose(edited_orth, base_orth, atol=1e-12)


def test_coordinate_transfer_alpha_is_exact_fraction():
    base = np.array([1.0, 0.0])
    source = np.array([5.0, 3.0])
    u = np.array([1.0, 0.0])
    for alpha in (-1.0, 0.0, 0.25, 1.0, 2.0):
        delta = coordinate_transfer_delta(base, source, u, alpha)
        got = np.dot(base + delta, u) - np.dot(base, u)
        expected = alpha * (np.dot(source, u) - np.dot(base, u))
        assert np.isclose(got, expected)


def test_random_orthogonal_direction_is_unit_and_orthogonal():
    u = normalized_direction(np.array([1.0, 2.0, 3.0, 4.0]))
    r1 = random_unit_direction(4, 7, orthogonal_to=u)
    r2 = random_unit_direction(4, 7, orthogonal_to=u)
    np.testing.assert_allclose(r1, r2)
    assert np.isclose(np.linalg.norm(r1), 1.0)
    assert abs(float(np.dot(r1, u))) < 1e-12


def test_full_residual_patch_alpha_one_copies_source():
    base = np.array([1.0, 2.0, 3.0])
    source = np.array([-4.0, 5.0, 6.0])
    delta = full_residual_patch_delta(base, source, 1.0)
    np.testing.assert_allclose(base + delta, source)


def test_random_controls_match_truth_delta_norm_sample_by_sample():
    u = normalized_direction(np.array([1.0, 2.0, -1.0, 0.5]))
    random = random_unit_direction(4, 11)
    orthogonal = random_unit_direction(4, 12, orthogonal_to=u)
    sources = (
        (np.array([1.0, 0.0, 2.0, -1.0]), np.array([-2.0, 3.0, 1.0, 4.0])),
        (np.array([0.5, -2.0, 0.0, 1.0]), np.array([3.0, 2.0, -4.0, 0.0])),
    )
    for base, source in sources:
        for alpha in (-0.5, 0.5, 1.5):
            truth = coordinate_transfer_delta(base, source, u, alpha)
            q_gap = float(np.dot(source - base, u))
            random_delta = alpha * abs(q_gap) * random
            orthogonal_delta = alpha * abs(q_gap) * orthogonal
            assert np.isclose(np.linalg.norm(random_delta), np.linalg.norm(truth))
            assert np.isclose(np.linalg.norm(orthogonal_delta), np.linalg.norm(truth))
