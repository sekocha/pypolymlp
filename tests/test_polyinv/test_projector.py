"""Tests of building projector."""

from pypolymlp.polyinv.projector import build_projector


def test_build_projector_2():
    """Test build_projector."""
    core, lm_indices = build_projector([20, 20])
    assert core.shape == (41, 41)
    assert lm_indices.shape == (41, 2, 2)


def test_build_projector_3():
    """Test build_projector."""
    core, lm_indices = build_projector([2, 2, 2])
    assert core.shape == (19, 19)
    assert lm_indices.shape == (19, 3, 2)


def test_build_projector_4():
    """Test build_projector."""
    core, lm_indices = build_projector([1, 1, 2, 2])
    assert core.shape == (37, 37)
    assert lm_indices.shape == (37, 4, 2)


def test_build_projector_5():
    """Test build_projector."""
    core, lm_indices = build_projector([1, 1, 1, 2, 2])
    assert core.shape == (105, 105)
    assert lm_indices.shape == (105, 5, 2)


def test_build_projector_6():
    """Test build_projector."""
    core, lm_indices = build_projector([1, 1, 1, 1, 1, 1])
    assert core.shape == (141, 141)
    assert lm_indices.shape == (141, 6, 2)
