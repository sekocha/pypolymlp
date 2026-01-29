"""Tests of PolymlpEvalAccuracy."""

from pathlib import Path

import pytest

from pypolymlp.mlp_dev.core.eval_accuracy import PolymlpEvalAccuracy

cwd = Path(__file__).parent


def test_accuracy(regdata_mp_149, mlp_mp_149):
    """Test for PolymlpEvalAccuracy."""
    _, datasets = regdata_mp_149
    polymlp = PolymlpEvalAccuracy(mlp_mp_149)
    errors = polymlp.compute_error(datasets, log_energy=False)
    tag = "Train_Data_from_files"
    assert errors[tag]["energy"] == pytest.approx(5.84708583252534e-06)
    assert errors[tag]["force"] == pytest.approx(0.0028245295894005276)
