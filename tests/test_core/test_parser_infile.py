"""Tests of parser_infile.py."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.parser_infile import InputParser

cwd = Path(__file__).parent


def test_input_parser():
    """Test InputParser class."""
    filename = str(cwd) + "/../files/polymlp.in.infiletest"
    parser = InputParser(filename)
    strings = parser.dataset_strings
    strings_true = [
        ["data", "dataset0/vasprun-*.xml.polymlp"],
        ["data", "dataset1/vasprun-*.xml.polymlp", "False", "0.1"],
        ["train_data", "dataset2/vasprun-*.xml.polymlp"],
        ["train_data", "dataset3/vasprun-*.xml.polymlp", "False", "0.1"],
        ["test_data", "dataset4/vasprun-*.xml.polymlp"],
        ["test_data", "dataset5/vasprun-*.xml.polymlp", "False", "0.1"],
        ["data_md", "dataset6/vasprun.xml"],
        ["data_md", "dataset7/vasprun.xml", "False", "0.1"],
    ]
    np.testing.assert_equal(strings, strings_true)

    distance = parser.distance
    np.testing.assert_allclose(distance[("Mg", "Mg")], [1.0, 3.0])
    np.testing.assert_allclose(distance[("O", "O")], [1.0, 2.0])
    np.testing.assert_allclose(distance[("Mg", "O")], [2.0, 3.0, 4.0])

    assert parser.get_params("n_type", dtype=int) == 2
    assert parser.get_params("feature_type", dtype=str) == "gtinv"
    assert parser.get_params("cutoff", dtype=float) == pytest.approx(8.0)
    assert parser.get_params("include_force", dtype=bool) == True
    assert parser.get_params("gtinv_maxl", size=2, dtype=int) == [4, 4]

    alphas = parser.get_sequence("reg_alpha_params")
    np.testing.assert_allclose(alphas, [-3.0, -2.0, -1.0, 0.0, 1.0])
