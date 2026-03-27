"""Tests of functions to draw figures for formation energies."""

import shutil
from pathlib import Path

import numpy as np

from pypolymlp.calculator.auto.figures_formation import plot_binary_formation_energies

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_plot_binary_formation_energies():
    """Test plot_binary_formation_energies."""
    data_all = np.array(
        [
            [0.8, 0.2, -0.03495486],
            [0.5, 0.5, -0.04193504],
            [0.75, 0.25, -0.04421533],
            [0.75, 0.25, -0.04305931],
            [0.33333333, 0.66666667, -0.04163314],
            [0.42857143, 0.57142857, -0.03554862],
            [0.25, 0.75, -0.04123442],
            [0.33333333, 0.66666667, -0.01564521],
            [0.5, 0.5, -0.03767163],
            [0.75, 0.25, -0.04349769],
            [0.5, 0.5, -0.06030366],
        ]
    )
    data_convex = np.array(
        [
            [0.8, 0.2, -0.03495486],
            [0.75, 0.25, -0.04421533],
            [0.5, 0.5, -0.03767163],
            [0.25, 0.75, -0.04123442],
        ]
    )
    plot_binary_formation_energies(
        system="Ag-Au",
        pot_id="polymlp-00001",
        data_dft_all=data_all,
        data_dft_convex=data_convex,
        data_mlp_all=data_all,
        data_mlp_convex=data_convex,
        data_mlp_go_all=data_all,
        data_mlp_go_convex=data_convex,
        path_output="./tmp",
    )
    plot_binary_formation_energies(
        system="Ag-Au",
        pot_id="polymlp-00001",
        data_dft_all=data_all,
        data_dft_convex=data_convex,
        data_mlp_all=data_all,
        data_mlp_convex=data_convex,
        path_output="./tmp",
        filename_suffix="second",
    )
    shutil.rmtree("tmp")
