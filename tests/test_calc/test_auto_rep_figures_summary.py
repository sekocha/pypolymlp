"""Tests of functions to draw figures for summary."""

import shutil

import numpy as np

from pypolymlp.calculator.auto_repository.figures_summary import (
    plot_eqm_properties,
    plot_mlp_distribution,
)

system = "Ag"


def test_plot_mlp_distribution():
    """Test plot_mlp_distribution."""
    summary_all = np.array([[0.2, 0.1, 0.25], [0.1, 0.15, 0.02], [0.2, 0.35, 0.1]])
    summary_convex = summary_all
    plot_mlp_distribution(summary_all, summary_convex, system, path_output="tmp")
    shutil.rmtree("tmp")


def test_plot_eqm_properties(prototypes_Ag):
    """Test plot_eqm_properties."""
    prototype_data = [prototypes_Ag, prototypes_Ag]
    times = np.array([0.01, 0.02])
    plot_eqm_properties(prototype_data, times, system, path_output="tmp")
    shutil.rmtree("tmp")
