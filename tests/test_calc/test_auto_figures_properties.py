"""Tests of functions to draw figures for properties."""

import shutil

import numpy as np

from pypolymlp.calculator.auto.figures_properties import (
    plot_energy_distribution,
    plot_eos,
    plot_eos_separate,
    plot_phonon,
    plot_prototype_prediction,
    plot_qha,
)

system = "Ag"
pot_id = "polymlp-00001"


def test_plot_prototype_prediction():
    """Test plot_prototype_prediction."""
    data = np.array([[0.2, 0.25, "Str1"], [0.1, 0.15, "Str2"], [0.2, 0.35, "Str3"]])
    plot_prototype_prediction(data, system, pot_id, path_output="tmp")
    shutil.rmtree("tmp")


def test_plot_eos(prototypes_Ag):
    """Test plot_eos."""
    plot_eos(prototypes_Ag, system, pot_id, path_output="tmp")
    shutil.rmtree("tmp")


def test_plot_eos_separate(prototypes_Ag):
    """Test plot_eos_separate."""
    plot_eos_separate(prototypes_Ag, system, pot_id, path_output="tmp")
    shutil.rmtree("tmp")


def test_plot_phonon(prototypes_Ag):
    """Test plot_phonon."""
    plot_phonon(prototypes_Ag, system, pot_id, path_output="tmp")
    shutil.rmtree("tmp")


def test_plot_qha(prototypes_Ag):
    """Test plot_qha."""
    plot_qha(
        prototypes_Ag, system, pot_id, target="thermal_expansion", path_output="tmp"
    )
    plot_qha(prototypes_Ag, system, pot_id, target="bulk_modulus", path_output="tmp")
    shutil.rmtree("tmp")


def test_plot_energy_distribution():
    """Test plot_energy_distribution."""
    data_train = np.array([[0.2, 0.25], [0.1, 0.15], [0.2, 0.35]])
    data_test = np.array([[0.2, 0.25], [0.1, 0.15], [0.2, 0.35]])
    plot_energy_distribution(
        data_train,
        data_test,
        system,
        pot_id,
        path_output="tmp",
    )
    shutil.rmtree("tmp")
