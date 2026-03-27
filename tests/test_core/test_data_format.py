"""Tests of data_format."""

import copy
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.data_format import PolymlpParamsSingle, PolymlpStructure

cwd = Path(__file__).parent


def test_polymlp_structure(structure_rocksalt):
    """Test for PolymlpStructure class."""
    st = structure_rocksalt
    assert st.axis.shape == (3, 3)
    assert st.positions.shape == (3, 8)
    np.testing.assert_equal(st.n_atoms, [4, 4])
    np.testing.assert_equal(st.types, [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_equal(st.elements, ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"])
    assert st.volume == pytest.approx(64.0)

    _ = PolymlpStructure(
        axis=st.axis,
        positions=st.positions,
        n_atoms=st.n_atoms,
        elements=st.elements,
        types=st.types,
        volume=64.0,
        supercell_matrix=np.eye(3),
        n_unitcells=1,
        comment="test",
        name="POSCAR-rocksalt",
        masses=[2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    )

    st_rev = copy.deepcopy(st)
    st_rev.types = [0, 0, 1, 1, 0, 0, 1, 1]
    st_rev.elements = ["Mg", "Mg", "O", "O", "Mg", "Mg", "O", "O"]
    st_rev.positions = st_rev.positions[:, np.array([0, 1, 4, 5, 2, 3, 6, 7])]
    st_rev = st_rev.reorder()

    np.testing.assert_allclose(st.axis, st_rev.axis)
    np.testing.assert_allclose(st.positions, st_rev.positions)
    np.testing.assert_equal(st.n_atoms, st_rev.n_atoms)
    np.testing.assert_equal(st.types, st_rev.types)
    np.testing.assert_equal(st.elements, st_rev.elements)

    np.testing.assert_allclose(st.composition(["Mg", "O"]), [0.5, 0.5])
    np.testing.assert_allclose(st.composition(["Sr", "Mg", "O"]), [0, 0.5, 0.5])


def test_PolymlpParamsSingle(params_MgO):
    """Test PolymlpParamsSingle."""
    params_single = params_MgO.params
    assert isinstance(params_single, PolymlpParamsSingle)
    assert params_single.n_type == 2
    assert tuple(params_single.elements) == ("Mg", "O")
    assert len(params_single.atomic_energy) == 2
    assert params_single.enable_spins is None
    assert len(params_single.regression_alpha) == 5
    assert len(params_single.alphas) == 5
    assert params_single.dataset_type == "vasp"
    assert params_single.include_force
    assert params_single.include_stress
    assert not params_single.print_memory

    assert params_single.type_indices is None
    assert params_single.type_full is None

    assert params_single.temperature == 300
    assert params_single.electron_property == "free_energy"
    assert isinstance(params_single.as_dict(), dict)


def test_PolymlpParamsSingle_spin(params_MgO):
    """Test PolymlpParamsSingle for spin configurations."""
    params_single = copy.deepcopy(params_MgO.params)
    params_single.enable_spins = (True, False)
    params_single._set_params_spins()
    assert isinstance(params_single, PolymlpParamsSingle)

    assert params_single.n_type == 3
    assert tuple(params_single.elements) == ("Mg", "Mg", "O")
    assert len(params_single.atomic_energy) == 3
    assert params_single.enable_spins == (True, True, False)

    params_ele = params_single.params_elements
    assert params_ele.n_type == 2
    assert tuple(params_ele.elements) == ("Mg", "O")
    assert len(params_ele.atomic_energy) == 2
    assert params_ele.enable_spins == (True, False)


def test_PolymlpModelParams_spin(params_MgO):
    """Test PolymlpModelParamsSingle for spin configurations."""
    params_single = copy.deepcopy(params_MgO.params)
    params_single.enable_spins = (True, False)
    model = params_single.model

    map_types = {0: [0, 1], 1: [2]}
    model.revise_params(map_types)
    assert len(model.pair_params_conditional) == 6
