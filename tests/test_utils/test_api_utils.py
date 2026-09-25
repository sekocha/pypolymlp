"""Tests of API for using utilities."""

import glob
import os
import shutil
from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_utils import PypolymlpUtils

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/"


def test_compress():
    """Test for compressing vasprun.xml files."""
    polymlp = PypolymlpUtils(verbose=True)
    files = [path_file + "vasprun-00001-MgO.xml", path_file + "vasprun-00002-MgO.xml"]
    polymlp.compress_vaspruns(files, n_jobs=1)
    os.remove(path_file + "vasprun-00001-MgO.xml.polymlp")
    os.remove(path_file + "vasprun-00002-MgO.xml.polymlp")


def test_electron_properties():
    """Test for computing electronic excitation from vasprun.xml files."""
    polymlp = PypolymlpUtils(verbose=True)
    files = [path_file + "vasprun-00001-Ti-full.xml"]
    polymlp.compute_electron_properties_from_vaspruns(
        files, temp_max=100, temp_step=50, n_jobs=1
    )
    os.remove(path_file + "electron-00001-Ti-full.yaml")


def test_estimate_computational_cost1():
    """Test for estimating computational costs of polymlps."""
    polymlp = PypolymlpUtils(verbose=True)
    pot = path_file + "polymlp.yaml.MgO"
    polymlp.estimate_polymlp_comp_cost(pot=pot, n_calc=3)
    os.remove("polymlp_cost.yaml")


def test_estimate_computational_cost2():
    """Test for estimating computational costs of polymlps."""
    polymlp = PypolymlpUtils(verbose=True)
    pot = path_file + "polymlp.yaml.MgO"
    poscar = path_file + "POSCAR-rocksalt"
    polymlp.estimate_polymlp_comp_cost(pot=pot, poscar=poscar, n_calc=3)
    os.remove("polymlp_cost.yaml")


def test_estimate_computational_cost3():
    """Test for estimating computational costs of polymlps."""
    polymlp = PypolymlpUtils(verbose=True)
    path_pot = [
        path_file + "mlps_count_times/polymlp-001",
        path_file + "mlps_count_times/polymlp-002",
    ]
    polymlp.estimate_polymlp_comp_cost(path_pot=path_pot, n_calc=3)
    for path in path_pot:
        os.remove(path + "/polymlp_cost.yaml")


def test_find_optimal_mlps():
    """Test for find optimal polymlps."""
    polymlp = PypolymlpUtils(verbose=True)
    polymlp_dirs = glob.glob(path_file + "/grid-Ti/polymlp-*")
    key = "test_vasprun_low"
    polymlp.find_optimal_mlps(polymlp_dirs, key)

    os.remove("polymlp_summary_all.yaml")
    os.remove("polymlp_summary_convex.yaml")


def test_spglib():
    """Test spglib functions."""
    polymlp = PypolymlpUtils(verbose=True)
    poscar = path_file + "POSCAR-rocksalt"
    polymlp.init_symmetry(poscar=poscar, symprec=1e-5)
    structure = polymlp.refine_cell()
    assert len(structure.elements) == 8
    polymlp.print_poscar(structure)
    polymlp.write_poscar_file(structure, filename="tmp")
    os.remove("tmp")
    spg = polymlp.get_spacegroup()
    assert spg == "Fm-3m (225)"


def test_auto_divide():
    """Test divide_dataset."""
    polymlp = PypolymlpUtils(verbose=True)
    files = [path_file + "vasprun-00001-MgO.xml", path_file + "vasprun-00002-MgO.xml"]
    elements = ["Mg", "O"]
    polymlp.divide_dataset(files, elements, n_divide=6)
    shutil.rmtree("polymlp_datasets")


def test_generate_kim_model():
    """Test generate_kim_model."""
    user_id = "b3113743-4f85-48da-86e1-85acf6bb3388"
    author = "Seko"
    citation1 = {
        "article-number": "{011101}",
        "author": "Seko, Atsuto",
    }
    citations = [citation1]

    polymlp = PypolymlpUtils()
    polymlp.generate_kim_model(
        path_file + "polymlp.yaml.MgO",
        author=author,
        performance_level=1,
        project_id=123,
        project_version=2,
        model_driver="Polymlp__MD_367995833009_000",
        content_origin="pypolymlp",
        contributor_id=user_id,
        developer=[user_id],
        maintainer_id=user_id,
        citations=citations,
    )
    for d in glob.glob("Polymlp_Seko_*_MgO__MO_000000000123_002"):
        shutil.rmtree(d)


def test_enumerate_models():
    """Test enumerate_models."""
    polymlp = PypolymlpUtils()
    polymlp.enumerate_models(elements=["Ag", "Au"], path="tmp")
    shutil.rmtree("tmp")
    polymlp.enumerate_models(elements=["Be"], path="tmp")
    shutil.rmtree("tmp")
    polymlp.enumerate_models(elements=["Be"], path="tmp", hybrid=True)
    shutil.rmtree("tmp")

    polymlp.enumerate_models(
        elements=["Be"],
        cutoffs=None,
        nums_gaussians=None,
        model_types=(2, 3, 4),
        maxps=(2,),
        gtinv=True,
        gtinv_order_ub=3,
        gtinv_maxl_ub=(16, 12),
        gtinv_maxl_int=(8, 4),
        include_force=True,
        include_stress=True,
        regression_alpha=(-4, 1, 6),
        path="tmp",
        hybrid=True,
    )
    shutil.rmtree("tmp")


def test_supercell(structure_rocksalt):
    """Test for supercell."""
    polymlp = PypolymlpUtils()
    sup = polymlp.generate_supercell(
        structure=structure_rocksalt,
        supercell_matrix=(1, 2, 3),
    )
    assert sup.volume == pytest.approx(384.0)
    assert sum(sup.n_atoms) == 48
    assert len(sup.elements) == 48
    assert len(sup.types) == 48
    assert sup.positions.shape == (3, 48)

    sup = polymlp.generate_supercell(
        structure=structure_rocksalt,
        supercell_matrix=[[1, 0, 0], [0, 1, -1], [0, 1, 1]],
    )

    assert sup.volume == pytest.approx(128.0)
    assert sum(sup.n_atoms) == 16
    assert len(sup.elements) == 16
    assert len(sup.types) == 16
    assert sup.positions.shape == (3, 16)


def test_supercell_three_directions(structure_rocksalt):
    """Test for supercell functions using three directions."""
    polymlp = PypolymlpUtils()
    sup = polymlp.generate_supercell_three_directions(
        structure=structure_rocksalt,
        direction1=(1, 0, 0),
        direction2=(0, 1, -1),
        direction3=(0, 1, 1),
        n_layers=3,
    )
    assert sup.volume == pytest.approx(384.0)
    assert sum(sup.n_atoms) == 48
    assert len(sup.elements) == 48
    assert len(sup.types) == 48
    assert sup.positions.shape == (3, 48)


def test_get_slab(structure_rocksalt):
    """Test for get_slab."""
    polymlp = PypolymlpUtils()
    sup = polymlp.generate_slab_model(
        structure=structure_rocksalt,
        direction1=(1, -1, 0),
        direction2=(1, 1, -2),
        direction3=(1, 1, 1),
        n_layers=3,
        vacuum_width=15.0,
        end_frac=None,
    )
    assert sup.volume == pytest.approx(1152.0)
    assert sup.axis[0, 0] == pytest.approx(5.656854249492381)
    assert sup.axis[1, 1] == pytest.approx(9.797958971132712)
    assert sup.axis[2, 2] == pytest.approx(35.78460969082653)
    assert sum(sup.n_atoms) == 144
    assert len(sup.elements) == 144
    assert len(sup.types) == 144
    assert sup.positions.shape == (3, 144)

    sup = polymlp.generate_slab_model(
        structure=structure_rocksalt,
        direction1=(1, -1, 0),
        direction2=(1, 1, -2),
        direction3=(1, 1, 1),
        n_layers=3,
        vacuum_width=15.0,
        end_frac=0.5,
    )
    assert sup.volume == pytest.approx(1152.0)
    assert sup.axis[0, 0] == pytest.approx(5.656854249492381)
    assert sup.axis[1, 1] == pytest.approx(9.797958971132712)
    assert sup.axis[2, 2] == pytest.approx(35.78460969082653)
    assert sum(sup.n_atoms) == 152
