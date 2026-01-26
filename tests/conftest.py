"""Pytest conftest.py."""

from __future__ import annotations

import copy
from pathlib import Path

import phono3py
import pytest
from phono3py.api_phono3py import Phono3py

from pypolymlp.core.interface_vasp import PolymlpStructure, Poscar
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_dev.core.api_mlpdev import PolymlpDevCore
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def pytest_addoption(parser):
    """Add command option to pytest."""
    parser.addoption(
        "--runbig", action="store_true", default=False, help="run big tests"
    )


def pytest_configure(config):
    """Set up marker big."""
    config.addinivalue_line("markers", "big: mark test as big to run")


def pytest_collection_modifyitems(config, items):
    """Add mechanism to run with --runbig."""
    if config.getoption("--runbig"):
        # --runbig given in cli: do not skip slow tests
        return
    skip_big = pytest.mark.skip(reason="need --runbig option to run")
    for item in items:
        if "big" in item.keywords:
            item.add_marker(skip_big)


@pytest.fixture(scope="session")
def structure_rocksalt() -> PolymlpStructure:
    """Return rocksalt structure."""
    return Poscar(path_file + "/POSCAR-rocksalt").structure


@pytest.fixture(scope="session")
def phono3py_mp_149() -> Phono3py:
    """Return phono3py data instance."""
    yaml = path_file + "/phonopy_training_dataset.yaml.xz"
    ph3 = phono3py.load(yaml, produce_fc=False, log_level=1)
    return ph3


@pytest.fixture(scope="session")
def regdata_mp_149(phono3py_mp_149):
    """Return regression data."""
    pypolymlp = Pypolymlp()
    infile = path_file + "/polymlp.in.phono3py.Si"
    pypolymlp.load_parameter_file(infile, prefix=path_file)
    params = pypolymlp.parameters
    data = pypolymlp.train
    return (params, data)


@pytest.fixture(scope="session")
def dataxy_mp_149(regdata_mp_149):
    """Return regression data."""
    params, datasets_ = regdata_mp_149
    core = PolymlpDevCore(params, use_gradient=False)
    data_xy = core.calc_xy(datasets_)
    return data_xy


@pytest.fixture(scope="session")
def dataxy_xtx_xty_mp_149(regdata_mp_149):
    """Return regression data."""
    params, datasets = regdata_mp_149
    datasets_ = copy.deepcopy(datasets)
    core = PolymlpDevCore(params, use_gradient=False)
    data_xy = core.calc_xtx_xty(datasets_)
    return data_xy


@pytest.fixture(scope="session")
def mlp_mp_149(regdata_mp_149):
    """Return mlp data."""
    params, _ = regdata_mp_149
    pypolymlp = Pypolymlp()
    mlp = path_file + "/polymlp.yaml.Si"
    pypolymlp.load_mlp(mlp)
    pypolymlp.summary.params = params
    return pypolymlp.summary


@pytest.fixture(scope="session")
def params_MgO():
    """Return parameters for MgO."""
    infile = path_file + "/polymlp.in"
    parser = ParamsParser(infile, prefix_data_location=path_file, parse_dft=False)
    return parser.params
