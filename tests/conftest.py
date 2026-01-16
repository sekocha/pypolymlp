"""Pytest conftest.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pypolymlp.core.interface_vasp import PolymlpStructure, Poscar

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
