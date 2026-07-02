"""Tests of SSCHA calculations using Lammps API."""

import os
import shutil
from pathlib import Path

from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"
poscar = path_file + "Ti-Al/POSCAR-Al"


def test_sscha_Al(property_mlp_Ti_Al):
    """Test SSCHA calculations from polymlp using API."""
    sscha = PypolymlpSSCHA(verbose=True)
    sscha.load_poscar(poscar, (2, 2, 2))
    sscha.set_polymlp(properties=property_mlp_Ti_Al)
    sscha._pot = "tmp"
    sscha.run(temp=700, tol=0.03, mixing=0.5, path="tmp", use_mkl=False)
    shutil.rmtree("tmp")


def test_sscha_geometry_opt(property_mlp_Ti_Al):
    """Test SSCHA Geometry optimization."""
    sscha = PypolymlpSSCHA(verbose=True)
    sscha.load_poscar(poscar, (2, 2, 2))
    sscha.set_polymlp(properties=property_mlp_Ti_Al)
    sscha._pot = "tmp"

    sscha.run_geometry_optimization(
        temp=700,
        tol=0.03,
        mixing=0.5,
        use_mkl=False,
        gtol=1e-0,
        go_maxiter=2,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        pressure=0.01,
    )
    shutil.rmtree("sscha")
    os.remove("POSCAR_eqm")


def test_sscha_elastic(property_mlp_Ti_Al):
    """Test SSCHA elastic constant calculation."""
    sscha = PypolymlpSSCHA(verbose=True)
    sscha.load_poscar(poscar, (2, 2, 2))
    sscha.set_polymlp(properties=property_mlp_Ti_Al)
    sscha._pot = "tmp"

    sscha.run_elastic(
        temp=300,
        tol=0.05,
        mixing=0.95,
        use_mkl=False,
        gtol=1e-1,
    )
    shutil.rmtree("sscha")
    os.remove("POSCAR_eqm")
    os.remove("polymlp_elastic_sscha.yaml")
