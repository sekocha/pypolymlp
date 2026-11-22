"""Tests of command line calculations"""

import os
import shutil
import subprocess
from pathlib import Path

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "POSCAR.fcc.Ag"
pot = path_file + "polymlp.yaml.pair.Ag"
common_str = " --pot " + pot + " --poscar " + poscar + " "


def test_sscha():
    """Test command lines."""
    cmd = (
        "pypolymlp-sscha --temp 300 --supercell 2 2 2 "
        "--sscha_max_iter 2 --n_samples 10 10 --mesh 3 3 3 "
        "--tol 0.05 --mixing 0.5" + common_str
    )
    subprocess.run(cmd.split(), check=True)

    cmd = (
        "pypolymlp-sscha --temp 300 --supercell 2 0 0 0 2 0 0 0 2 "
        "--sscha_max_iter 2 --n_samples 10 10 --mesh 3 3 3 "
        "--tol 0.05 --mixing 0.5" + common_str
    )
    subprocess.run(cmd.split(), check=True)

    cmd = (
        "pypolymlp-sscha --temp 300 --supercell 2 0 0 0 2 0 1 0 2 "
        "--sscha_max_iter 2 --n_samples 10 10 --mesh 3 3 3 "
        "--tol 0.05 --mixing 0.5" + common_str
    )
    subprocess.run(cmd.split(), check=True)


def test_sscha_options():
    """Test command lines."""
    cmd = (
        "pypolymlp-sscha --supercell 2 2 2 "
        "--temp_min 100 --temp_max 150 --temp_step 100 --ascending_temp "
        "--init random --cutoff_fc2 8 --disable_precondition "
        "--n_samples 10 10 --sscha_max_iter 2 --mesh 3 3 3 "
        "--symfc_symprec 1e-3 --symfc_batch_size 10 --symfc_disable_mkl "
        "--symfc_use_gradient_solver --save_pdos "
        "--tol 0.05 --mixing 0.5" + common_str
    )
    subprocess.run(cmd.split(), check=True)

    shutil.rmtree("sscha")


def test_sscha_go():
    """Test command lines."""
    cmd = (
        "pypolymlp-sscha --temp 300 --supercell 2 2 2 "
        "--geometry_optimization --gtol 0.1 --scale_axis 10 --maxiter 1 "
        "--sscha_max_iter 2 --n_samples 10 10 --mesh 3 3 3 "
        "--tol 0.05 --mixing 0.5" + common_str
    )
    subprocess.run(cmd.split(), check=True)
    shutil.rmtree("sscha")
    os.remove("POSCAR_eqm")


def test_sscha_eos():
    """Test command lines."""
    cmd = (
        "pypolymlp-sscha --temp 300 --supercell 1 1 1 "
        "--eos "
        "--sscha_max_iter 1 --n_samples 10 10 --mesh 1 1 1 "
        "--tol 0.05 --mixing 0.5" + common_str
    )
    subprocess.run(cmd.split(), check=True)
    shutil.rmtree("sscha")
    os.remove("polymlp_eos.yaml")


def test_sscha_elastic():
    """Test command lines."""
    cmd = (
        "pypolymlp-sscha --temp 300 --supercell 1 1 1 "
        "--elastic --gtol 0.1 --scale_axis 10 --maxiter 1 "
        "--sscha_max_iter 1 --n_samples 10 10 --mesh 1 1 1 "
        "--tol 0.05 --mixing 0.5" + common_str
    )
    subprocess.run(cmd.split(), check=True)
    shutil.rmtree("sscha")
    os.remove("polymlp_elastic_sscha.yaml")
    os.remove("POSCAR_eqm")


# TODO
def test_sscha_gsfe():
    """Test command lines."""
