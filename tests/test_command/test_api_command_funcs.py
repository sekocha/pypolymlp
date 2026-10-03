"""Test func_calc for APIs."""

import glob
import os
import shutil
from pathlib import Path

from pypolymlp.api.func_calc import (
    run_calculations,
    run_elastic,
    run_eos,
    run_geometry_optimization,
    run_gsfe,
    run_phonon,
)
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.api.run_polymlp_calc import _parse_args_pypolymlp_calc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "POSCAR.fcc.Al"
pot = path_file + "polymlp.yaml.gtinv.Al"


calc = PypolymlpCalc(pot=pot, verbose=True)
calc.load_poscars(poscar)
unitcell = calc.first_structure


def test_run_functions():
    """Test func_calc."""
    args = _parse_args_pypolymlp_calc([])
    args.poscar = args.poscars = poscar

    run_geometry_optimization(args, calc)
    run_geometry_optimization(args, calc, structure=unitcell)

    run_elastic(args, calc)
    run_elastic(args, calc, structure=unitcell)

    run_eos(args, calc)
    run_eos(args, calc, structure=unitcell)

    run_phonon(args, calc)

    args.disp1 = (1, 0, 0)
    args.disp2 = (0, 1, 0)
    args.slip = (0, 0, 1)
    run_gsfe(args, calc)

    os.remove("POSCAR_eqm")
    os.remove("fc2.hdf5")
    os.remove("gsfe.dat")
    os.remove("polymlp_elastic.yaml")
    os.remove("polymlp_eos.yaml")
    os.remove("polymlp_phonon.yaml")
    shutil.rmtree("polymlp_phonon_qha")
    shutil.rmtree("poscars")
    for f in glob.glob("phonon*"):
        os.remove(f)


def test_run_calculations():
    """Test func_calc."""
    args = _parse_args_pypolymlp_calc(["--features", "--pot", pot])
    args.poscar = args.poscars = poscar
    run_calculations(args, calc)
    os.remove("features.npy")


# fc2.hdf5
# features.npy
# files/
# gsfe.dat
# phonon_mesh_qpoints.txt
# phonon_thermal_properties.yaml
# phonon_total_dos.dat
# polymlp_elastic.yaml
# polymlp_eos.yaml
# polymlp_phonon_qha/
# polymlp_phonon.yaml
