#!/usr/bin/env python
import argparse
import signal

from phono3py.file_IO import write_fc3_to_hdf5

from pypolymlp.calculator.fc import PolymlpFC
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_io import Restart

if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        type=str,
        default="sscha_results.yaml",
        help="sscha_results.yaml file to be parsed.",
    )
    parser.add_argument(
        "--fc2",
        type=str,
        default="fc2.hdf5",
        help="fc2.hdf5 file to be parsed.",
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=None,
        help="polymlp.lammps file",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=10000,
        help="Number of sample supercells",
    )
    args = parser.parse_args()

    res = Restart(args.yaml, fc2hdf5=args.fc2)

    pot = res.polymlp if args.pot is None else args.pot
    prop = Properties(pot=pot)

    print("Restart: SSCHA FC3 calculation")
    print("  yaml:        ", args.yaml)
    print("  fc2 :        ", args.fc2)
    print("  mlp :        ", pot)
    print("  n_structures:", args.n_samples)
    print("  temperature :", res.temperature)

    ph_real = HarmonicReal(
        res.supercell,
        prop,
        n_unitcells=res.n_unitcells,
        fc2=res.force_constants,
    )
    ph_real.run(t=res.temperature, n_samples=args.n_samples)
    """Codes up to here are totally the same as used in distribution.py"""

    polyfc = PolymlpFC(supercell=res.supercell_phonopy)
    polyfc.run(disps=ph_real.displacements, forces=ph_real.forces, write_fc=False)

    print("Writing fc3.hdf5")
    write_fc3_to_hdf5(polyfc.fc3)
