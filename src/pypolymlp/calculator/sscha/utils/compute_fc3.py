#!/usr/bin/env python
import argparse
import signal

import phono3py
from phono3py.file_IO import write_fc3_to_hdf5

from pypolymlp.calculator.fc import PolymlpFC
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_utils import Restart

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
    parser.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        default=[19, 19, 19],
        help="k-mesh used for phono3py calculation",
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

    print("Running phono3py to compute real self energy.")
    ph3 = phono3py.load(
        unitcell=res.unitcell_phonopy,
        supercell_matrix=res.supercell_matrix,
        primitive_matrix="auto",
        log_level=1,
    )
    ph3.mesh_numbers = args.mesh
    ph3.init_phph_interaction()
    ph3.run_phonon_solver()
    freq, _, bzgrid = ph3.get_phonon_data()
    # print(bzgrid)
    print(ph3._bz_grid.grg2bzg)

    frequency_points, deltas = ph3.run_real_self_energy(
        grid_points=range(freq.shape[0]),
        temperatures=[res.temperature],
        frequency_points_at_bands=True,
        # write_hdf5=True,
    )

    frequency_shifted = freq - deltas[0][0]
    frequency_shifted = frequency_shifted[ph3._bz_grid.grg2bzg]

#    from phono3py.other.kaccum import KappaDOSTHM
