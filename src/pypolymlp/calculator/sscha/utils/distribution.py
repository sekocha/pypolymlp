#!/usr/bin/env python
import argparse
import os
import signal

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_io import Restart  # save_cell,
from pypolymlp.utils.vasp_utils import write_poscar_file

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
        default=100,
        help="Number of sample supercells",
    )
    args = parser.parse_args()

    res = Restart(args.yaml, fc2hdf5=args.fc2)

    pot = res.polymlp if args.pot is None else args.pot
    prop = Properties(pot=pot)

    print("Restart: SSCHA distribution calculation")
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

    disps = ph_real.displacements.transpose((0, 2, 1))
    forces = ph_real.forces.transpose((0, 2, 1))

    energies = ph_real.full_potentials
    e0 = ph_real.static_potential
    st_dicts = ph_real.supercells

    np.save("sscha_disps.npy", disps)
    np.save("sscha_forces.npy", forces)
    print("sscha_disps.npy and sscha_forces.npy are generated.")
    print("- shape:", forces.shape)

    np.save("sscha_energies.npy", np.array(energies))
    f = open("sscha_static_energy.dat", "w")
    print(e0, file=f)
    f.close()
    print("sscha_energies.npy and sscha_static_energy.dat are generated.")
    print("- shape:", len(energies))

    os.makedirs("sscha_poscars", exist_ok=True)
    for i, st in enumerate(st_dicts, 1):
        filename = "sscha_poscars/POSCAR-" + str(i).zfill(4)
        write_poscar_file(st, filename=filename)
    print("sscha_poscars/POSCAR* are generated.")

    """
    f = open('sscha_cells.yaml', 'w')
    save_cell(res.unitcell, tag='unitcell', fstream=f)
    save_cell(ph_real.supercell, tag='supercell', fstream=f)
    f.close()
    """
