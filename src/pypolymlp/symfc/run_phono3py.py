#!/usr/bin/env python
import argparse
import signal

import phono3py

if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar", type=str, default=None, help="poscar")
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        default=[19, 19, 19],
        help="k-mesh used for phono3py calculation",
    )
    args = parser.parse_args()

    ph3 = phono3py.load(
        unitcell_filename=args.poscar,
        supercell_matrix=args.supercell,
        primitive_matrix="auto",
        log_level=1,
    )
    ph3.mesh_numbers = args.mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=range(0, 1001, 10), write_kappa=True)
#    # Conductivity_RTA object
#    print(ph3.thermal_conductivity.kappa)
