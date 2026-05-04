"""Command lines for performing molecular dynamics."""

import argparse
import os
import shutil
import signal

import numpy as np

from pypolymlp.api.pypolymlp_md import PypolymlpMD
from pypolymlp.core.utils import print_credit


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pot", nargs="*", type=str, default="polymlp.yaml", help="polymlp file."
    )
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        default="POSCAR",
        help="Initial structure.",
    )
    parser.add_argument(
        "--supercell_size",
        type=int,
        nargs=3,
        default=(1, 1, 1),
        help="Diagonal supercell size.",
    )
    parser.add_argument(
        "--thermostat",
        type=str,
        choices=["Langevin", "Nose-Hoover"],
        default="Langevin",
        help="Thermostat.",
    )
    parser.add_argument("--temp", type=float, default=300.0, help="Temperature.")
    parser.add_argument("--time_step", type=float, default=1.0, help="Time step (fs).")
    parser.add_argument(
        "--friction",
        type=float,
        default=0.01,
        help="Friction in Langevin thermostat (1/fs).",
    )
    parser.add_argument(
        "--ttime",
        type=float,
        default=20.0,
        help="Time step interact with thermostat in Langevin thermostat (fs).",
    )
    parser.add_argument(
        "--n_eq", type=int, default=2000, help="Number of equilibration steps."
    )
    parser.add_argument("--n_steps", type=int, default=20000, help="Number of steps.")

    # for free energy perturbation and MD simulations for mixed state.
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Alpha value for reference state.",
    )
    parser.add_argument(
        "--fc2",
        type=str,
        default=None,
        help="Force constant HDF5 file for reference state.",
    )
    parser.add_argument(
        "--pot_ref",
        nargs="*",
        type=str,
        default=None,
        help="Reference polymlp state.",
    )

    # for TI
    parser.add_argument(
        "--ti",
        action="store_true",
        help="Run thermodynamics integration.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=15, help="Number of MD simulations for TI."
    )
    parser.add_argument(
        "--max_alpha", type=float, default=1.0, help="Maximum alpha value for TI."
    )
    parser.add_argument(
        "--fc2_path",
        type=str,
        default=None,
        help="Directory path for automatically finding reference FC2 state.",
    )

    args = parser.parse_args()

    print_credit()
    np.set_printoptions(legacy="1.21")

    path = "/".join(os.path.abspath(args.poscar).split("/")[:-1])
    if args.ti:
        print("Run TI from FC2 reference state.", flush=True)

        path += "/ti/" + str(args.temp)
        os.makedirs(path, exist_ok=True)
        md = PypolymlpMD(verbose=True)
        md.load_poscar(args.poscar)
        md.set_supercell(args.supercell_size)

        if args.fc2_path is not None:
            fc2 = md.find_reference(args.fc2_path, args.temp)
        else:
            fc2 = args.fc2
        print("Polymlp:      ", args.pot, flush=True)
        print("Reference FC2:", fc2, flush=True)

        md.set_ase_calculator_with_fc2(pot=args.pot, fc2hdf5=fc2)
        shutil.copy(fc2, path + "/fc2_ref.hdf5")

        md.run_thermodynamic_integration(
            thermostat=args.thermostat,
            n_alphas=args.n_samples,
            max_alpha=args.max_alpha,
            temperature=args.temp,
            time_step=args.time_step,
            ttime=args.ttime,
            friction=args.friction,
            n_eq=args.n_eq,
            n_steps=args.n_steps,
        )
        md.save_ti_yaml(filename=path + "/polymlp_ti.yaml")
    else:
        print("Run MD with NVT thermostat.", flush=True)
        print("Polymlp:      ", args.pot, flush=True)
        md = PypolymlpMD(verbose=True)
        md.load_poscar(args.poscar)
        md.set_supercell(args.supercell_size)

        if args.fc2 is None and args.pot_ref is None:
            md.set_ase_calculator(pot=args.pot)
        elif args.fc2 is not None:
            print("Reference FC2:", args.fc2, flush=True)
            md.set_ase_calculator_with_fc2(
                pot=args.pot,
                fc2hdf5=args.fc2,
                alpha=args.alpha,
            )
        else:
            print("Reference Polymlp:", args.pot_ref, flush=True)
            md.set_ase_calculator_with_reference(
                pot=args.pot,
                pot_ref=args.pot_ref,
                alpha=args.alpha,
            )

        md.run_md_nvt(
            thermostat=args.thermostat,
            temperature=args.temp,
            time_step=args.time_step,
            friction=args.friction,
            ttime=args.ttime,
            n_eq=args.n_eq,
            n_steps=args.n_steps,
            interval_save_forces=None,
            interval_save_trajectory=None,
            interval_log=1,
            logfile=path + "/polymlp_md_log.dat",
        )
        md.save_yaml(filename=path + "/polymlp_md.yaml")
