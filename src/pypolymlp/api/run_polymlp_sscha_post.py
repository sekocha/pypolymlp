"""Command lines for post SSCHA calculations."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_sscha_post import PypolymlpSSCHAPost


def run():
    """Command lines for post SSCHA calculations.

    Examples
    --------
    # Calculation of thermodynamic properties using SSCHA results
    pypolymlp-sscha-post --properties --yaml ./*/sscha/*/sscha_results.yaml

    # Calculation of energies and forces for structures generated from a converged FC2
    pypolymlp-sscha-post --distribution --yaml sscha/300/sscha_results.yaml
                         --fc2 sscha/300/fc2.hdf5 --n_samples 20 --pot polymlp.lammps

    # Calculation of phase transition temperature
    pypolymlp-sscha-post --transition hcp/sscha_properties.yaml bcc/sscha_properties.yaml

    # Calculation of phase transition temperature and pressure
    pypolymlp-sscha-post --boundary hcp/sscha_properties.yaml bcc/sscha_properties.yaml
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--properties",
        action="store_true",
        help="Calculate thermodynamic properties.",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Calculate properties of structures sampled from FC2 density matrix.",
    )
    parser.add_argument(
        "--transition",
        nargs=2,
        type=str,
        help="Find phase transition.",
    )
    parser.add_argument(
        "--boundary",
        nargs=2,
        type=str,
        help="Compute phase boundary.",
    )
    parser.add_argument(
        "--yaml",
        nargs="*",
        type=str,
        default=None,
        help="sscha_results.yaml files",
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
        help="polymlp files",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=100,
        help="Number of sample supercells",
    )
    parser.add_argument(
        "--save_poscars",
        action="store_true",
        help="Save POSCAR files of distribution",
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")

    sscha = PypolymlpSSCHAPost(verbose=True)
    if args.properties:
        sscha.compute_thermodynamic_properties(
            args.yaml, filename="sscha_properties.yaml"
        )

    elif args.distribution:
        sscha.init_structure_distribution(
            yamlfile=args.yaml[0],
            fc2file=args.fc2,
            pot=args.pot,
        )
        sscha.run_structure_distribution(n_samples=args.n_samples)
        sscha.save_structure_distribution(path=".", save_poscars=args.save_poscars)
    elif args.transition:
        tc_linear, tc_poly = sscha.find_phase_transition(
            args.transition[0],
            args.transition[1],
        )
        print("Tc (Linear fit)     :", np.round(tc_linear, 1), flush=True)
        print("Tc (Polynomial fit) :", np.round(tc_poly, 1), flush=True)

    elif args.boundary:
        boundary = sscha.compute_phase_boundary(
            args.boundary[0],
            args.boundary[1],
        )
        print("phase_boundary:", flush=True)
        for data in boundary:
            print("- pressure:    ", np.round(data[0], 5), flush=True)
            print("  temperature: ", np.round(data[1], 2), flush=True)
