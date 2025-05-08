#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

from ase_md import run_NVT

def get_argparse():
    parser = argparse.ArgumentParser(
        description="Run_NVT_MD_simulation",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-p', '--poscar_file',
                        type=str,
                        default="POSCAR",
                        help="POSCAR file path. e.g. 'POSCAR'")
    parser.add_argument('-m', '--potential_files',
                        type=str,
                        nargs="*",
                        default="polymlp.lammps",
                        help="potential file paths. e.g. 'polymlp.lammps polymlp_1000K.lammps'")
    parser.add_argument('--temperature',
                        type=int,
                        default=300,
                        help="temperature in K e.g. '0'")
    parser.add_argument('--time_step',
                        type=float,
                        default=1.0,
                        help="time step in fs. e.g. '1.0'")
    parser.add_argument('--ttime',
                        type=float,
                        default=20.0,
                        help="timescale of the thermostat in fs. e.g. '20.0'")
    parser.add_argument('--n_eq',
                        type=int,
                        default=50,
                        help="number of equilibration steps. e.g. '5000'")
    parser.add_argument('--n_steps',
                        type=int,
                        default=2000,
                        help="number of production steps. e.g. '10000'")
    argments = parser.parse_args()
    return argments


def main(
    poscar_file:str,
    potentials:list,
    temperature:list,
    time_step:float,
    ttime:float,
    n_eq:int,
    n_steps:int,
):
    print(f"### Running NVT MD simulation ###")
    run_NVT(
        poscar_file=poscar_file,
        potentials=potentials,
        temperature=temperature,
        time_step=time_step,
        ttime=ttime,
        n_eq=n_eq,
        n_steps=n_steps
    )
    print(f"### End simulation ###")


if __name__ == '__main__':
    args = get_argparse()
    main(
        poscar_file=args.poscar_file,
        potentials=args.potential_files,
        temperature=args.temperature,
        time_step=args.time_step,
        ttime=args.ttime,
        n_eq=args.n_eq,
        n_steps=args.n_steps,
    )
