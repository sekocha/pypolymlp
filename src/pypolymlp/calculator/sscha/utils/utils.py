#!/usr/bin/env python
import numpy as np
import yaml

def check_imaginary(fname, tol=-0.1, print_log=True):

    dos = np.loadtxt(fname, dtype=float)
    occ_ids = np.where(dos[:,1] > 0)[0]
    eigs = dos[occ_ids]
    if eigs[0,0] < tol:
        imag_ids = np.where(eigs[:,0] < 0.001)[0]
        if print_log:
            print(fname, eigs[0,0], "{:f}".format(sum(eigs[imag_ids,1])))
        return True
    return False

def parse_free_energy_yaml(yaml_file):

    data = yaml.safe_load(open(yaml_file))
    free_energies = [[d['temperature'], d['free_energy']]
                    for d in data['equilibrium']]
    return np.array(free_energies)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files',
                        nargs='*',
                        type=str,
                        default=None,
                        help='targets')
    args = parser.parse_args()

    tol = -0.1
    for fname in sorted(args.files):
        check_imaginary(fname, tol=tol, print_log=True)

