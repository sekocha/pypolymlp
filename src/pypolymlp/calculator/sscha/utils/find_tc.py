#!/usr/bin/env python
import argparse

import numpy as np
import sympy

from pypolymlp.calculator.sscha.utils.utils import parse_summary_yaml


def get_diff(data1, data2):
    dict1, dict2 = dict(data1), dict(data2)
    diff = []
    for k, v in sorted(dict1.items()):
        if k in dict2:
            diff.append([k, v, dict2[k], dict2[k] - v])
    return np.array(diff)


def find_tc(d1, order=3):

    z1 = np.polyfit(d1[:, 0], d1[:, 1], order)
    x = sympy.Symbol("x")

    if order == 4:
        Sol2 = sympy.solve(
            z1[0] * x**4 + z1[1] * x**3 + z1[2] * x**2 + z1[3] * x + z1[4]
        )
    elif order == 3:
        Sol2 = sympy.solve(z1[0] * x**3 + z1[1] * x**2 + z1[2] * x**1 + z1[3])
    elif order == 2:
        Sol2 = sympy.solve(z1[0] * x**2 + z1[1] * x**1 + z1[2])
    elif order == 1:
        Sol2 = sympy.solve(z1[0] * x**1 + z1[1])
    else:
        raise KeyError(" order != 1 - 4")

    sol = []
    for s in Sol2:
        real, imag = s.as_real_imag()
        if abs(imag) < 1e-15 and real > 0 and real < 5000:
            sol.append(real)
    if len(sol) > 0:
        return sorted(sol)
    return []


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml", nargs=2, type=str, default=None, help="two yaml files"
    )
    args = parser.parse_args()

    ft1 = parse_summary_yaml(args.yaml[0])
    ft2 = parse_summary_yaml(args.yaml[1])
    diff = get_diff(ft1, ft2)

    if diff.shape[0] > 1:
        tc = find_tc(diff[:, [0, 3]], order=3)
        # tc = find_tc(diff[:,[0,3]], order=1)
    else:
        tc = []
    print(" tc =", tc)

    f = open("free_energy.dat", "w")
    print("# temp., F:", args.yaml[1], "-", args.yaml[0], file=f)
    for d in diff:
        print(int(d[0]), d[1], d[2], d[3], file=f)
    f.close()
