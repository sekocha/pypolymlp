#!/usr/bin/env python
import argparse
import itertools
import os
from math import pi, sin

import numpy as np
from scipy.integrate import quad


def chi(x, l, n):
    return sin((2 * l + 1) * n * x / 2) / sin(x * n / 2)


def chi1(x, lcomb):
    return np.prod([chi(x, l, 1) for l in lcomb])


def n_all_lcomb(lcomb, lproj):
    num, _ = quad(
        lambda x: chi(x, lproj, 1) * chi1(x, lcomb) * sin(x / 2) ** 2,
        0,
        2 * pi,
    )
    return round(num / pi)


def n_all(maxl, order, lproj):

    lcomb_all = itertools.combinations_with_replacement(range(0, maxl + 1), order)
    lcomb_all = list(lcomb_all)
    n_list = [n_all_lcomb(lcomb, lproj) for lcomb in lcomb_all]
    return n_list, lcomb_all


def print_n_comb(n_list, lcomb_all, fname="l_list.yaml"):

    f = open(fname, "w")
    print("l_combinations:", file=f)
    for n, comb in zip(n_list, lcomb_all):
        if n > 0:
            print("- l:", list(comb), file=f)
            print("  num:", n, file=f)
            print("", file=f)
    print("", file=f)
    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--maxl", type=int, default=20, help="Max l value.")
    parser.add_argument("--order", type=int, default=None, help="n-th order product.")
    parser.add_argument("--lproj", type=int, default=0, help="l value for projection")
    args = parser.parse_args()

    orders = list(range(1, 7)) if args.order is None else [args.order]
    for order in orders:
        n_list, lcomb_all = n_all(args.maxl, order, args.lproj)

        n_total = []
        for l in range(0, args.maxl + 1):
            n1 = sum([n for n, lcomb in zip(n_list, lcomb_all) if max(lcomb) <= l])
            n_total.append(n1)

        print("Order:", order)
        print("Cumulative number of combinations:", n_total)
        os.makedirs("./lists_ver2", exist_ok=True)
        filename = (
            "./lists_ver2/lcomb-order" + str(order) + "-l" + str(args.lproj) + ".yaml"
        )
        print_n_comb(n_list, lcomb_all, fname=filename)
