#!/usr/bin/env python
import argparse

import numpy as np

from pypolymlp.calculator.sscha.utils.find_tc import find_tc
from pypolymlp.calculator.sscha.utils.utils import SummaryEOSYaml


def fit_gp(d1, order=3):
    return np.polyfit(d1[:, 0], d1[:, 1], order)


def find_optimal_order(d1_train, d1_test, d2_train, d2_test):

    min_rmse = 1e10
    for order in [3, 6, 9, 12, 15]:
        poly1 = fit_gp(d1_train, order=order)
        poly2 = fit_gp(d2_train, order=order)

        gibbs_t1 = d1_test[:, 1]
        gibbs_t2 = d2_test[:, 1]
        gibbs_p1 = np.polyval(poly1, d1_test[:, 0])
        gibbs_p2 = np.polyval(poly2, d2_test[:, 0])
        rmse1 = np.sqrt(np.average(np.square(gibbs_p1 - gibbs_t1)))
        rmse2 = np.sqrt(np.average(np.square(gibbs_p2 - gibbs_t2)))
        if rmse1 + rmse2 < min_rmse:
            min_rmse = rmse1 + rmse2
            order_opt = order
            poly_opt = poly2 - poly1

    return poly_opt, order_opt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml", nargs=2, type=str, default=None, help="two yaml files"
    )
    parser.add_argument("--p_max", type=float, default=10.0, help="maximum pressure")
    parser.add_argument("--p_step", type=float, default=1.0, help="pressure interval")
    parser.add_argument(
        "--tc_min",
        type=float,
        default=0.0,
        help="minimum of transition temperature",
    )
    parser.add_argument(
        "--tc_max",
        type=float,
        default=np.inf,
        help="maximum of transition temperature",
    )
    args = parser.parse_args()

    summary1 = SummaryEOSYaml(args.yaml[0])
    summary2 = SummaryEOSYaml(args.yaml[1])

    gp1 = summary1.eos_fit_gibbs
    gp2 = summary2.eos_fit_gibbs

    poly_diff_dict = dict()
    for temp, gp_data in gp1.items():
        if temp in gp2:
            gp2_data = gp2[temp]
            d1_train = [gp_data[i] for i in range(len(gp_data)) if i % 10 != 0]
            d1_test = [gp_data[i] for i in range(len(gp_data)) if i % 10 == 0]
            d2_train = [gp2_data[i] for i in range(len(gp2_data)) if i % 10 != 0]
            d2_test = [gp2_data[i] for i in range(len(gp2_data)) if i % 10 == 0]
            d1_train = np.array(d1_train)
            d1_test = np.array(d1_test)
            d2_train = np.array(d2_train)
            d2_test = np.array(d2_test)
            poly_diff, order_opt = find_optimal_order(
                d1_train, d1_test, d2_train, d2_test
            )
            poly_diff_dict[temp] = poly_diff

    print(" # pressure (GPa), temperature (K)")
    order = 3
    for press in np.arange(0, args.p_max + 0.01, args.p_step):
        gt_data = []
        for temp in poly_diff_dict.keys():
            gibbs = np.polyval(poly_diff_dict[temp], press)
            gt_data.append([temp, gibbs])
        tc = find_tc(np.array(gt_data), order=order)
        tc = [t for t in tc if t <= args.tc_max and t >= args.tc_min]
        if len(tc) == 1:
            print(press, tc[0])
        else:
            # print('warning: multiple transition temperatures are found.')
            print(press, tc)
