#!/usr/bin/env python
from math import sqrt

import numpy as np
import spglib


def upper_triangle_axis(a, b, c, d, e, f):
    return np.array([[a, d, e], [0, b, f], [0, 0, c]])


def symmetric_metric(a, b, c, d, e, f):
    return np.array([[a, d, e], [d, b, f], [e, f, c]])


def transformation_matrix(axis_old, axis_new):
    return np.linalg.inv(axis_old) @ axis_new


def axis_to_metric(axis: np.array):
    return axis.T @ axis


def metric_to_axis(metric):

    # using notation in lammps documentation (triclinic boxes)
    aa, bb, cc = metric[:3]
    ab, bc, ac = metric[3:]
    ax = sqrt(aa)
    bx = ab / ax

    try:
        by = sqrt(bb - bx * bx)
        if by / sqrt(bb) < 0.01:
            raise ValueError
    except ValueError:
        raise ValueError

    cx = ac / ax
    cy = (bc - bx * cx) / by

    try:
        cz = sqrt(cc - cx * cx - cy * cy)
        if cz / sqrt(cc) < 0.01:
            raise ValueError
    except ValueError:
        raise ValueError

    return upper_triangle_axis(ax, by, cz, bx, cx, cy)


def axis_to_niggli_reduce(axis: np.array):

    lattice = axis.T
    niggli_lattice = spglib.niggli_reduce(lattice, eps=1e-15).T
    transformation = transformation_matrix(axis, niggli_lattice)
    metric = axis_to_metric(niggli_lattice)
    return niggli_lattice, transformation, metric
