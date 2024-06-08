#!/usr/bin/env python
import itertools

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


class AlloyEnergy:

    def __init__(self, data_min=None, data_ch=None):

        if data_min is not None:
            self.data_min = np.array(data_min)
            self.n_type = len(self.data_min[-1]) - 1
        else:
            self.data_min = data_min

        self.data_ch = data_ch
        self.e_end = None

    def compute_ch(self, data_min=None):

        if data_min is not None:
            self.data_min = np.array(data_min)
            self.n_type = len(self.data_min[-1]) - 1

        data = np.hstack([self.data_min[:, :-2], self.data_min[:, -1:]])
        ch = ConvexHull(data)
        v_convex = np.unique(ch.simplices)
        data_ch = self.data_min[v_convex].astype(float)

        self.e_end = self.find_end_members(data_ch)
        e_form_ch = self.compute_formation_e(data_ch)

        lower_convex = np.where(e_form_ch <= 1e-15)[0]
        self.data_ch = data_ch[lower_convex]
        return self.data_ch, v_convex[lower_convex]

    def find_end_members(self, data):

        data = np.array(data)
        n_type = len(data[-1]) - 1
        v_end = [np.where(data[:, col] == 1.0)[0][0] for col in range(n_type)]
        self.e_end = data[v_end][:, -1]
        return self.e_end

    def compute_formation_e(self, data):

        if self.e_end is None:
            self.e_end = self.find_end_members(data)

        data = np.array(data).astype(float)
        form_e = data[:, -1] - np.dot(data[:, :-1], self.e_end)
        return form_e

    """
    def compute_formation_e2(self, comp, e_array):

        if self.e_end is None:
            self.e_end = self.find_end_members(data)

        e_ref = np.dot(np.array(comp), self.e_end)
        form_e = np.array(e_array) - e_ref
        return form_e
    """

    def initialize_composition_partition(self):

        if self.data_ch is None:
            raise KeyError(
                "convex hull is required before "
                "using initialize_composition_partition"
            )

        self.comps_rec_array, self.e_tri_array = [], []
        for d in itertools.combinations(self.data_ch, self.n_type):
            d = np.array(d)
            comps = d[:, :-1].T
            e_tri = d[:, -1]
            rec = np.linalg.pinv(comps)
            self.comps_rec_array.append(rec)
            self.e_tri_array.append(e_tri)
        self.comps_rec_array = np.array(self.comps_rec_array)

    def get_convex_hull_energy(self, comp, tol=1e-5):

        ehull = 1e10
        partition_all = np.dot(self.comps_rec_array, comp)
        for partition, e_tri in zip(partition_all, self.e_tri_array):
            if (
                np.all(partition > -tol)
                and np.all(partition < 1 + tol)
                and abs(sum(partition) - 1.0) < tol
            ):
                ehull_trial = np.dot(partition, e_tri)
                if ehull_trial < ehull:
                    ehull = ehull_trial

        return ehull

    """Sometimes return wrong values"""

    def initialize_composition_partition_not_exact(self):

        if self.data_ch is None:
            raise KeyError(
                "convex hull is required before "
                "using initialize_composition_partition"
            )
        self.comps_rec_array = []
        self.e_tri_array = []
        if self.n_type > 2:
            tri = Delaunay(self.data_ch[:, :-2])
            ids = tri.simplices[10]
            for ids in tri.simplices:
                comps = np.array([self.data_ch[i, :-1] for i in ids]).T
                e_tri = np.array([self.data_ch[i, -1] for i in ids])
                comps_rec = np.linalg.pinv(comps)
                self.comps_rec_array.append(comps_rec)
                self.e_tri_array.append(e_tri)
        elif self.n_type == 2:
            data = sorted([tuple(c) for c in self.data_ch])
            for d1, d2 in zip(data[:-1], data[1:]):
                comps = np.vstack([d1[:-1], d2[:-1]]).T
                e_tri = np.array([d1[-1], d2[-1]])
                comps_rec = np.linalg.pinv(comps)
                self.comps_rec_array.append(comps_rec)
                self.e_tri_array.append(e_tri)

    """Sometimes return wrong values"""

    def get_convex_hull_energy_not_exact(self, comp):

        tol = 1e-5
        for rec, e_tri in zip(self.comps_rec_array, self.e_tri_array):
            partition = np.dot(rec, comp)
            if (
                np.all(partition > -tol)
                and np.all(partition < 1 + tol)
                and abs(sum(partition) - 1.0) < tol
            ):
                ehull = np.dot(partition, e_tri)
                break

        return ehull
