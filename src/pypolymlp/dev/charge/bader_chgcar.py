#!/usr/bin/env python
import numpy as np
from scipy.special import sph_harm
from math import *
import itertools

''' todo: not to use pyml '''
from pyml.common.readvasp import Poscar
from pyml.common.structure import Structure

class Chg:
    def __init__(self, fname='CHG'):
        p = Poscar(fname)
        self.axis, self.positions, n_atoms, _, _ = p.get_structure()
        #st = Structure(self.axis, self.positions, n_atoms, elements, types)
        #self.vol = st.calc_volume()

        f = open(fname)
        lines2 = f.readlines()
        f.close()

        start = sum(n_atoms)+9
        self.grid = [int(i) for i in lines2[start].split()]
        self.ngrid = np.prod(self.grid)

        chg = [float(s) for line in lines2[start+1:] for s in line.split()]
        self.chg = np.array(chg) / self.ngrid

        self.grid_fracs = np.array([np.array([x[2],x[1],x[0]]) / self.grid \
            for x in itertools.product(range(self.grid[2]), \
            range(self.grid[1]), range(self.grid[0]))]).T


    def get_grid(self):
        return self.grid
    def get_grid_coordinates(self):
        self.grid_coordinates = np.dot(self.axis, self.grid_fracs.T)
        return self.grid_coordinates

    def get_grid_coordinates_atomcenter(self, atom):
        pos1 = self.positions[:,atom]
        frac_new = self.grid_fracs \
            - np.tile(pos1,(self.grid_fracs.shape[1], 1)).T
        frac_new[np.where(frac_new > 0.5)] -= 1.0
        frac_new[np.where(frac_new < -0.5)] += 1.0
        return np.dot(self.axis, frac_new)

    def get_ngrid(self):
        return self.ngrid
    def get_chg(self):
        return self.chg

#    def get_volume(self):
#        return self.vol


if __name__ == '__main__':
    _, _, n_atoms, _, _ = Poscar('POSCAR').get_structure()
    for i in range(sum(n_atoms)):
        chg = Chg('BvAt'+str(i+1).zfill(4)+'.dat')
        charge = chg.get_chg()
        grid_vec = chg.get_grid_coordinates_atomcenter(i)

        m1 = sum(charge)
        print(i, m1)
        m2 = np.sum([c*v for c, v in zip(charge, grid_vec.T)], axis=0)
        print(i, m2)
    

    #moment(grid_vec, charge, site, st.get_positions().shape[1], vec)

#def cart2sph(vec):
#    x, y, z = vec
#    XsqPlusYsq = x**2 + y**2
#    r = sqrt(XsqPlusYsq + z**2)               # r
#    elev = acos(z/r)                          # theta
#    az = atan2(y,x)                           # phi
#    return r, elev, az
#


#def moment(grid_vec, charge, site, n_site, vec):
#    m1 = [0.0 for n in range(n_site)]
#    m2 = [np.array([0.0, 0.0, 0.0]) for n in range(n_site)]
#    m3 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for n in range(n_site)]
#    tmp = [[0.0, 0.0, 0.0] for n in range(n_site)]
#    output = []
#    for i, (sph_vec, chg, s, v) in enumerate(zip(grid_vec, charge, site,vec)):
#        r = sph_vec[0]
#        if (r > 0.01):
#            m1[s] += chg
#    #    y1p1 = sph_harm(1, 1, sph_vec[2], sph_vec[1])
#    #    y10 = sph_harm(0, 1, sph_vec[2], sph_vec[1])
#    #    y1m1 = sph_harm(-1, 1, sph_vec[2], sph_vec[1])
#    #    m2[s][0] += chg * r * y1p1
#    #    m2[s][1] += chg * r * y10
#    #    m2[s][2] += chg * r * y1m1
#        tmp[s] += v
#        m2[s] += chg * v
##        m2[s][0] += chg * v[0]
##        m2[s][1] += chg * v[1]
##        m2[s][2] += chg * v[2]
#        if (s == 5):
#            output.append(tuple([chg, tuple(v)]))
##    for a in sorted(output):
##        print(a)
#
#    print(m1)
#    print(sum(m1))
#    print(np.array(m2))
#    print(sum(np.array(m2)))
#    print(np.array(tmp))



