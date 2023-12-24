#!/usr/bin/env python
import numpy as np
from scipy.special import sph_harm
from math import *
import itertools

''' todo: not to use pyml '''
from pyml.common.readvasp import Poscar, Vasprun
from pyml.common.structure import Structure

def cart2sph(vec):
    x, y, z = vec
    XsqPlusYsq = x**2 + y**2
    r = sqrt(XsqPlusYsq + z**2)               # r
    elev = acos(z/r)                          # theta
    az = atan2(y,x)                           # phi
    return r, elev, az

def read_structure(poscar_file=None,vasprun_file=None):
    if (poscar_file != None):
        p = Poscar(poscar_file)
        axis, positions, n_atoms, elements, types = p.get_structure()
    elif (vasprun_file != None):
        v = Vasprun(vasprun_file)
        axis, positions, n_atoms, vol, elements, types = v.get_structure()
    st = Structure(axis, positions, n_atoms, elements, types)
    return st

def read_bader_analysis(bcf_file='BCF.dat'):
    f = open(bcf_file)
    lines2 = f.readlines()
    f.close()

    grid, charge, site = [], [], []
    for line in lines2[2:-1]:
        d = line.split()
        grid.append([float(x) for x in d[1:4]])
        charge.append(float(d[4]))
        site.append(int(d[5])-1)

    return np.array(grid), charge, site

def find_grid_vector(st, grid, site):
    posc = st.get_positions_cartesian()
    transc = [np.dot(st.get_axis(), np.array(t)) \
        for t in itertools.product([-1,0,1], [-1,0,1],[-1,0,1])]

    grid_vec, vec = [], []
    for g, s in zip(grid, site):
        dis_list = [np.linalg.norm(g-posc[:,s]-t) for t in transc]
        index = dis_list.index(min(dis_list))
        vec.append(g-posc[:,s]-transc[index])
        grid_vec.append(cart2sph(g-posc[:,s]-transc[index]))
    return np.array(grid_vec), vec

def moment(grid_vec, charge, site, n_site, vec):
    m1 = [0.0 for n in range(n_site)]
    m2 = [np.array([0.0, 0.0, 0.0]) for n in range(n_site)]
    m3 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for n in range(n_site)]
    tmp = [[0.0, 0.0, 0.0] for n in range(n_site)]
    output = []
    for i, (sph_vec, chg, s, v) in enumerate(zip(grid_vec, charge, site,vec)):
        r = sph_vec[0]
        if (r > 0.01):
            m1[s] += chg
    #    y1p1 = sph_harm(1, 1, sph_vec[2], sph_vec[1])
    #    y10 = sph_harm(0, 1, sph_vec[2], sph_vec[1])
    #    y1m1 = sph_harm(-1, 1, sph_vec[2], sph_vec[1])
    #    m2[s][0] += chg * r * y1p1
    #    m2[s][1] += chg * r * y10
    #    m2[s][2] += chg * r * y1m1
        tmp[s] += v
        m2[s] += chg * v
#        m2[s][0] += chg * v[0]
#        m2[s][1] += chg * v[1]
#        m2[s][2] += chg * v[2]
        if (s == 5):
            output.append(tuple([chg, tuple(v)]))
#    for a in sorted(output):
#        print(a)

    print(m1)
    print(sum(m1))
    print(np.array(m2))
    print(sum(np.array(m2)))
    print(np.array(tmp))


if __name__ == '__main__':
    st = read_structure(vasprun_file='vasprun.xml')
    grid, charge, site = read_bader_analysis(bcf_file='BCF.dat')
    grid_vec, vec = find_grid_vector(st, grid, site)
    moment(grid_vec, charge, site, st.get_positions().shape[1], vec)

