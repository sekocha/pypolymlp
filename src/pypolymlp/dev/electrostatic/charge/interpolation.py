#!/usr/bin/env python
import sys
import numpy as np
import itertools
from math import *

from pyml.common.readvasp import Poscar, Chg

def fft_interpolation\
    (vasp_charge_density, grid_original, grid_expand, vol, axis, origin):

#    reciprocal_axis = np.linalg.inv(axis).T

    charge_d_r = vasp_charge_density.reshape\
        (tuple(reversed(grid_original))).transpose((2,1,0))
    fft1 = np.fft.fftn(charge_d_r)

    point_charge = fft1[0,0,0] * vol / np.prod(grid_original)
    print('point charge = ', point_charge, np.real(point_charge))

    dipole = np.array([0+0j, 0+0j, 0+0j])

    for i in range(1,grid_original[0]):
        if (i < grid_original[0]/2):
            index = i
        else:
            index = i-grid_original[0]
        gx = index * 2 * pi / axis[0,0]
        phase = e ** (1j * gx * (origin[0] - 0.5) * axis[0,0])
        dipole[0] += -1j * fft1[i,0,0] * phase / gx

    for i in range(1,grid_original[1]):
        if (i < grid_original[1]/2):
            index = i
        else:
            index = i-grid_original[1]
        gy = index * 2 * pi / axis[1,1]
        phase = e ** (1j * gy * (origin[1] - 0.5) * axis[1,1])
        dipole[1] += -1j * fft1[0,i,0] * phase / gy

    for i in range(1,grid_original[2]):
        if (i < grid_original[2]/2):
            index = i
        else:
            index = i-grid_original[2]
        gz = index * 2 * pi / axis[2,2]
        phase = e ** (1j * gz * (origin[2] -0.5) * axis[2,2])
        dipole[2] += -1j * fft1[0,0,i] * phase / gz

    dipole = dipole * vol / np.prod(grid_original)
    print('dipole =', dipole, np.real(dipole))

    fft2 = np.fft.irfftn(fft1, grid_expand)
    charge = np.ravel(fft2) * vol / np.prod(grid_original)
    return charge

def get_grid_fractional_coordinates(grid):

    grid_fracs = np.array([np.array(x) for x in itertools.product\
        (*[range(g) for g in grid])]).T
    grid_fracs = np.array([g/grid[i] for i, g in enumerate(grid_fracs)])

    return grid_fracs

def get_grid_cartesian_coordinates(grid_fracs, axis, origin):

    frac_new = grid_fracs - np.tile(origin, (grid_fracs.shape[1], 1)).T
    frac_new[np.where(frac_new > 0.5)] -= 1.0
    frac_new[np.where(frac_new < -0.5)] += 1.0
    return np.dot(axis, frac_new)

#@numba.jit
#def grid_quadrupole(v):
#    return 1.5 * np.outer(v, v) - 0.5 * np.dot(v,v) * np.identity(3)
 
if __name__ == '__main__':

    axis, positions, n_atoms, _, _ = Poscar('POSCAR').get_structure()
    chg = Chg('BvAt0006.dat')
    grid_original, grid_expand = chg.grid, np.array(chg.grid)*1
    grid_fracs = get_grid_fractional_coordinates(grid_expand)

    for i in range(sum(n_atoms)):
        print(' ### atom', i, ' ###')
        chg = Chg('BvAt'+str(i+1).zfill(4)+'.dat')
        charge = fft_interpolation\
            (chg.get_chg_density(), grid_original, grid_expand, \
            chg.vol, axis, positions[:,i])

        grid_vec = get_grid_cartesian_coordinates\
            (grid_fracs, axis, positions[:,i])

        cutoff = np.where(np.linalg.norm(grid_vec, axis=0) < 2.0)[0]

        charge = -np.array(charge)
        m1 = np.sum(charge)
        print(i, m1)
        m2 = np.dot(grid_vec, charge)
        print(i, m2)
        m2 = np.dot(grid_vec[:,cutoff], charge[cutoff])
        print(i, m2, 'cutoff')

        dis2 = 0.5 * np.square(np.linalg.norm(grid_vec, axis=0))
        print(dis2.shape)

        grid_quadrupole = [1.5 * v[:,None]*v[None,:] \
            - d * np.identity(3) for v, d in zip(grid_vec.T, dis2)]
        grid_quadrupole = np.array(grid_quadrupole).transpose((1,2,0))
        m3 = np.dot(grid_quadrupole[:,:,cutoff], charge[cutoff])
        print(i, m3)


        octa2 = []
        for v in grid_vec.T:
            tmp = np.zeros([3,3,3])
            for j in range(3):
                tmp[:,j,j] += v
                tmp[j,:,j] += v
                tmp[j,j,:] += v
            octa2.append(tmp)

        grid_octapole = [2.5 * v[:,None,None]*v[None,:,None]*v[None,None,:]\
            - d * o for v, d, o in zip(grid_vec.T, dis2, octa2)]
        grid_octapole = np.array(grid_octapole).transpose((1,2,3,0))
        m4 = np.dot(grid_octapole[:,:,:,cutoff], charge[cutoff])
        print(i, m4)




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



