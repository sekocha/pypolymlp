#!/usr/bin/env python
import numpy as np
import os
from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure

def read_summary_nonequiv(file1):

    f = open(file1)
    lines = f.readlines()[1:]
    f.close()

    data = []
    for l in lines:
        txt1 = l.split(',')
        id1 = txt1[0]
        sum1, anx, sg = txt1[-3:]
        if len(txt1) == 5:
            type1 = txt1[1]
        elif len(txt1) > 5:
            type1_split = txt1[1:-3]
            type1 = '.'.join(type1_split)

        txt1 = sg.split()
        sg_symbol = txt1[0]
        sg_num = int(txt1[1].replace('(','').replace(')',''))

        data.append([id1, type1, sum1, anx, sg_symbol, sg_num])

    return data


def analysis(poscar_name=None, st=None, neighbor_scale=1.2):

    if poscar_name is not None and st is None:
        st = Poscar(poscar1).get_structure_class()

    natoms_sum = sum(st.n_atoms)
    st.calc_neighbors_bop(neighbor_scale=neighbor_scale)
    dis, _, _, = st.get_neighbors()

    coord = [sum([len(d1) for d1 in d]) for d in dis]

    zmean, zstd = np.mean(coord), np.std(coord)
    return zmean, zstd, natoms_sum


n_ele, system = 1, 'all'
#n_ele, system = 2, 'alloy'
#n_ele, system = 2, 'ionic'
#n_ele, system = 3, 'alloy'
#n_ele, system = 3, 'ionic'

target_system = str(n_ele) + '-' + system

ifile = '../icsd_entries/' + target_system + '/summary_nonequiv'
data = read_summary_nonequiv(ifile)

outdir = './' + target_system + '/'
os.makedirs(outdir, exist_ok=True)
outfile = outdir + '/summary_nonequiv'

poscardir = '../icsd_entries/poscars/'

if n_ele == 1 and system == 'all':
    n_atoms_sum_max = 12
    sg_num_min = 75
    zmean_min, zmean_max, zstd_max = 2, 16, 2.0
elif n_ele == 2 and system == 'alloy':
    n_atoms_sum_max = 12
    sg_num_min = 75
    zmean_min, zmean_max, zstd_max = 6, 16, 2.0
elif n_ele == 2 and system == 'ionic':
    n_atoms_sum_max = 16
    sg_num_min = 75
    zmean_min, zmean_max, zstd_max = 2, 8, 10.0
elif n_ele == 3 and system == 'alloy':
    n_atoms_sum_max = 12
    sg_num_min = 75
    zmean_min, zmean_max, zstd_max = 6, 16, 2.0
elif n_ele == 3 and system == 'ionic':
    n_atoms_sum_max = 16
    sg_num_min = 75
    zmean_min, zmean_max, zstd_max = 2, 8, 10.0

f = open(outfile, 'w')
print('# CollCode, StructureType, Sum, ANX, SpaceGroup', file=f)
for id1, type1, sum1, anx, sg_symbol, sg_num in data:
    poscar1 = poscardir + 'icsd-' + str(id1)
    st = Poscar(poscar1).get_structure_class()
    natoms_sum = sum(st.n_atoms)
    if natoms_sum <= n_atoms_sum_max and sg_num >= sg_num_min:
        zmean, zstd, _ = analysis(st=st)
        if zmean >= zmean_min and zmean <= zmean_max and zstd < zstd_max:
            print(id1, type1, sum1, anx, 
                 ' ' + sg_symbol + ' ('+str(sg_num)+')', sep=',', file=f)

f.close()

