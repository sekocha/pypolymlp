#!/usr/bin/env python
import numpy as np
import argparse

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure

class StructureOperations:

    def __init__(self, 
                 filename=None,
                 structure=None,
                 format_init='vasp'):

        if format_init == 'vasp':
            p = Poscar(filename)
            self.axis, self.positions, self.n_atoms, \
                self.elements, self.types = p.get_structure()
        elif format_init == 'structure_class':
            if structure is None:
                raise ValueError("StructureOperation: structure is None")
            self.axis = structure.axis
            self.positions = structure.positions
            self.n_atoms = structure.n_atoms
            self.elements = structure.elements
            self.types = structure.types

    def expand(self, eps=1.0):
        eps1 = pow(eps, 0.3333333333333)
        self.axis *= eps1

    def disp(self, eps=0.001):
        size1, size2 = self.positions.shape
        positions_disp = (2 * np.random.rand(size1, size2) - 1) * eps
        self.positions += positions_disp

    def remove(self, index):
        begin = int(np.sum(self.n_atoms[:index]))
        end = begin + self.n_atoms[index]
        self.positions = np.delete(self.positions, range(begin,end), axis=1)
        del self.n_atoms[index]
        del self.elements[begin:end]
        del self.types[begin:end]

    def remove_zero_atoms(self):
        self.n_atoms = [n for n in self.n_atoms if n > 0]

    def element_permutation(self, order=None, index1=None, index2=None):

        if order is None:
            order = list(range(len(self.n_atoms)))
            order[index1], order[index2] = order[index2], order[index1]

        positions, n_atoms, types, elements = [], [], [], []
        for i in order:
            begin = int(np.sum(self.n_atoms[:i]))
            end = int(begin+self.n_atoms[i])
            n_atoms.append(self.n_atoms[i])
            positions.extend(self.positions.T[begin:end])
            types.extend(self.types[begin:end])
            elements.extend(self.elements[begin:end])

        self.n_atoms, self.types = n_atoms, types
        self.elements = elements
        self.positions = np.array(positions).T

    def get_minimum_neighborhood_distance(self, each_atom=True):
        structure = Structure(self.axis, 
                              self.positions, 
                              self.n_atoms, 
                              self.elements, 
                              self.types)
        if each_atom == True:
            return structure.calc_min_distance()
        else:
            return min(structure.calc_min_distance())

    def supercell(self, size=[1,1,1], overwrite=True):

        structure = Structure(self.axis, 
                              self.positions, 
                              self.n_atoms, 
                              self.elements, 
                              self.types)
        structure.supercell(size)

        if overwrite:
            self.axis = structure.axis
            self.positions = structure.positions
            self.n_atoms = structure.n_atoms
            self.elements = structure.elements
            self.types = structure.types
        else:
            return structure

    def expand_multiple(self, eps_min=0.8, eps_max=2.0, n_eps=10):
        volmin = pow(eps_min, 0.3333333333333)
        volmax = pow(eps_max, 0.3333333333333)
        eps_array = np.linspace(volmin,volmax,n_eps)
        axis_array = [self.axis * eps for eps in eps_array]
        st_array = [Structure(a, self.positions, self.n_atoms, 
                    self.elements, self.types) for a in axis_array]
        return st_array


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar', 
                        type=str,
                        default='POSCAR',
                        help='poscar file name')

    parser.add_argument('--volume',
                        action='store_true', 
                        help='changing volume')
    parser.add_argument('--volume_min', 
                        type=float, 
                        default=0.8,
                        help='Minimum volume ratio')
    parser.add_argument('--volume_max', 
                        type=float, 
                        default=2.0,
                        help='Maximum volume ratio')
    parser.add_argument('--volume_n', 
                        type=int, 
                        default=10,
                        help='Number of volumes')

    parser.add_argument('--disp',
                        action='store_true', 
                        help='random atomic displacements')
    parser.add_argument('--disp_eps', 
                        type=float, 
                        default=0.001,
                        help='eps for displacement')

    parser.add_argument('--supercell',
                        nargs=3,
                        type=int, 
                        default=None,
                        help='construct supercell')

    parser.add_argument('--permutation',
                        nargs='*',
                        type=int, 
                        default=None,
                        help='element permutation')
    parser.add_argument('--elimination',
                        type=int, 
                        default=None,
                        help='element elimination')

    parser.add_argument('--distance_minimum',
                        action='store_true', 
                        help='show the minimum distance')
    parser.add_argument('--distance_minimum_each_atom',
                        action='store_true', 
                        help='show the minimum distance for each atom')

    args = parser.parse_args()

    stop = StructureOperations(filename=args.poscar, format_init='vasp')

    if args.volume:
        st_array = stop.expand_multiple(eps_min=args.volume_min,
                                        eps_max=args.volume_max,
                                        n_eps=args.volume_n)
        for i, st in enumerate(st_array):
            stop.print_poscar_tofile(structure=st,
                                     filename='MPOSCAR_'+str(i).zfill(3), 
                                     header=args.poscar)

    if args.disp:
        stop.disp(eps=args.disp_eps)
        stop.print_poscar()

    if args.permutation is not None:
        order = [i-1 for i in args.permutation]
        stop.element_permutation(order=order)
        stop.print_poscar()

    if args.elimination is not None:
        stop.remove(args.elimination-1)
        stop.print_poscar()

    if args.distance_minimum:
        mindis = stop.get_minimum_neighborhood_distance(each_atom=False)
        print(mindis)
    if args.distance_minimum_each_atom:
        mindis = stop.get_minimum_neighborhood_distance()
        print(mindis)

    if args.supercell is not None:
        stop.supercell(size=args.supercell)
        stop.print_poscar()
