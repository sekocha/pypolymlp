#!/usr/bin/env python 
import numpy as np

from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.calculator.properties import Properties

from scipy.optimize import minimize
import time


class Minimize:
    
    def __init__(self, cell, pot=None, params_dict=None, coeffs=None):

        if pot is not None:
            params_dict, mlp_dict = load_mlp_lammps(filename=pot)
            coeffs = mlp_dict['coeffs'] / mlp_dict['scales']

        self.prop = Properties(params_dict=params_dict, coeffs=coeffs)

        self.__energy = None
        self.__force = None
        self.__stress = None

        self.__axis = cell['axis']
        self.__n_atoms = cell['n_atoms']
        self.__types = cell['types']
        self.__elements = cell['elements']
        self.__axis_inv = np.linalg.inv(cell['axis'])

        self.__x0 = (self.__axis @ cell['positions']).T.reshape(-1)
        self.__res = None

    def fun(self, x, args=None):

        st_dict = self.x_to_st_dict(x)
        self.__energy, self.__force, self.stress = self.prop.eval(st_dict)
        return self.__energy

    def jac(self, x, args=None):

        derivatives = - self.__force.T.reshape(-1)
        return derivatives

    def x_to_st_dict(self, x):

        r = x.reshape((-1,3)).T
        positions = self.__axis_inv @ r

        st_dict = {
            'axis': self.__axis,
            'positions': positions,
            'n_atoms': self.__n_atoms,
            'types': self.__types,
            'elements': self.__elements,
        }
        return st_dict

    def run(self): 
        options = {
            'gtol': 1e-3,
        }
        self.__res = minimize(self.fun, 
                              self.__x0, 
                              method='CG', 
                              jac=self.jac,
                              options=options)
        return self

    @property
    def structure(self):
        return self.x_to_st_dict(self.__res.x)

    @property
    def energy(self):
        return self.__res.fun

    @property
    def residual_forces(self):
        return - self.__res.jac.reshape((-1,3)).T
    
   
if __name__ == '__main__':

    import argparse
    from pypolymlp.core.interface_vasp import Poscar

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar', 
                        type=str, 
                        default=None,
                        help='poscar file')
    parser.add_argument('--pot', 
                        type=str, 
                        default='polymlp.lammps',
                        help='polymlp file')
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).get_structure()
    minobj = Minimize(unitcell, pot=args.pot)
    minobj.run()

    print(minobj.energy)
    print(minobj.residual_forces)



