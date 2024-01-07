#!/usr/bin/env python 
import numpy as np
import itertools
from collections import defaultdict

from pypolymlp.mlp_gen.generator import (
    run_generator_single_dataset,
    run_generator_single_dataset_from_params,
    run_generator_single_dataset_from_params_and_datasets,
)

from pypolymlp.cxx.lib import libmlpcpp
from pypolymlp.utils.displacements_utils import convert_disps_to_positions


class Pypolymlp:

    def __init__(self):

        """
        Keys in params_dict
        --------------------
        - n_type
        - include_force
        - include_stress
        - atomic_energy
        - dataset_type
        - model
          - cutoff
          - model_type
          - max_p
          - max_l
          - feature_type
          - pair_type
          - pair_params
          - gtinv
            - order
            - max_l
            - lm_seq
            - l_comb
            - lm_coeffs
        - reg
          - method
          - alpha
        - dft
          - train (dataset locations)
          - test (dataset locations)
        """
        self._params_dict = dict()
        self._params_dict['model'] = dict()
        self._params_dict['model']['gtinv'] = dict()
        self._params_dict['reg'] = dict()
        self._params_dict['dft'] = dict()
        self._params_dict['dft']['train'] = dict()
        self._params_dict['dft']['test'] = dict()

        self.train_dft_dict = None
        self.test_dft_dict = None

        self._mlp_dict = None

    def set_params(
            self, 
            elements,
            include_force=True,
            include_stress=False,
            cutoff=6.0,
            model_type=4,
            max_p=2,
            feature_type='gtinv',
            gaussian_params1=[1.0,1.0,1],
            gaussian_params2=[0.0,5.0,7],
            reg_alpha_params=[-3.0,1.0,5],
            gtinv_order=3,
            gtinv_maxl=[4,4,2,1,1],
            atomic_energy=None,
            rearrange_by_elements=True,
        ):

        '''
        Assign input parameters.

        Parameters
        ----------
        elements: Element species, (e.g., ['Mg','O'])
        include_force: Considering force entries
        include_stress: Considering stress entries
        cutoff: Cutoff radius
        model_type: Polynomial function type
            model_type = 1: Linear polynomial of polynomial invariants
            model_type = 2: Polynomial of polynomial invariants
            model_type = 3: Polynomial of pair invariants 
                            + linear polynomial of polynomial invariants
            model_type = 4: Polynomial of pair and second-order invariants
                            + linear polynomial of polynomial invariants
        max_p: Order of polynomial function
        feature_type: 'gtinv' or 'pair'
        gaussian_params: Parameters for exp[- param1 * (r - param2)**2]
            Parameters are given as np.linspace(p[0], p[1], p[2]),
            where p[0], p[1], and p[2] are given by gaussian_params1 
            and gaussian_params2.
        reg_alpha_params: Parameters for penalty term in 
            linear ridge regression. Parameters are given as 
            np.linspace(p[0], p[1], p[2]).
        gtinv_order: Maximum order of polynomial invariants.
        gtinv_maxl: Maximum angular numbers of polynomial invariants. 
            [maxl for order=2, maxl for order=3, ...]
        atomic_energy: Atomic energies.
        rearrange_by_elements: Set True if not developing special MLPs.

        All parameters are stored in self._params_dict.
        '''

        self._params_dict['elements'] = elements
        n_type = len(elements)
        self._params_dict['n_type'] = n_type
        self._params_dict['include_force'] = include_force
        self._params_dict['include_stress'] = include_stress

        model = self._params_dict['model']
        model['cutoff'] = cutoff

        if model_type > 4:
            raise ValueError('model_type != 1, 2, 3, or 4')
        model['model_type'] = model_type
            
        if max_p > 3:
            raise ValueError('model_type != 1, 2, or 3')
        model['max_p'] = max_p

        if feature_type != 'gtinv' and feature_type != 'pair':
            raise ValueError('feature_type != gtinv or pair')
        model['feature_type'] = feature_type

        model['pair_type'] = 'gaussian'
        if len(gaussian_params1) != 3:
            raise ValueError('len(gaussian_params1) != 3')
        if len(gaussian_params2) != 3:
            raise ValueError('len(gaussian_params2) != 3')
        params1 = self.__sequence(gaussian_params1)
        params2 = self.__sequence(gaussian_params2)
        model['pair_params'] = list(itertools.product(params1, params2))
        model['pair_params'].append([0.0,0.0])

        gtinv_dict = self._params_dict['model']['gtinv']
        if model['feature_type'] == 'gtinv':
            gtinv_dict['order'] = gtinv_order
            size = gtinv_dict['order'] - 1
            if len(gtinv_maxl) < size:
                raise ValueError('size (gtinv_maxl) !=', size)
            gtinv_dict['max_l'] = gtinv_maxl
            gtinv_sym = [False for i in range(size)]
            rgi = libmlpcpp.Readgtinv(gtinv_dict['order'],
                                      gtinv_dict['max_l'],
                                      gtinv_sym,
                                      n_type)
            gtinv_dict['lm_seq'] = rgi.get_lm_seq()
            gtinv_dict['l_comb'] = rgi.get_l_comb()
            gtinv_dict['lm_coeffs'] = rgi.get_lm_coeffs()
            model['max_l'] = max(gtinv_dict['max_l'])
        else:
            gtinv_dict['order'] = 0
            gtinv_dict['max_l'] = []
            gtinv_dict['lm_seq'] = []
            gtinv_dict['l_comb'] = []
            gtinv_dict['lm_coeffs'] = []
            model['max_l'] = 0

        if len(reg_alpha_params) != 3:
            raise ValueError('len(reg_alpha_params) != 3')
        self._params_dict['reg']['method'] = 'ridge'
        self._params_dict['reg']['alpha'] = self.__sequence(reg_alpha_params)

        if atomic_energy is None:
            self._params_dict['atomic_energy'] = [0.0 for i in range(n_type)]
        else:
            if len(atomic_energy) != n_type:
                raise ValueError('len(atomic_energy) != n_type')
            self._params_dict['atomic_energy'] = atomic_energy

        if rearrange_by_elements:
            self._params_dict['element_order'] = elements 
        else:
            self._params_dict['element_order'] = None

    def set_datasets_vasp(self, train_vaspruns, test_vaspruns):

        self._params_dict['dataset_type'] = 'vasp'
        self._params_dict['dft']['train'] = sorted(train_vaspruns)
        self._params_dict['dft']['test'] = sorted(test_vaspruns)

    def set_datasets_phono3py(
            self, 
            train_yaml, 
            train_energy_dat, 
            test_yaml, 
            test_energy_dat,
            train_ids=None,
            test_ids=None,
    ):

        self._params_dict['dataset_type'] = 'phono3py'
        data = self._params_dict['dft']
        data['train'], data['test'] = dict(), dict()
        data['train']['phono3py_yaml'] = train_yaml
        data['train']['energy'] = train_energy_dat
        data['test']['phono3py_yaml'] = test_yaml
        data['test']['energy'] = test_energy_dat

        data['train']['indices'] = train_ids
        data['test']['indices'] = test_ids


    def set_datasets_displacements(
            self,
            train_disps, 
            train_forces, 
            train_energies,
            test_disps,
            test_forces,
            test_energies,
            st_dict
    ):

        ''' 
        Parameters
        ----------

        displacements: (n_samples, 3, n_atom)
        forces: (n_samples, 3, n_atom)
        energies: (n_samples)
        '''
        self.train_dft_dict = self.__set_dft_dict(
                train_disps, 
                train_forces, 
                train_energies, 
                st_dict
        )
        self.test_dft_dict = self.__set_dft_dict(
                test_disps, 
                test_forces, 
                test_energies, 
                st_dict
        )

    def __set_dft_dict(self, disps, forces, energies, st_dict):

        positions_all = convert_disps_to_positions(disps,
                                                   st_dict['axis'],
                                                   st_dict['positions'])
        ''' common part in core/interface_phono3py.py'''
        dft_dict = defaultdict(list)
        dft_dict['energy'] = energies
        dft_dict['stress'] = np.zeros(forces.shape[0] * 6)

        for positions_iter, forces_iter in zip(positions_all, forces):
            st = dict()
            st['axis'] = st_dict['axis']
            st['positions'] = positions_iter
            st['n_atoms'] = st_dict['n_atoms']
            st['elements'] = st_dict['elements']
            st['types'] = st_dict['types']
            st['volume'] = st_dict['volume']

            dft_dict['force'].extend(forces_iter.T.reshape(-1))
            dft_dict['structures'].append(st)
        dft_dict['force'] = np.array(dft_dict['force'])

        elements_rep = dft_dict['structures'][0]['elements']
        dft_dict['elements'] = sorted(set(elements_rep), key=elements_rep.index)
        dft_dict['total_n_atoms'] = np.array([sum(st['n_atoms'])
                                             for st in dft_dict['structures']])
        n_data = len(dft_dict['structures'])
        dft_dict['filenames'] = ['disp-' + str(i+1).zfill(5)
                                   for i in range(n_data)]
        return dft_dict

    def __sequence(self, params):
        return np.linspace(float(params[0]), float(params[1]), int(params[2]))

    def run(self, file_params=None, log=True):

        if file_params is not None:
            self._mlp_dict = run_generator_single_dataset(file_params, log=log)
        else:
            if self.train_dft_dict is None:
                self._mlp_dict = run_generator_single_dataset_from_params(
                    self._params_dict,
                    log=log,
                )
            else:
                self._mlp_dict \
                    = run_generator_single_dataset_from_params_and_datasets(
                    self._params_dict,
                    self.train_dft_dict,
                    self.test_dft_dict,
                    log=log,
                )

    @ property
    def parameters(self):
        return self._params_dict

    @ property
    def summary(self):
        return self._mlp_dict

