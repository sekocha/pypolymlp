#!/usr/bin/env python 
import numpy as np

from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.core.interface_vasp import parse_vaspruns


def get_variable_with_max_length(multiple_params_dicts, key):

    array = []
    for single in multiple_params_dicts:
        if len(single[key]) > len(array):
            array = single[key]
    return array


def set_common_params_dict(multiple_params_dicts):

    keys = set()
    for single in multiple_params_dicts:
        for k in single.keys():
            keys.add(k)

    common_params_dict = copy.copy(multiple_params_dicts[0])

    n_type = max([single['n_type'] for single in multiple_params_dicts])

    elements = get_variable_with_max_length(multiple_params_dicts, 'elements')
    bool_element_order = [single['element_order']
                         for single in multiple_params_dicts] == True
    element_order = elements if bool_element_order else None

    atom_e = get_variable_with_max_length(
        multiple_params_dicts, 'atomic_energy'
    )

    common_params_dict['n_type'] = n_type
    common_params_dict['elements'] = elements
    common_params_dict['element_order'] = element_order
    common_params_dict['atomic_energy'] = atom_e

    return common_params_dict


class PolymlpDevParams:
    """
    Variables in params_dict
    ------------------------
      - n_type
      - include_force
      - include_stress
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
      - atomic_energy
      - reg
        - method
        - alpha
      - dft
        - train (vasprun locations)
        - test (vasprun locations)
    
    Variables in dft_dict (train_dft_dict, test_dft_dict)
    -----------------------------------------------------
        - energy
        - force
        - stress
        - structures
          - structure (1) 
            - axis
            - positions
            - n_atoms
            - types
            - elements
          - ...
        - elements
        - volumes
        - total_n_atoms
    
    Variables in reg_dict
        - x
        - y
        - first_indices [(ebegin, fbegin, sbegin), ...]
        - n_data (ne, nf, ns)
        - scales
    """
    def __init__(self):

        self.__params_dict = None
        self.__hybrid_params_dicts = None

        self.__train_dict = None
        self.__test_dict = None

    def parse_infile(self, infile):

        p = ParamsParser(infile)
        self.__params_dict = p.get_params()
        return self

    def parse_hybrid_infiles(self, infiles):

        self.__hybrid_params_dicts = [
            ParamsParser(infile, multiple_datasets=True).get_params() 
            for infile in infiles
        ]
        self.__params_dict = set_common_params_dict(params)
        return self

    def parse_single_dataset(self):

        if self.__params_dict is None:
            raise ValueError('parse_dataset: params_dict is needed.')

        dataset_type = self.__params_dict['dataset_type'] 
        if dataset_type == 'vasp':
            self.__train_dict = parse_vaspruns(
                self.__params_dict['dft']['train'],
                element_order=self.__params_dict['element_order']
            )
            self.__test_dict = parse_vaspruns(
                self.__params_dict['dft']['test'],
                element_order=self.__params_dict['element_order']
            )
        elif dataset_type == 'phono3py':
            from pypolymlp.core.interface_phono3py import parse_phono3py_yaml

            self.__train_dict = parse_phono3py_yaml(
                self.__params_dict['dft']['train']['phono3py_yaml'],
                self.__params_dict['dft']['train']['energy'],
                element_order=self.__params_dict['element_order'],
                select_ids=self.__params_dict['dft']['train']['indices'],
                use_phonon_dataset=False
            )
            self.__test_dict = parse_phono3py_yaml(
                self.__params_dict['dft']['test']['phono3py_yaml'],
                self.__params_dict['dft']['test']['energy'],
                element_order=self.__params_dict['element_order'],
                select_ids=self.__params_dict['dft']['test']['indices'],
                use_phonon_dataset=False
            )

        self.__train_dict = self.__apply_atomic_energy(self.__train_dict)
        self.__test_dict = self.__apply_atomic_energy(self.__test_dict)
        return self

    def parse_multiple_datasets(self):

        if self.__params_dict is None:
            raise ValueError('parse_dataset: params_dict is needed.')

        dataset_type = self.__params_dict['dataset_type'] 
        if dataset_type == 'vasp':
            element_order = self.__params_dict['element_order']
            self.__train_dict, self.__test_dict = dict(), dict()
            for set_id, dict1 in self.__params_dict['dft']['train'].items():
                self.__train_dict[set_id] = parse_vaspruns(
                    dict1['vaspruns'], element_order=element_order
                )
                self.__train_dict[set_id].update(dict1)

            for set_id, dict1 in self.__params_dict['dft']['test'].items():
                self.__test_dict[set_id] = parse_vaspruns(
                    dict1['vaspruns'], element_order=element_order
                )
                self.__test_dict[set_id].update(dict1)
        else:
            raise KeyError('Only dataset_type = vasp is available.')

        for _, dft_dict in self.__train_dict.items():
            dft_dict = self.__apply_atomic_energy(dft_dict)
        for _, dft_dict in self.__test_dict.items():
            dft_dict = self.__apply_atomic_energy(dft_dict)

        return self

    def __apply_atomic_energy(self, dft_dict):

        energy = dft_dict['energy']
        structures = dft_dict['structures']
        atom_e = self.__params_dict['atomic_energy']
        coh_energy = [e - np.dot(st['n_atoms'], atom_e)
                        for e, st in zip(energy, structures)]
        dft_dict['energy'] = np.array(coh_energy)
        return dft_dict

    @property
    def params_dict(self):
        return self.__params_dict

    @params_dict.setter
    def params_dict(self, params):
        self.__params_dict = params

    @property
    def hybrid_params_dicts(self):
        return self.__hybrid_params_dicts

    @hybrid_params_dicts.setter
    def hybrid_params_dicts(self, params):
        self.__hybrid_params_dicts = params
        self.__params_dict = set_common_params_dict(params)

    @property
    def train_dict(self):
        return self.__train_dict

    @property
    def test_dict(self):
        return self.__test_dict

    @property
    def min_energy(self):
        min_e = 1e10
        for dft_dict in self.__train_dict.values():
            e_per_atom = dft_dict['energy'] / dft_dict['total_n_atoms']
            min_e_trial = np.min(e_per_atom)
            if min_e_trial < min_e:
                min_e = min_e_trial
        return min_e


