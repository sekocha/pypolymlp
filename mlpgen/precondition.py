#!/usr/bin/env python
import numpy as np

from sklearn.preprocessing import StandardScaler
import polymlp_generator.mlpgen.numba_support as numba_support

class Precondition:

    def __init__(self, 
                 reg_dict, 
                 dft_dict,
                 params_dict,
                 scaler=None, 
                 weight_stress=0.1): 

        self.x = reg_dict['x']
        self.first_indices = reg_dict['first_indices']
        self.dft_dict = dft_dict
        self.params_dict = params_dict

        self.n_data, self.n_features = self.x.shape
        if params_dict['include_force']:
            self.ne = self.first_indices[0][2]
            self.ns = self.first_indices[0][1] - self.ne
            self.nf = self.n_data - self.ne - self.ns
        else:
            self.ne = self.n_data
            self.nf, self.ns = 0, 0

        self.y = np.zeros(self.n_data)

        self.__apply_atomic_energy()
        self.__apply_scale(scaler=scaler)
        self.__apply_weight(weight_stress=weight_stress)

        # todo: atomic energy
        # todo: multiple datasets
        # todo: for alloy including end-members
        reg_dict['y'] = self.y

    def __apply_atomic_energy(self):

        energy = self.dft_dict['energy']
        structures = self.dft_dict['structures']
        atom_e = self.params_dict['atomic_energy']

        coh_energy_array = []
        for e, st in zip(energy, structures):
            coh_e = e
            for na, ea in zip(st['n_atoms'], atom_e):
                coh_e = coh_e - na * ea
            coh_energy_array.append(coh_e)
        self.dft_dict['energy'] = np.array(coh_energy_array)

    def __apply_scale(self, scaler=None):

        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler(with_mean=False).fit(self.x[:self.ne])

        '''
            correctly-working version 
                self.x /= self.scaler.scale_
        '''
        #numba_support.mat_divide_vec(self.x, self.scaler.scale_)
        numba_support.mat_prod_vec(self.x, 
                                   np.reciprocal(self.scaler.scale_),
                                   axis=1)

    def __apply_weight(self, weight_stress=0.1):

        self.__apply_weight_percentage(weight_stress=weight_stress)
        #self.apply_given_weight(weight_stress=weight_stress)

    def __apply_weight_percentage(self, weight_stress=0.1):

        if self.params_dict['include_force']:
            ebegin, fbegin, sbegin = self.first_indices[0]
            eend, fend, send = sbegin, self.n_data, fbegin
        else:
            ebegin, eend = 0, self.ne

        energy = self.dft_dict['energy']
        n_total_atoms = [sum(st['n_atoms']) 
                         for st in self.dft_dict['structures']]
        e_per_atom = energy / np.array(n_total_atoms)
        #print(e_per_atom)

        # todo: should be examined
        min_e = np.min(e_per_atom)
        e_th1 = min_e * 0.75
        e_th2 = min_e * 0.50
        
        weight_e = np.ones(len(energy))
        weight_e[e_per_atom > e_th1] = 0.5
        weight_e[e_per_atom > e_th2] = 0.3
        weight_e[e_per_atom > 0.0] = 0.1

        self.y[ebegin:eend] = weight_e * energy
        numba_support.mat_prod_vec(self.x[ebegin:eend], weight_e, axis=0)

        if self.params_dict['include_force']:
            force = self.dft_dict['force']

            log1 = np.log10(np.abs(force))
            w1 = np.array([pow(10, -v) for v in log1])
            weight_f = np.minimum(w1, np.ones(len(w1)))

            self.y[fbegin:fend] = weight_f * force
            numba_support.mat_prod_vec(self.x[fbegin:fend], weight_f, axis=0)

            if self.params_dict['include_stress']:
                stress = self.dft_dict['stress']
                log1 = np.log10(np.abs(stress))
                w1 = np.array([pow(5, -v) for v in log1])
                weight_s = np.minimum(w1, np.ones(len(w1))) * weight_stress

                self.y[sbegin:send] = weight_s * stress
                numba_support.mat_prod_vec(self.x[sbegin:send], weight_s, axis=0)
            else:
                self.x[sbegin:send,:] = 0.0
                self.y[sbegin:send] = 0.0

        return 0

    """
    # todo: for multiple datasets
    def apply_given_weight(self, d, weight_stress=0.1):

        ebegin, eend, fbegin, fend, sbegin, send = d.get_array_indices()
        self.y[ebegin:eend] = d.weight * d.model_e
        if math.isclose(d.weight, 1.0) == False:
            numba_support.mat_prod(self.x[ebegin:eend], d.weight)
        if d.wforce == True:
            self.y[fbegin:fend] = d.weight * d.model_f
            if math.isclose(d.weight, 1.0) == False:
                numba_support.mat_prod(self.x[fbegin:fend], d.weight)
            if d.wstress == True:
                w1 = d.weight * weight_stress
                numba_support.mat_prod(self.x[sbegin:send], w1)
                self.y[sbegin:send] = w1 * d.model_s
            else:
                self.x[sbegin:send,:] = 0.0
                self.y[sbegin:send] = 0.0
        return 0
    """

    def print_data_shape(self, header='training data size'):

        print(' ', header)
        print('  X (all) :', self.x.shape)
        print('   - n (energy) =', self.ne)
        print('   - n (force)  =', self.nf)
        print('   - n (stress) =', self.ns)

    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_scaler(self):
        return self.scaler
