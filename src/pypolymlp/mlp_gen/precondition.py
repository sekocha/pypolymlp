#!/usr/bin/env python
import numpy as np


def apply_atomic_energy(dft_dict, params_dict):

    energy = dft_dict["energy"]
    structures = dft_dict["structures"]
    atom_e = params_dict["atomic_energy"]
    coh_energy = [
        e - np.dot(st["n_atoms"], atom_e) for e, st in zip(energy, structures)
    ]
    dft_dict["energy"] = np.array(coh_energy)
    return dft_dict


def __set_weight_energy_data(energy, total_n_atoms, min_e=None):

    # todo: more appropriate procedure for finding weight values
    e_per_atom = energy / total_n_atoms
    if min_e is None:
        min_e = np.min(e_per_atom)
    e_th1, e_th2 = min_e * 0.75, min_e * 0.50

    weight_e = np.ones(len(energy))
    weight_e[e_per_atom > e_th1] = 0.5
    weight_e[e_per_atom > e_th2] = 0.3
    weight_e[e_per_atom > 0.0] = 0.1
    return weight_e


def __set_weight_force_data(force):

    log1 = np.log10(np.abs(force))
    w1 = np.array([pow(10, -v) for v in log1])
    weight_f = np.minimum(w1, np.ones(len(w1)))
    return weight_f


def __set_weight_stress_data(stress, weight_stress):

    log1 = np.log10(np.abs(stress))
    w1 = np.array([pow(5, -v) for v in log1])
    weight_s = np.minimum(w1, np.ones(len(w1))) * weight_stress
    return weight_s


def apply_weight_percentage(
    x,
    y,
    w,
    dft_dict,
    params_dict,
    first_indices,
    weight_stress=0.1,
    min_e=None,
):

    if "include_force" in dft_dict:
        include_force = dft_dict["include_force"]
    else:
        include_force = params_dict["include_force"]

    if include_force == False:
        include_stress = False
    else:
        include_stress = params_dict["include_stress"]

    ebegin, fbegin, sbegin = first_indices
    eend = ebegin + len(dft_dict["energy"])
    if include_force:
        fend = fbegin + len(dft_dict["force"])
        send = sbegin + len(dft_dict["stress"])

    energy = dft_dict["energy"]
    weight_e = __set_weight_energy_data(energy, dft_dict["total_n_atoms"], min_e=min_e)
    if "weight" in dft_dict:
        weight_e *= dft_dict["weight"]

    w[ebegin:eend] = weight_e
    y[ebegin:eend] = weight_e * energy

    x[ebegin:eend] *= weight_e[:, np.newaxis]
    """ numba version
    import pypolymlp.mlp_gen.numba_support as numba_support
    numba_support.mat_prod_vec(x[ebegin:eend], weight_e, axis=0)
    """

    if include_force:
        force = dft_dict["force"]
        weight_f = __set_weight_force_data(force)
        if "weight" in dft_dict:
            weight_f *= dft_dict["weight"]
        w[fbegin:fend] = weight_f
        y[fbegin:fend] = weight_f * force
        x[fbegin:fend] *= weight_f[:, np.newaxis]
        """ numba version
        numba_support.mat_prod_vec(x[fbegin:fend], weight_f, axis=0)
        """

        if include_stress:
            stress = dft_dict["stress"]
            if "weight" in dft_dict:
                weight_const = weight_stress * dft_dict["weight"]
            else:
                weight_const = weight_stress
            weight_s = __set_weight_stress_data(stress, weight_const)
            w[sbegin:send] = weight_s
            y[sbegin:send] = weight_s * stress
            x[sbegin:send] *= weight_s[:, np.newaxis]
            """ numba version
            numba_support.mat_prod_vec(x[sbegin:send], weight_s, axis=0)
            """
        else:
            x[sbegin:send, :] = 0.0
            y[sbegin:send] = 0.0
            w[sbegin:send] = 0.0
    return x, y, w


class Precondition:

    def __init__(
        self,
        reg_dict,
        dft_dict,
        params_dict,
        scales=None,
        weight_stress=0.1,
    ):

        self.x = reg_dict["x"]
        self.first_indices = reg_dict["first_indices"][0]  # single dataset
        self.ne, self.nf, self.ns = reg_dict["n_data"]
        self.n_data, self.n_features = self.x.shape

        self.reg_dict = reg_dict
        self.dft_dict = dft_dict
        self.params_dict = params_dict

        self.y = np.zeros(self.n_data)
        self.w = np.ones(self.n_data)
        self.scales = None

        self.dft_dict = apply_atomic_energy(self.dft_dict, self.params_dict)
        self.__apply_scales(scales=scales)
        self.__apply_weight(weight_stress=weight_stress)

        self.reg_dict["x"] = self.x
        self.reg_dict["y"] = self.y
        self.reg_dict["weight"] = self.w
        self.reg_dict["scales"] = self.scales

    def __apply_scales(self, scales=None):

        if scales is not None:
            self.scales = scales
        else:
            self.scales = np.std(self.x[: self.ne], axis=0)

        self.x /= self.scales
        """ correctly-working numba version
            numba_support.mat_prod_vec(self.x, np.reciprocal(self.scales),
                                       axis=1)
        """

    def __apply_weight(self, weight_stress=0.1):

        res = apply_weight_percentage(
            self.x,
            self.y,
            self.w,
            self.dft_dict,
            self.params_dict,
            self.first_indices,
            weight_stress=weight_stress,
        )
        self.x, self.y, self.w = res

    def print_data_shape(self, header="training data size"):

        print("  " + header + ":", self.x.shape)
        print("   - n (energy) =", self.ne)
        print("   - n (force)  =", self.nf)
        print("   - n (stress) =", self.ns)

    def get_scales(self):
        return self.scales

    def get_updated_regression_dict(self):
        return self.reg_dict
