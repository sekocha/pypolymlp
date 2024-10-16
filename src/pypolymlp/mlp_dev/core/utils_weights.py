"""Functions for applying weights"""

from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams


def _set_weight_energy_data(energy, total_n_atoms, min_e=None):

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


def _set_weight_force_data(force):

    log1 = np.log10(np.abs(force))
    w1 = np.array([pow(10, -v) for v in log1])
    weight_f = np.minimum(w1, np.ones(len(w1)))
    return weight_f


def _set_weight_stress_data(stress, weight_stress):

    log1 = np.log10(np.abs(stress))
    w1 = np.array([pow(5, -v) for v in log1])
    weight_s = np.minimum(w1, np.ones(len(w1))) * weight_stress
    return weight_s


def apply_weight_percentage(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    dft: PolymlpDataDFT,
    params: PolymlpParams,
    first_indices: list,
    weight_stress: float = 0.1,
    min_e: Optional[float] = None,
):

    include_force = dft.include_force
    if include_force == False:
        include_stress = False
    else:
        include_stress = params.include_stress

    ebegin, fbegin, sbegin = first_indices
    eend = ebegin + len(dft.energies)
    if include_force:
        fend = fbegin + len(dft.forces)
        send = sbegin + len(dft.stresses)

    weight_e = _set_weight_energy_data(dft.energies, dft.total_n_atoms, min_e=min_e)
    weight_e *= dft.weight

    w[ebegin:eend] = weight_e
    y[ebegin:eend] = weight_e * dft.energies

    x[ebegin:eend] *= weight_e[:, np.newaxis]

    if include_force:
        weight_f = _set_weight_force_data(dft.forces)
        weight_f *= dft.weight
        w[fbegin:fend] = weight_f
        y[fbegin:fend] = weight_f * dft.forces
        x[fbegin:fend] *= weight_f[:, np.newaxis]

        if include_stress:
            weight_const = weight_stress * dft.weight
            weight_s = _set_weight_stress_data(dft.stresses, weight_const)
            w[sbegin:send] = weight_s
            y[sbegin:send] = weight_s * dft.stresses
            x[sbegin:send] *= weight_s[:, np.newaxis]
        else:
            x[sbegin:send, :] = 0.0
            y[sbegin:send] = 0.0
            w[sbegin:send] = 0.0
    return x, y, w
