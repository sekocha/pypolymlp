"""Functions for applying weights"""

from typing import Optional

import numpy as np

from pypolymlp.core.dataset import Dataset
from pypolymlp.mlp_dev.core.utils import get_observation_slices


def _set_weight_energy_data(energy, total_n_atoms, min_e: Optional[float] = None):
    """Define weights to energy data."""
    # TODO: more appropriate procedure for finding weight values
    e_per_atom = energy / total_n_atoms
    if min_e is None:
        min_e = np.min(e_per_atom)
    e_th1, e_th2 = min_e * 0.75, min_e * 0.50

    weight_e = np.ones(len(energy))
    weight_e[e_per_atom > e_th1] = 0.5
    weight_e[e_per_atom > e_th2] = 0.3
    weight_e[e_per_atom > 0.0] = 0.1
    return weight_e


def _set_weight_force_data(
    forces: np.ndarray,
    weight_force: float,
    tol: float = 1e-12,
):
    """Define weights to force data."""
    weight_f = np.abs(forces)
    weight_f[weight_f < tol] = tol
    weight_f = np.reciprocal(weight_f)
    weight_f[weight_f > 1.0] = 1.0
    weight_f *= weight_force
    return weight_f


def _set_weight_stress_data(
    stress: np.ndarray,
    weight_stress: float,
    tol: float = 1e-12,
):
    """Define weights to stress data."""
    nonzero = np.abs(stress) > tol
    log1 = np.ones(len(stress)) * np.log10(tol)
    log1[nonzero] = np.log10(np.abs(stress)[nonzero])

    weight_s = np.power(5, -log1)
    weight_s[weight_s > 1.0] = 1.0
    weight_s *= weight_stress
    return weight_s


def _weight_energy(
    dataset: Dataset,
    use_auto_weight: bool = True,
    min_e: Optional[float] = None,
):
    """Get weights for energy."""
    if use_auto_weight:
        weight_e = _set_weight_energy_data(
            dataset.energies,
            dataset.total_n_atoms,
            min_e=min_e,
        )
        weight_e *= dataset.weight
    else:
        weight_e = np.full(len(dataset.energies), dataset.weight)
    return weight_e


def _weight_force(
    dataset: Dataset,
    use_auto_weight: bool = True,
    weight_force: float = 1.0,
):
    """Get weights for force."""
    weight_const = weight_force * dataset.weight
    if use_auto_weight:
        weight_f = _set_weight_force_data(dataset.forces, weight_const)
    else:
        weight_f = np.full(len(dataset.forces), weight_const)
    return weight_f


def _weight_stress(
    dataset: Dataset,
    use_auto_weight: bool = True,
    weight_stress: float = 0.1,
):
    """Get weights for stress."""
    weight_const = weight_stress * dataset.weight
    if use_auto_weight:
        weight_s = _set_weight_stress_data(dataset.stresses, weight_const)
    else:
        weight_s = np.full(len(dataset.stresses), weight_const)
    return weight_s


def _assign_weights(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    begin: int,
    end: int,
    weights: np.ndarray,
):
    """Assign weights."""
    w[begin:end] = weights
    y[begin:end] *= weights
    x[begin:end] *= weights[:, np.newaxis]
    return x, y, w


def apply_weights(
    x: np.ndarray,
    y: np.ndarray,
    dataset: Dataset,
    first_indices: list,
    use_auto_weight: bool = True,
    weight_force: float = 1.0,
    weight_stress: float = 0.1,
    min_e: Optional[float] = None,
):
    """Apply weights to data."""
    if x.shape[0] != y.shape[0]:
        raise RuntimeError("Shapes of X and y not consistent.")

    include_force = dataset.include_force
    include_stress = dataset.include_stress

    slices = get_observation_slices(dataset, first_indices)
    ebegin, eend, fbegin, fend, sbegin, send = slices

    w = np.ones(y.shape[0])

    weight_e = _weight_energy(dataset, use_auto_weight, min_e)
    x, y, w = _assign_weights(x, y, w, ebegin, eend, weight_e)

    if include_force:
        weight_f = _weight_force(dataset, use_auto_weight, weight_force)
        x, y, w = _assign_weights(x, y, w, fbegin, fend, weight_f)

    if include_stress:
        weight_s = _weight_stress(dataset, use_auto_weight, weight_stress)
        x, y, w = _assign_weights(x, y, w, sbegin, send, weight_s)

    if include_force and not include_stress:
        x[sbegin:send, :] = 0.0
        y[sbegin:send] = 0.0
        w[sbegin:send] = 0.0
    return x, y, w


def apply_weights_overwrite(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    dataset: Dataset,
    first_indices: list,
    use_auto_weight: bool = True,
    weight_force: float = 1.0,
    weight_stress: float = 0.1,
    min_e: Optional[float] = None,
):
    """Apply weights to data."""
    include_force = dataset.include_force
    include_stress = dataset.include_stress

    slices = get_observation_slices(dataset, first_indices)
    ebegin, eend, fbegin, fend, sbegin, send = slices

    weight_e = _weight_energy(dataset, use_auto_weight, min_e)
    w[ebegin:eend] = weight_e
    y[ebegin:eend] = weight_e * dataset.energies
    x[ebegin:eend] *= weight_e[:, np.newaxis]

    if include_force:
        weight_f = _weight_force(dataset, use_auto_weight, weight_force)
        w[fbegin:fend] = weight_f
        y[fbegin:fend] = weight_f * dataset.forces
        x[fbegin:fend] *= weight_f[:, np.newaxis]

    if include_stress:
        weight_s = _weight_stress(dataset, use_auto_weight, weight_stress)
        w[sbegin:send] = weight_s
        y[sbegin:send] = weight_s * dataset.stresses
        x[sbegin:send] *= weight_s[:, np.newaxis]

    if include_force and not include_stress:
        x[sbegin:send, :] = 0.0
        y[sbegin:send] = 0.0
        w[sbegin:send] = 0.0
    return x, y, w
