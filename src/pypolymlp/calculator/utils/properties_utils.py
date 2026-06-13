"""Utility functions for calculating properties."""

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoGPa


def find_active_atoms(
    structures: list[PolymlpStructure],
    elements_string: list[str],
):
    """Reconstruct structures only using active atoms."""
    # TODO: Implement active atoms for spin-configuration.
    if len(elements_string) != len(np.unique(elements_string)):
        raise RuntimeError("Not available for system with spin configurations.")

    structures_active = []
    active_atoms_all = []
    active_bools = []
    for st in structures:
        active_atoms = np.array(
            [i for i, ele in enumerate(st.elements) if ele in elements_string]
        )
        types = np.array([elements_string.index(st.elements[i]) for i in active_atoms])
        n_atoms = [np.count_nonzero(types == i) for i in range(len(elements_string))]

        if len(active_atoms) > 0:
            st_active = PolymlpStructure(
                axis=st.axis,
                positions=st.positions[:, active_atoms],
                n_atoms=n_atoms,
                elements=np.array(st.elements)[active_atoms],
                types=types,
            )
            structures_active.append(st_active)
            active_atoms_all.append(active_atoms)
            active_bools.append(True)
        else:
            active_bools.append(False)

    return structures_active, active_atoms_all, np.array(active_bools)


def convert_stresses_in_gpa(stresses: np.ndarray, structures: list[PolymlpStructure]):
    """Calculate stress tensor values in GPa."""
    if not isinstance(stresses, np.ndarray):
        stresses = np.array(stresses)
    volumes = np.array([st.volume for st in structures])
    stresses_gpa = np.zeros(stresses.shape)
    for i in range(6):
        stresses_gpa[:, i] = stresses[:, i] / volumes * EVtoGPa
    return stresses_gpa
