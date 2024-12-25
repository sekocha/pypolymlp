"""Interfaces for setting DFT datasets."""

from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpStructure
from pypolymlp.core.utils import permute_atoms


def set_dataset_from_structures(
    structures: list[PolymlpStructure],
    energies: np.ndarray,
    forces: Optional[list[np.ndarray]] = None,
    stresses: Optional[np.ndarray] = None,
    element_order: bool = None,
) -> PolymlpDataDFT:
    """Return DFT dataset in PolymlpDataDFT."""

    assert len(structures) == len(energies)
    exist_force = True if forces is not None else False
    exist_stress = True if stresses is not None else False
    include_force = exist_force

    if forces is None:
        forces = [np.zeros((3, len(st.elements))) for st in structures]
    if stresses is None:
        stresses = [np.zeros((3, 3)) for _ in energies]

    forces_data, stresses_data, volumes_data = [], [], []
    for st, force_st, sigma in zip(structures, forces, stresses):
        if element_order is not None:
            st, force_st = permute_atoms(st, force_st, element_order)
        force_ravel = np.ravel(force_st, order="F")
        forces_data.extend(force_ravel)
        if sigma is not None:
            s = [
                sigma[0][0],
                sigma[1][1],
                sigma[2][2],
                sigma[0][1],
                sigma[1][2],
                sigma[2][0],
            ]
            stresses_data.extend(s)
        volumes_data.append(st.volume)

    total_n_atoms = np.array([sum(st.n_atoms) for st in structures])
    files = [st.name for st in structures]

    if element_order is None:
        # This part must be tested. In general, element_order is not None.
        elements_size = [len(st.n_atoms) for st in structures]
        elements = structures[np.argmax(elements_size)].elements
        elements = sorted(set(elements), key=elements.index)
    else:
        elements = element_order

    dft = PolymlpDataDFT(
        energies=np.array(energies),
        forces=np.array(forces_data),
        stresses=np.array(stresses_data),
        volumes=np.array(volumes_data),
        structures=structures,
        total_n_atoms=total_n_atoms,
        files=files,
        elements=elements,
        include_force=include_force,
        exist_force=exist_force,
        exist_stress=exist_stress,
        weight=1.0,
    )
    return dft
