"""Class for computing stress tensor in SSCHA."""

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.calculator.utils.fc_utils import eval_properties_fc2
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.displacements import generate_random_const_displacements
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp


def compute_harmonic_stress(
    structure: PolymlpStructure,
    fc2: np.ndarray,
    n_samples: int = 100,
):
    """Compute harmonic stress."""
    disps, sample_structures = generate_random_const_displacements(
        structure,
        n_samples=n_samples,
        displacements=0.01,
    )
    disps = disps.transpose((0, 2, 1)).reshape((n_samples, -1))
    print(disps)
    energies, forces = [], []
    for d in disps:
        e, f, _ = eval_properties_fc2(fc2, d)
        energies.append(e)
        forces.append(f)

    polymlp = Pypolymlp(verbose=False)
    elements = np.unique(structure.elements)
    polymlp.set_params(
        elements=elements,
        cutoff=6.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        n_gaussians=8,
    )
    polymlp.set_datasets_structures_autodiv(
        structures=sample_structures,
        energies=energies,
        forces=forces,
        stresses=None,
        train_ratio=0.9,
    )
    polymlp.run()

    params = polymlp.summary.params
    coeffs = polymlp.summary.scaled_coeffs
    calc = PypolymlpCalc(params=params, coeffs=coeffs)
    e_pred, f_pred, stress = calc.eval(sample_structures)
    for f1, f2 in zip(forces, f_pred):
        print(f1[0])
        print(f2[0])
    _, _, stress = calc.eval(structure)
    return stress[0]
