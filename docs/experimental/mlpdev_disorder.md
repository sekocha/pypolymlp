# MLP development for substitutional disordered states

```python
import numpy as np

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
from pypolymlp.postproc.disorder import PolymlpDisorder


occupancy = [
    [("Sr", 2/3), ("Cu", 1/3)],
    [("Cu", 2/3), ("Sr", 1/3)],
    [("O", 1.0)],
]

disorder = PolymlpDisorder(
    occupancy=occupancy,
    pot="polymlp.yaml",
    lattice="POSCAR",
    supercell_size=(6, 6, 6),
    verbose=True,
)
disorder.set_displaced_lattices(n_samples=5000, max_distance=1.0)
disorder.eval_random_properties(n_samples=1000)
polymlp = disorder.polymlp

polymlp.set_params(
    elements=["Sr", "Cu", "O"],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[8, 8],
    gaussian_params2=[0.0, 7.0, 12],
    atomic_energy=[0, 0, 0],
    reg_alpha_params=(-5, 5, 30),
)

energies, forces, _ = disorder.properties
polymlp.set_datasets_structures_autodiv(
    structures=disorder.structures,
    energies=energies,
    forces=forces,
    stresses=None,
)

polymlp.fit(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml.disorder")
polymlp.estimate_error(log_energy=True, verbose=True)
```
