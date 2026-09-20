# Cluster Expansion in Substitutional (Alloy) Systems

> **Note**: Requires [`pyclupan`](https://github.com/sekocha/pyclupan).

The cluster expansion (CE) method describes the energetics of substitutional alloy systems using a generalized Ising model.
By combining polynomial MLPs for alloy systems with the CE method, CE models can be constructed, enabling the systematic calculation of the energies and formation energies of ordered alloy structures.
Currently, CE model construction and prediction using `pypolymlp` in combination with `pyclupan` are available through the Python API of `pyclupan`.


An example of CE model estimation and formation energy calculation using a polynomial MLP for the Ag–Au binary alloy system is shown below.

```python
import numpy as np
from pyclupan.api.pyclupan import Pyclupan

element_strings = ["Ag", "Au"]
elements = [[0, 1]]
poscar = "POSCAR"

max_supercell_size_full = 6
max_supercell_size_partial = 10
cluster_max_order = 4
cluster_cutoffs = (6.0, 6.0, 6.0)

pyclupan = Pyclupan(verbose=True)
pyclupan.set_lattice_and_elements(
    elements=elements,
    element_strings=element_strings,
    poscar=poscar,
)
pyclupan.enum_derivatives(
    min_supercell_size=2,
    max_supercell_size=max_supercell_size_full,
)
pyclupan.enum_derivatives(
    min_supercell_size=max_supercell_size_full + 1,
    max_supercell_size=max_supercell_size_partial,
    n_samples=100,
)

pyclupan.eval_energies(pot="polymlp.yaml")
pyclupan.enum_cluster(max_order=cluster_max_order, cutoffs=cluster_cutoffs)
pyclupan.eval_cluster_functions()
pyclupan.eval_ecis()

pyclupan.enum_derivatives(
    min_supercell_size=max_supercell_size_partial + 1,
    max_supercell_size=12,
)

pyclupan.eval_ce_energies()
pyclupan.eval_ce_formation_energies()
```

In this example, the following calculations using the CE method are performed for the Ag–Au binary alloy system, using the FCC lattice specified by `POSCAR` as the input lattice and the polynomial MLP specified by `polymlp.yaml`.

1. Enumerate derivative structures (symmetrically non-equivalent alloy configurations).
   - Enumerate all derivative structures with up to six-fold expansions of the primitive FCC lattice.
   - For 7–10-fold expansions of the primitive lattice, enumerate all derivative structures and randomly sample 100 structures for each expansion size. The sampled structures are used for the energy calculations in the training process.
   - The remaining structures are reserved for energy predictions using the CE model.

2. Evaluate the energies of the enumerated derivative structures using the polynomial MLP.

3. Enumerate symmetrically non-equivalent clusters on the lattice.

4. Calculate the cluster functions (correlation functions) of the enumerated clusters for the derivative structures.

5. Estimate the effective cluster interactions (ECIs) using the energies and cluster functions of the derivative structures.
   - The ECIs are estimated using the Lasso approach to determine the optimal CE model.

6. Enumerate all derivative structures with 11–12-fold expansions of the primitive lattice.

7. Calculate the energies and formation energies of all derivative structures with up to 12-fold expansions using the CE model.
