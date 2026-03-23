# Force Constant Calculation

> **Note**: Requires `symfc`, `phonopy`, and `phono3py`.

Second- and third-order force constants can be calculated using both the command-line interface and the Python API.
First, structures with random atomic displacements are generated, and the corresponding forces are computed using a given polynomial MLP.
Using the resulting displacement–force datasets, the second- and third-order force constants are then estimated via a projector-based efficient algorithm combined with linear regression [1].
The resulting force constants satisfy invariance with respect to index permutations, space group operations, and infinitesimal translations.

[1] "Projector-based efficient estimation of force constants",
A. Seko and A. Togo, Phys. Rev. B, **110**, 214302 (2024)
[[doi](https://doi.org/10.1103/PhysRevB.110.214302)]
[[arxiv](https://arxiv.org/abs/2403.03588)].


## Using command line interface

The `--force_constants` option activates the force constant calculations.

```shell
> pypolymlp-calc --force_constants --pot polymlp.yaml --poscar POSCAR --supercell 3 3 2 --fc_n_samples 100 --disp 0.001 --fc_orders 2 3
```

If a cutoff radius is introduced to evaluate FC3s, use "--cutoff" option as follows.
```shell
> pypolymlp-calc --force_constants --pot polymlp.yaml --poscar POSCAR --geometry_optimization --fc_n_samples 300 --fc_orders 2 3 --disp 0.001 --batch_size 100 --supercell 3 3 2 --cutoff 6
```

## Using Python API

### Force constant calculations using a POSCAR file
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

"""Force constant calculations.

symfc and phonopy is required.

Parameters
----------
unitcell: Unit cell of equilibrium structure.
supercell_matrix: Supercell matrix. (Only diagonal elements are valid.)
cutoff: Cutoff distance for force constant calculation.

disps: Displacements. shape=(n_str, 3, n_atom).
forces: Forces. shape=(n_str, 3, n_atom).
n_samples: Number of supercells sampled.
distance: Displacement magnitude in angstroms.
is_plusminus: Consider plus and minus displacements.
orders: Force constant orders.
batch_size: Batch size for force constant regression.
is_compact_fc: Generate compact forms of force constants.
use_mkl: Use MKL in symfc.
"""

polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_poscars("POSCAR")

# Geometry optimization (Optional)
polymlp.init_geometry_optimization(
    with_sym=True,
    relax_cell=False,
    relax_positions=True,
)
polymlp.run_geometry_optimization()

polymlp.init_fc(supercell_matrix=np.diag([3,3,2]), cutoff=None)
polymlp.run_fc(
    n_samples=100,
    distance=0.001,
    is_plusminus=False,
    orders=(2, 3),
    batch_size=100,
    is_compact_fc=True,
    use_mkl=True,
)

# fc2.hdf5 and fc3.hdf5 will be generated.
polymlp.save_fc()
```

### Force constant calculations using phono3py.yaml.xz
```python
from pypolymlp.calculator.fc import PolymlpFC

polyfc = PolymlpFC(
    phono3py_yaml='phono3py_params_wurtzite_AgI.yaml.xz',
    use_phonon_dataset=False,
    pot='polymlp.yaml',
)

"""optional"""
polyfc.run_geometry_optimization()

"""If not using sample(), displacements are read from phono3py.yaml.xz"""
polyfc.sample(n_samples=100, displacements=0.001, is_plusminus=False)

"""fc2.hdf5 and fc3.hdf5 will be generated."""
polyfc.run(batch_size=100)
```
