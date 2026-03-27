# Lattice Thermal Conductivity Calculation

> **Note**: Requires `symfc`, `phonopy`, and `phono3py`.

As an option in force constant calculations, lattice thermal conductivities can be computed using the standard procedure implemented in `phono3py`.
To calculate lattice thermal conductivity, second- and third-order force constants must be computed in advance.

See also [Force Constant Calculation](calc_fc.md).

## Using command line interface

Both the `--force_constants` and `--run_ltc` options enable lattice thermal conductivity calculations.
Once the force constant calculations are completed, the lattice thermal conductivity calculations are executed successively.

```shell
> pypolymlp-calc --force_constants --pot polymlp.yaml --poscar POSCAR --supercell 3 3 2 --fc_n_samples 100 --disp 0.01 --fc_orders 2 3 --run_ltc --ltc_mesh 19 19 19
```
The available options for force constant calculation can be found in [Force Constant Calculation](calc_fc.md).

## Using Python API

```python
import numpy as np
import phono3py
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

"""
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

ph3 = phono3py.load(
    unitcell_filename="POSCAR",
    supercell_matrix=(3, 3, 2),
    primitive_matrix="auto",
    log_level=True,
)
ph3.mesh_numbers = (19, 19, 19)
ph3.init_phph_interaction()
ph3.run_thermal_conductivity(
    temperatures=range(0, 1001, 10),
    write_kappa=True,
)
```
