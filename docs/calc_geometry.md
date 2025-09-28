# Local geometry optimization
## Using command line interface
```shell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml --no_symmetry
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml --fix_cell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml --fix_atom
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml --method CG
```
## Using Python API
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

"""
symfc is required if with_sym = True.

Parameters
----------
init_str: Initial structure.
with_sym: Consider symmetry.
relax_cell: Relax cell.
relax_volume: Relax volume.
relax_positions: Relax atomic positions.
pressure: Pressure in GPa.

(in run_geometry_optimization)
method: Optimization method, CG, BFGS, L-BFGS-B, or SLSQP.
        If relax_volume = False, SLSQP is automatically used.
gtol: Tolerance for gradients.
maxiter: Maximum iteration in scipy optimization.
c1: c1 parameter in scipy optimization.
c2: c2 parameter in scipy optimization.

Returns
-------
energy: Energy at the final iteration.
n_iter: Number of iterations required for convergence.
success: Return True if optimization finished successfully.
"""

polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_poscars("POSCAR")

polymlp.init_geometry_optimization(
    with_sym=True,
    relax_cell=True,
    relax_positions=True,
    relax_volume=True,
    pressure=0.0,
)
e0, n_iter, success = polymlp.run_geometry_optimization()
if success:
    polymlp.save_poscars(filename="POSCAR_CONVERGE")

# Converged structure
converged_structure = polymlp.converged_structure
```
