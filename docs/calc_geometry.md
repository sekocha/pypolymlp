# Local Geometry Optimization

Local geometry optimization for a structure can be performed using the command-line interface and the Python API.

## Using command-line interface

The `--geometry_optimization` option activates local geometry optimization for a structure specified in `POSCAR`, using a given polynomial MLP.
The options `--no_symmetry`, `--fix_cell`, `--fix_volume`, and `--fix_atom` control the degrees of freedom during optimization.
The available options are as follows:

```
  --pressure PRESSURE   Pressure (in GPa)
  --no_symmetry         Ignore symmetric properties in geometry optimization
  --fix_cell            Fix cell shape and volume in geometry optimization
  --fix_volume          Fix cell volume in geometry optimization
  --fix_atom            Fix atomic positions in geometry optimization
  --method {BFGS,CG,L-BFGS-B,SLSQP}
                        Algorithm for geometry optimization
```

Some examples of geometry optimizations are shown below:

- **Full geometry optimization without using the symmetry of the initial structure**

```shell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml --no_symmetry
```
- **Full geometry optimization with symmetric constraints of the initial structure**

```shell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml
```
- **Geometry optimization with fixed cell shape and volume**

```shell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml --fix_cell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml --fix_cell --no_symmetry
```
- **Geometry optimization for cell shape and volume only**

```shell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.yaml --fix_atom
```

## Using Python API

Geometry optimizations can also be performed using the Python API.
Before using `run_geometry_optimization`, set the initial structure and initialize the optimization with `init_geometry_optimization`.

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
