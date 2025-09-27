# Local geometry optimization
## Using command line interface
```shell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --no_symmetry
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --fix_cell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --fix_atom
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --method CG
```
## Using Python API
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

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
```
