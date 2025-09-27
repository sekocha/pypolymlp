# Molecular dynamics calculations
`ASE` and `phonopy` are required.

## Using command line interface

```shell
> pypolymlp-md --poscar POSCAR --pot polymlp.yaml --supercell_size 3 3 3 --temp 300 --n_eq 5000 --n_steps 20000
```

## Using Python API
```python
from pypolymlp.api.pypolymlp_md import PypolymlpMD

"""
Parameters
----------
temperature : int
    Target temperature (K).
time_step : float
    Time step for MD (fs).
friction : float
    Friction coefficient for Langevin thermostat (1/fs).
n_eq : int
    Number of equilibration steps.
n_steps : int
    Number of production steps.
"""

md = PypolymlpMD(verbose=True)
md.load_poscar("POSCAR")
md.set_supercell([4, 4, 3])

md.set_ase_calculator(pot="polymlp.yaml")
md.run_Langevin(
    temperature=300,
    time_step=1.0,
    friction=0.01,
    n_eq=5000,
    n_steps=20000,
)
md.save_yaml(filename="polymlp_md.yaml")
```
