# Molecular Dynamics Calculations

> **Note**: Requires `ASE` and `phonopy`.

Molecular dynamics calculations can be performed using algorithms implemented in ASE.
The ASE calculator for the polynomial MLP implemented in `pypolymlp` is used to compute the properties required for molecular dynamics.

## Using command line interface

The `pypolymlp-md` command performs molecular dynamics simulations in the NVT ensemble.

```shell
> pypolymlp-md --poscar POSCAR --pot polymlp.yaml --supercell_size 3 3 3 --temp 300 --n_eq 5000 --n_steps 20000
```

The available options are as follows:
```
  --pot [POT ...]       polymlp file.
  -p, --poscar POSCAR   Initial structure.
  --supercell_size SUPERCELL_SIZE SUPERCELL_SIZE SUPERCELL_SIZE
                        Diagonal supercell size.
  --thermostat {Langevin,Nose-Hoover}
                        Thermostat.
  --temp TEMP           Temperature.
  --time_step TIME_STEP
                        Time step (fs).
  --friction FRICTION   Friction in Langevin thermostat (1/fs).
  --ttime TTIME         Time step interact with thermostat in Langevin thermostat
                        (fs).
  --n_eq N_EQ           Number of equilibration steps.
  --n_steps N_STEPS     Number of steps.
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
