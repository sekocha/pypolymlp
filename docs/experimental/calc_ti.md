# Thermodynamic integration using MD
`ASE` and `phonopy` are required.

## Using command line interface
Molecular dynamics calculations are performed for structure `POSCAR` using polymomial MLP `polymlp.yaml`. If `--fc2_path` option is provided, a reference state is automatically searched in `--fc2_path` directory. In pypolymlp, the effective force constants at the lowest temperature in `--fc2_path` directory is used as the reference.

```shell
pypolymlp-md --ti --poscar POSCAR --pot polymlp.yamls --supercell_size 3 3 3 --temp 500 --n_eq 5000 --n_steps 40000 --n_samples 20 --fc2_path ./sscha --max_alpha 0.98
```

## Using Python API

```python
from pypolymlp.api.pypolymlp_md import PypolymlpMD

"""
Parameters
----------
thermostat: Thermostat.
n_alphas: Number of sample points for thermodynamic integration
          using Gaussian quadrature.
temperature : int
    Target temperature (K).
time_step : float
    Time step for MD (fs).
ttime : float
    Timescale of the Nose-Hoover thermostat (fs).
friction : float
    Friction coefficient for Langevin thermostat (1/fs).
n_eq : int
    Number of equilibration steps.
n_steps : int
    Number of production steps.
heat_capacity: bool
    Calculate heat capacity.
"""

md = PypolymlpMD(verbose=True)
md.load_poscar("POSCAR")
md.set_supercell([5, 5, 3])

md.set_ase_calculator_with_fc2(pot="polymlp.yaml", fc2hdf5="fc2.hdf5")
md.run_thermodynamic_integration(
    thermostat="Langevin",
    temperature=300.0,
    time_step=1.0,
    friction=0.01,
    n_eq=5000,
    n_steps=40000,
    heat_capacity=False,
)
md.save_thermodynamic_integration_yaml(filename="polymlp_ti.yaml")
```
or
```python
from pypolymlp.api.pypolymlp_md import run_thermodynamic_integration

md = run_thermodynamic_integration(
    pot="polymlp.yaml",
    poscar="POSCAR",
    supercell_size=(5, 5, 3),
    fc2hdf5="fc2.hdf5",
    temperature=300.0,
    n_alphas=15,
    n_eq=5000,
    n_steps=40000,
    heat_capacity=False,
    filename="polymlp_ti.yaml",
)
```
