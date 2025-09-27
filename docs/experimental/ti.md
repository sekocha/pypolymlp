## Thermodynamic integration using MD
(Requirement: ASE, phonopy)
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
    n_steps=20000,
    heat_capacity=False,
)
md.save_thermodynamic_integration_yaml(filename="polymlp_ti.yaml")
```

or
```python
from pypolymlp.api.pypolymlp_md import run_thermodynamic_integration

md = run_thermodynamic_integration(
    pot="polymlp.lammps",
    poscar="POSCAR",
    supercell_size=(5, 5, 3),
    fc2hdf5="fc2.hdf5",
    temperature=300.0,
    n_alphas=15,
    n_eq=5000,
    n_steps=20000,
    heat_capacity=False,
    filename="polymlp_ti.yaml",
)
```
