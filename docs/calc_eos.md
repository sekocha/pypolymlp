# Equation of States (EOS) Calculation

> **Note**: Requires `phonopy`.

The equation of state for a structure specified in `POSCAR` is calculated using the given polynomial MLP.

## Using command-line interface

The `--eos` option activates the equation-of-state (EOS) calculation.

```shell
> pypolymlp-calc --eos --poscar POSCAR --pot polymlp.yaml
```

## Using Python API
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

"""Run EOS calculations.

phonopy is required if eos_fit = True.

Parameters
----------
structure: Equilibrium structure.
eps_min: Lower bound of volume change.
eps_max: Upper bound of volume change.
eps_step: Interval of volume change.
fine_grid: Use a fine grid around equilibrium structure.
eos_fit: Fit vinet EOS curve using volume-energy data.

volumes = np.arange(eps_min, eps_max, eps_step) * eq_volume
"""

polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_structures_from_files(poscars='POSCAR')
polymlp.run_eos(
    eps_min=0.7,
    eps_max=2.0,
    eps_step=0.03,
    fine_grid=True,
    eos_fit=True,
)
polymlp.write_eos()
energy0, volume0, bulk_modulus = polymlp.eos_fit_data
```
