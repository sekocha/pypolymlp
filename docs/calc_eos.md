# Equation of states (EOS) calculation
`phonopy` is required for EOS calculation.

## Using command line interface
```shell
> pypolymlp-calc --eos --poscar POSCAR --pot polymlp.yaml
```

## Using Python API
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

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
