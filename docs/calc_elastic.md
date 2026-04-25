# Elastic Constant Calculation

The `--elastic` option activates the calculation of elastic constants for a structure using a given polynomial MLP.

## Using command-line interface
```shell
> pypolymlp-calc --elastic --poscar POSCAR --pot polymlp.yaml
```

## Using Python API
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

"""Run elastic constant calculations.

pymatgen is required.

Returns
-------
elastic_constants: Elastic constants in GPa. shape=(6,6).
"""
polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_poscars("POSCAR")
elastic_constants = polymlp.run_elastic_constants()
polymlp.write_elastic_constants(filename="polymlp_elastic.yaml")

# attributes:
# elastic_constants = polymlp.elastic_constants
```
