# Elastic constant calculation

`pymatgen` is required for elastic constant calculation.

## Using command line interface
```shell
> pypolymlp-calc --elastic --poscar POSCAR --pot polymlp.lammps
```

## Using Python API
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.lammps")
polymlp.load_poscars("POSCAR")
polymlp.run_elastic_constants()
polymlp.write_elastic_constants(filename="polymlp_elastic.yaml")
elastic_constants = polymlp.elastic_constants
```
