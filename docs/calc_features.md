# Polynomial Invariant Calculation

Polynomial invariants used in polynomial MLPs can be calculated using the command-line interface and the Python API.

## Using command line interface

The `--features` option enables polynomial invariant calculations.
Input parameters can be provided from two types of files: an input file used for developing an MLP, or a file containing a developed MLP.
The input file used for developing the MLP can be specified with the `-i` option.
The polynomial MLP file can be specified with the `--pot` option.
Multiple structures can be specified using the `--poscars` option.

```shell
> pypolymlp-calc --features --pot polymlp.yaml --poscars */POSCAR
> pypolymlp-calc --features -i polymlp.in --poscars */POSCAR
```

## Using Python API

### Polynomial invariant calculation using `polymlp.in`
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

"""Compute features.

Parameters
----------
structures: Structures for computing polynomial invariants
develop_infile: A pypolymlp input file for developing MLP.

Return
------
features: Structural features. shape=(n_str, n_features)
          if both features_force and features_stress are False.
"""

polymlp = PypolymlpCalc(require_mlp=False)
polymlp.load_structures_from_files(poscars=["POSCAR1", "POSCAR2", "POSCAR3"])
features = polymlp.run_features(
    develop_infile="polymlp.in",
    features_force=False,
    features_stress=False,
)
polymlp.save_features()
```

### Polynomial invariant calculation using `polymlp.yaml`
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_structures_from_files(poscars=["POSCAR1", "POSCAR2", "POSCAR3"])
features = polymlp.run_features(
    features_force=False,
    features_stress=False,
)
polymlp.save_features()
```
