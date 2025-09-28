# Polynomial invariant calculation

## Using command line interface
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
