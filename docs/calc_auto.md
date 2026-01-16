# Systematic property calculations for repository construction

## 1. Grid search of polynomial MLPs
## 2. Computational cost estimation of polynomial MLPs

Computational costs for multiple polynomial MLPs are estimated as follows.
```shell
> pypolymlp-utils --calc_cost -d ./polymlp*
```

## 3. Search of optimal MLPs on convex hull

Optimal MLPs on the convex hull can be found as follows.
```shell
pypolymlp-utils --find_optimal ./polymlp* --key test-disp1
```
The error values that are used for the error metric are required to be specified by `--key` option.


## 4. Systematic property calculations using optimal MLPs
```shell
> cd polymlp-optimal1
> python3 run_auto.py
```
`run_auto.py` is a Python script written by using `PypolymlpAutoCalc`.

```python
import numpy as np
from pypolymlp.calculator.auto.pypolymlp_autocalc import PypolymlpAutoCalc

calc = PypolymlpAutoCalc(pot="polymlp.yaml", verbose=True)
calc.load_structures()
calc.run()
```
