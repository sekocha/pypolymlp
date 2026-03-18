# Systematic Generation of Input Files for Polynomial MLP Development

> Note: Requires version 0.19.1 or later.

## Using Command-Line Interface
```shell
# For elemental systems
pypolymlp-utils --generate_models --generate_models_elements Be

# Enumeration with hybrid models for elemental systems
pypolymlp-utils --generate_models --generate_models_elements Si --enable_hybrid

# For binary alloy systems
pypolymlp-utils --generate_models --generate_models_elements Ag Au
pypolymlp-utils --generate_models --generate_models_system Ag-Au

# For ternary alloy systems
pypolymlp-utils --generate_models --generate_models_elements Cu Ag Au
pypolymlp-utils --generate_models --generate_models_system Cu-Ag-Au
```

## Using Python API

- **Enumeration with single models**
```python
from pypolymlp.api.pypolymlp_utils import PypolymlpUtils

elements = ["Ag", "Au"]
utils = PypolymlpUtils(verbose=True)
utils.enumerate_models(path="polymlps")
```

- **Enumeration with hybrid models**
```python
from pypolymlp.api.pypolymlp_utils import PypolymlpUtils

elements = ["Ag", "Au"]
utils = PypolymlpUtils(verbose=True)
utils.enumerate_models(path="polymlps", hybrid=True)
```

- **Parameter settings**
```python
from pypolymlp.api.pypolymlp_utils import PypolymlpUtils

elements = ["Ag", "Au"]
utils = PypolymlpUtils(verbose=True)
utils.enumerate_models(
    model_types=(2, 3, 4),
    maxps=(2,),
    gtinv=True,
    gtinv_order_ub=3,
    gtinv_maxl_ub=(16, 12),
    gtinv_maxl_int=(8, 4),
    include_force=True,
    include_stress=True,
    regression_alpha=(-4, 1, 6),
    path="polymlps",
    hybrid=True,
)
```
See [Notes on Polymlp Parameters](mlpdev_params.md) for details on the input parameters.
