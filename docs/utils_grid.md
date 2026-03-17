# Systematic Generation of Input Files for Polynomial MLP Development

- **Command-line interface**
```shell
# For elemental systems
pypolymlp-utils --generate_models --generate_models_elements Be

# For binary alloy systems
pypolymlp-utils --generate_models --generate_models_elements Ag Au
pypolymlp-utils --generate_models --generate_models_system Ag-Au

# For ternary alloy systems
pypolymlp-utils --generate_models --generate_models_elements Cu Ag Au
pypolymlp-utils --generate_models --generate_models_system Cu-Ag-Au
```

- **Python API**
```python
from pypolymlp.api.pypolymlp_utils import PypolymlpUtils

elements = ["Ag", "Au"]

utils = PypolymlpUtils(verbose=True)
utils.enumerate_models(path="polymlps")
```
