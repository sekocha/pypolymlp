# Systematic Generation of Input Files for Polynomial MLP Development

```python
from pypolymlp.api.pypolymlp_utils import PypolymlpUtils

elements = ["Ag", "Au"]

utils = PypolymlpUtils(verbose=True)
utils.enumerate_models(path="polymlps")
```
