# Systematic Generation of Input Files for Polynomial MLP Development

```python
from pypolymlp.utils.grid_search.pypolymlp_gridsearch import PolymlpGridSearch

grid1 = PolymlpGridSearch(elements=["Al"], verbose=True)
grid1.set_params(
    cutoffs=(6.0, 8.0),
    nums_gaussians=(7, 10),
    model_types=(3, 4),
    gtinv=True,
    gtinv_order_ub=3,
    gtinv_maxl_ub=(12, 8),
    gtinv_maxl_int=(4, 4),
    include_force=True,
    include_stress=True,
    regression_alpha=(-4, 1, 6),
)
grid1.run()
grid1.save_models(path="./polymlps")
```
