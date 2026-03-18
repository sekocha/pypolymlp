# Convex Hull (Pareto-Optimal) MLP Search

> Note: Requires version 0.19.1 or later.

## 1. Systematic Generation of Input Files for Polynomial MLP Development

### Using Command-Line Interface
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

### Using Python API

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


## 2. Systematic MLP development

Using the generated input files, MLPs are then systematically evaluated.
For details on developing each MLP, see [MLP development (Command-line)](mlpdev_command.md).

## 3. Computational Cost Estimation of MLPs

Once multiple MLPs with various models have been systematically evaluated, the computational cost of calculating the properties must also be assessed in order to evaluate the trade-off between computational cost and accuracy.

`calc_cost` option generates a file 'polymlp_cost.yaml', which is required for finding optimal MLPs.

1. Single polynomial MLP

```shell
> pypolymlp-utils --calc_cost --pot polymlp.yaml

# hybrid polynomial MLP
> pypolymlp-utils --calc_cost --pot polymlp.yaml*
```
2. Multiple polynomial MLPs

Consider the following file and directory structures, which can be found in `examples/utils/mlp_opt`.

```shell
> ls Ag-Au
polymlp-00001/  polymlp-00003/  polymlp-00005/  polymlp-00007/  polymlp-00009/
polymlp-00002/  polymlp-00004/  polymlp-00006/  polymlp-00008/

> ls Ag-Au/polymlp-00*
test/polymlp-00001:
polymlp.yaml

test/polymlp-00002:
polymlp.yaml

test/polymlp-00003:
polymlp.yaml

test/polymlp-00004:
polymlp.yaml

test/polymlp-00005:
polymlp.yaml

test/polymlp-00006:
polymlp.yaml

test/polymlp-00007:
polymlp.yaml

test/polymlp-00008:
polymlp.yaml

test/polymlp-00009:
polymlp.yaml
```
2. Multiple polynomial MLPs

Consider the following file and directory structures, which can be found in `examples/utils/mlp_opt`.

```shell
> ls Ag-Au
polymlp-00001/  polymlp-00003/  polymlp-00005/  polymlp-00007/  polymlp-00009/
polymlp-00002/  polymlp-00004/  polymlp-00006/  polymlp-00008/

> ls Ag-Au/polymlp-00*
test/polymlp-00001:
polymlp.yaml

test/polymlp-00002:
polymlp.yaml

test/polymlp-00003:
polymlp.yaml

test/polymlp-00004:
polymlp.yaml

test/polymlp-00005:
polymlp.yaml

test/polymlp-00006:
polymlp.yaml

test/polymlp-00007:
polymlp.yaml

test/polymlp-00008:
polymlp.yaml

test/polymlp-00009:
polymlp.yaml
```
2. Multiple polynomial MLPs

Consider the following file and directory structures, which can be found in `examples/utils/mlp_opt`.

```shell
> ls Ag-Au
polymlp-00001/  polymlp-00003/  polymlp-00005/  polymlp-00007/  polymlp-00009/
polymlp-00002/  polymlp-00004/  polymlp-00006/  polymlp-00008/

> ls Ag-Au/polymlp-00*
test/polymlp-00001:
polymlp.yaml

test/polymlp-00002:
polymlp.yaml

test/polymlp-00003:
polymlp.yaml

test/polymlp-00004:
polymlp.yaml

test/polymlp-00005:
polymlp.yaml

test/polymlp-00006:
polymlp.yaml

test/polymlp-00007:
polymlp.yaml

test/polymlp-00008:
polymlp.yaml

test/polymlp-00009:
polymlp.yaml
```

In this case, computational costs for multiple polynomial MLPs can be estimated as follows.
```shell
> pypolymlp-utils --calc_cost -d Ag-Au/polymlp-00*
```

# 4. Enumeration of optimal MLPs on convex hull
Consider the following file and directory structures, which can be found in `examples/utils/mlp_opt`.

```shell
> ls Ag-Au
polymlp-00001/  polymlp-00003/  polymlp-00005/  polymlp-00007/  polymlp-00009/
polymlp-00002/  polymlp-00004/  polymlp-00006/  polymlp-00008/

Ag-Au/polymlp-00001:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00002:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00003:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00004:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00005:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00006:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00007:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00008:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00009:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml
```
Files `polymlp_error.yaml` and `polymlp_cost.yaml` are needed for each MLP.

In this case, optimal MLPs on the convex hull can be found as follows.
```shell
> pypolymlp-utils --find_optimal Ag-Au/* --key test-disp1
```
