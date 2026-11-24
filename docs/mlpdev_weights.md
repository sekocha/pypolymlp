# Notes on Weights Assigned to Data Entries

## Automatically Assigned Weights

`pypolymlp` uses weighted linear ridge regression, in which the weighted sum of squared errors, rather than the standard (unweighted) sum of squared errors, is minimized during regression.

By default, weights are automatically assigned to individual data entries based on their energy values and the absolute values of their forces and virial stresses.
Specifically, smaller weights are assigned to data entries with larger absolute values of these quantities.

The rationale behind this automatic weighting scheme is twofold:
- To accurately reproduce the very small forces and virial stresses of structures near equilibrium (e.g., forces below 0.01 eV/Å).
- To reduce the influence of structures with extremely high energies and large repulsive forces. Including such structures in the training dataset is recommended to ensure robust and reliable geometry optimizations, molecular dynamics simulations, and global structure searches without failures.


## Definition of Weights

In `pypolymlp`, the weights assigned to energy, force, and stress entries are defined as follows:

- **Energy:**
  $w = w_{\mathrm{auto}} \times w_{\mathrm{dataset}}$
- **Force:**
  $
  w = w_{\mathrm{auto}} \times w_{\mathrm{dataset}} \times w_{\mathrm{force}}
  $
- **Stress:**
  $
  w = w_{\mathrm{auto}} \times w_{\mathrm{dataset}} \times w_{\mathrm{stress}}
  $

Here:

- $w_{\mathrm{auto}}$ is the automatically determined weight based on the value of each data entry.
The value of $w_{\mathrm{auto}}$ ranges from 0.0 to 1.0.
For details, see [Phys. Rev. B 110, 224102 (2024)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.224102) and [J. Chem. Phys. 165, 144104 (2026)](https://pubs.aip.org/aip/jcp/article-abstract/165/14/144104/3406254/Pressure-temperature-phase-diagram-from-global?redirectedFrom=fulltext).
- $w_{\mathrm{dataset}}$ is a weighting factor specified for each dataset (Default: 1.0).
- $w_{\mathrm{force}}$ and $w_{\mathrm{stress}}$ are additional scaling factors applied to force and stress entries, respectively.
By default, $w_{\mathrm{force}}$ and $w_{\mathrm{stress}}$ are set to 1.0 and 0.1, respectively.

## Disabling Automatic Weight Assignment

Available in version 0.23.0 and later.

`pypolymlp` allows automatically assigned weights to be disabled during regression.
When automatic weighting is disabled and all weights are set to 1.0, model estimation is regarded as standard (unweighted) linear ridge regression.

In the command-line interface, automatic weighting can be disabled by setting the `use_auto_weight` option to `False` in the input parameter file:

```text
use_auto_weight False
```
In the Python API, the same option can be specified in the `set_params` method as follows:
```python
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    ...,
    use_auto_weight=False,
    ...,
)
```

## How to Specify Custom Weights

The force and stress scaling factors can be specified in the input file as follows:
```text
weight_force 1.0
weight_stress 0.1
```
Alternatively, these factors can be specified using the `set_params` method in the Python API:
```python
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    ...,
    weight_force=1.0
    weight_stress=1.0
    ...,
)
```

The weighting factor specified for each dataset can be specified in the input file as follows:
```text
# tag, file location, include_force for dataset, weighting factor for dataset
train_data vasprun-train-*.xml True 0.8
data vasprun-data-*.xml True 0.8
```
The last column specifies the weighting factor assigned to each dataset.


## Notes on Prediction Error Evaluation for Weighted Regression

Prediction errors evaluated using conventional averages such as RMSE and MAE can be considerably larger than those obtained with standard (unweighted) regression, depending on the distribution of structures in the dataset.
In particular, when RMSE is used as an evaluation metric instead of MAE, the resulting scores can be strongly influenced by structures with large absolute property values and large prediction errors.
These structures are intentionally assigned significantly lower weights during regression.

Therefore, when using weighted ridge regression with automatically assigned weights, alternative evaluation metrics that account for the weighting scheme may be more appropriate than conventional averaged error metrics such as RMSE and MAE.
For example, percentage errors can provide a complementary measure of prediction accuracy.
Another option is to divide the dataset into multiple subsets according to property values and evaluate prediction errors separately for each subset.
This approach allows the prediction accuracy to be assessed specifically for structures that are particularly important in practical simulations.
