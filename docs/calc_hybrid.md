# Notes on Hybrid Polynomial MLPs

In hybrid polynomial MLPs, the potential energy is represented as a sum of multiple polynomial MLPs, where the regression coefficients of all models are determined simultaneously.
Using a hybrid framework often improves both the accuracy and computational efficiency of polynomial MLPs.

When calculating properties or performing simulations with hybrid polynomial MLPs in `pypolymlp`, specify multiple MLP files using the `--pot` option as follows:
```shell
--pot polymlp.yaml*
# or
--pot polymlp.yaml.1 polymlp.yaml.2 polymlp.yaml.3
```
via the command-line interface.

When using hybrid MLPs through the Python API, multiple MLP files can be specified as shown below:
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlps = ['polymlp.yaml.1', 'polymlp.yaml.2']
polymlp_calc = PypolymlpCalc(pot=polymlps)
```
