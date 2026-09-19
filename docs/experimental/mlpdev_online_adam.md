# Fine-Tuning an MLP Using Online Learning with the Adam Optimizer

> **Note:** This feature requires version 0.21.5 or later.

Starting from a polynomial MLP that has already been developed using a dataset, or from one distributed in the polynomial MLP repository, the MLP can be fine-tuned using an additional dataset through online learning with the Adam optimizer.

During fine-tuning, the MLP is updated to reduce the prediction errors for the additional dataset.

When fine-tuning a single MLP using online learning, a `polymlp.yaml` file must be specified with the `--pot` option, as follows:

```shell
> pypolymlp-online --pot polymlp.yaml --vaspruns vasprun-add-*.xml
```
Currently, `pypolymlp` supports `vasprun.xml` files as input for the additional dataset.

The parameters used by the Adam optimizer can be specified using the following options.
If `alpha > 0`, a regularization term, `alpha * ||(w - w0) / w0||^2`, is added to the objective function.

```python
"""
Parameters
----------
max_learning_rate: Maximum learning rate used as the initial learning rate. (Default: 1e-3)
alpha: Magnitude of the regularization term. (Default: 1.0)
beta: Parameter used to control the gradient update. (Default: 0.99)
batch_size: Minibatch size. If None, the minibatch size is automatically determined. (Default: None)
gtol: Gradient tolerance for convergence. (Default: 1e-5)
n_epochs: Number of epochs. (Default: 1000)
"""
```
```shell
> pypolymlp-online --pot polymlp.yaml --vaspruns vasprun-add-*.xml --alpha 0.1 --beta 0.99 --n_epochs 3000 --max_learning_rate 0.001 --gtol 1e-4
```

When fine-tuning a hybrid MLP using online learning, a set of `polymlp.yaml` files must be specified with the --pot option, as follows:
```shell
> pypolymlp-online --pot polymlp.yaml.1 polymlp.yaml.2 --vaspruns vasprun-add-*.xml
```
