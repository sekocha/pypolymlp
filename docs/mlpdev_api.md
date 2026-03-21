# Polynomial MLP development using Python API (VASP datasets)

## MLP development using a single model specified by a parameter file

In the following example, a polynomial MLP is developed from a single parameter file using the Python API of `pypolymlp`.
Datasets specified in `polymlp.in` are loaded to train the polynomial MLP.
The format of the input file is the same as that used for the command-line interface.
See [Command-line interface](mlpdev_command.md).

```python
import numpy as np
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.load_parameter_file("polymlp.in")
polymlp.load_datasets()
polymlp.run(verbose=True)

polymlp.save_mlp(filename="polymlp.yaml")

params = polymlp.parameters
mlp_info = polymlp.summary
```
```shell
> cat polymlp.in

    # Parameters
    # ----------
    # elements: Element species, (e.g., ['Mg','O'])
    # include_force: Include force entries
    # include_stress: Include virial stress entries
    # cutoff: Cutoff radius (Angstrom)
    # model_type: Polynomial function type
    #     model_type = 1: Linear polynomial of polynomial invariants
    #     model_type = 2: Polynomial of polynomial invariants
    #     model_type = 3: Polynomial of pair invariants
    #                     + linear polynomial of polynomial invariants
    #     model_type = 4: Polynomial of pair and second-order invariants
    #                     + linear polynomial of polynomial invariants
    # max_p: Order of polynomial function
    # feature_type: Structural feature type. 'gtinv' or 'pair'.
    # n_gaussians: Number of Gaussians.
    # reg_alpha_params: Parameters for penalty term in
    #                   linear ridge regression. Parameters are given as
    #                   np.linspace(p[0], p[1], p[2]).
    # gtinv_order: Maximum order of polynomial invariants.
    # gtinv_maxl: Maximum angular numbers of polynomial invariants.
    #             [maxl for order=2, maxl for order=3, ...]
    # atomic_energy: Atomic energies (in eV).

    n_type 2
    elements Mg O

    feature_type gtinv
    cutoff 8.0
    model_type 3
    max_p 2

    gtinv_order 3
    gtinv_maxl 4 4

    # Equivalent to
    # gaussian_params1 1.0 1.0 1
    # gaussian_params2 0.0 7.0 8
    n_gaussians 9

    reg_alpha_params -3 1 5

    atomic_energy -0.00040000 -1.85321219

    train_data vaspruns/train/vasprun-*.xml.polymlp
    test_data vaspruns/test/vasprun-*.xml.polymlp

    include_force True
```
In this example, the polynomial MLP file `polymlp.yaml` is generated.
When the coefficients of the polynomial MLP are needed, they can be obtained from the `summary` attribute.

## MLP development for a single model without using a parameter file

In the following example, a polynomial MLP is developed from datasets of `vasprun.xml` files located in `vaspruns/train` and `vaspruns/test`.
The input parameters of the polynomial MLP are specified using the `set_params` function.

See [Notes on Polymlp Parameters](mlpdev_params.md) for details on the input parameters.

```python
import numpy as np
import glob
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    elements=['Mg','O'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    n_gaussians=9,
    atomic_energy=[-0.00040000,-1.85321219],
)
train_vaspruns = glob.glob('vaspruns/train/vasprun-*.xml.polymlp')
test_vaspruns = glob.glob('vaspruns/test/vasprun-*.xml.polymlp')
polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```

## How to Use Multiple Datasets of `vasprun.xml` Files
If multiple datasets are used for training and testing, use the `set_multiple_datasets_vasp` function instead of `set_datasets_vasp`.

```python
train_vaspruns1 = glob.glob('vaspruns/train1/vasprun-*.xml.polymlp')
train_vaspruns2 = glob.glob('vaspruns/train2/vasprun-*.xml.polymlp')
test_vaspruns1 = glob.glob('vaspruns/test1/vasprun-*.xml.polymlp')
test_vaspruns2 = glob.glob('vaspruns/test2/vasprun-*.xml.polymlp')
polymlp.set_multiple_datasets_vasp(
    [train_vaspruns1, train_vaspruns2],
    [test_vaspruns1, test_vaspruns2]
)
polymlp.run(verbose=True)
```

## MLP development for a hybrid model

A hybrid polynomial MLP can be developed using the Python API.
Additional parameter sets can be specified using the `append_hybrid_params` function.

Version 0.19.0 or later.

```python
import numpy as np
import glob
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    elements=['Mg','O'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    n_gaussians=9,
    atomic_energy=[-0.00040000,-1.85321219],
)
polymlp.append_hybrid_params(
    elements=['Mg','O'],
    cutoff=4.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[6,6],
    n_gaussians=5,
)
polymlp.append_hybrid_params(
    elements=['Mg','O'],
    cutoff=3.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[8,8],
    n_gaussians=4,
)
train_vaspruns = glob.glob('vaspruns/train/vasprun-*.xml.polymlp')
test_vaspruns = glob.glob('vaspruns/test/vasprun-*.xml.polymlp')
polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```


## File Input and Output
- **Save the polynomial MLP to a file**
```python
polymlp.save_mlp(filename="polymlp.yaml")
```

- **Generate and save energy predictions**
By default, `polymlp.run` does not generate energy prediction files. To generate energy predictions for the training and test datasets, use the `fit` and `estimate_error` methods as shown below:
```python
polymlp.fit(verbose=True)
polymlp.estimate_error(log_energy=True, verbose=True)
```

- **Save RMS errors**
Save the Root Mean Square (RMS) errors and mean absolute errors (MAE) for both the training and test datasets:
```python
polymlp.save_errors(filename="polymlp_error.yaml")
```

- **Save model parameters***
Save the parameters used for developing the PolyMLP model:
```python
polymlp.save_params(filename="polymlp_params.yaml")
```
