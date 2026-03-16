# Interface to OpenMX Datasets

`pypolymlp` can be used for OpenMX datasets in a manner similar to VASP datasets. See [VASP (Command line interface)](mlpdev_command.md) and [VASP (Python API)](mlpdev_api.md).
When using command-line interface, the only difference is that `dataset_type` must be set to `openmx` in the input file.

Version 0.19.0 or later.

## MLP development using a single model via command-line interface

If the file is named `polymlp.in`, simply run `pypolymlp -i polymlp.in` and the potential file will be generated.
Details on input parameter and dataset settings can be found in [Notes on parameter and dataset settings](mlpdev_params.md).

```shell
> pypolymlp -i polymlp.in
> cat polymlp.in

    n_type 2
    elements Ag C

    feature_type gtinv
    cutoff 8.0
    model_type 3
    max_p 2

    gtinv_order 3
    gtinv_maxl 4 4

    n_gaussians 10

    reg_alpha_params -5 -1 5

    atomic_energy_unit Hartree
    atomic_energy -114.017822031176 -5.499002028934

    dataset_type openmx
    data_md simulation1.md
    data_md simulation2.md
    data_md simulation3.md

    include_force True
    include_stress False
```
In this example, the datasets will be automatically divided into training and test sets.


## MLP development using a hybrid model via command-line interface

A hybrid polynomial MLP represents the potential energy as the sum of multiple MLPs.
All coefficients of the hybrid model are simultaneously estimated using regression and the given datasets.

When a hybrid model with two polymlps is considered, the corresponding two input files must be specified as follows.


```shell
> pypolymlp -i polymlp1.in polymlp2.in
```
```shell
> cat polymlp1.in

    n_type 2
    elements Ag C

    feature_type gtinv
    cutoff 8.0
    model_type 3
    max_p 2

    gtinv_order 3
    gtinv_maxl 4 4

    n_gaussians 10

    reg_alpha_params -5 -1 5

    atomic_energy_unit Hartree
    atomic_energy -114.017822031176 -5.499002028934

    dataset_type openmx
    data_md simulation1.md
    data_md simulation2.md
    data_md simulation3.md

    include_force True
    include_stress False
```

```shell
> cat polymlp2.in
    n_type 2
    elements Ag C

    feature_type gtinv
    cutoff 4.0
    model_type 3
    max_p 2

    gtinv_order 3
    gtinv_maxl 8 8

    n_gaussians 5
```
Two polymlp files `polymlp.yaml.1` and `polymlp.yaml.2` will be generated.


## MLP development using a single model via Python API

In the following example, a polynomial MLP is developed using the `pypolymlp` Python API.
The model is constructed using datasets from `OpenMX` files.

The input parameters for the polynomial MLP are defined via the `set_params` function. For a detailed explanation of these parameters, please refer to [Notes on Polymlp Parameters](mlpdev_params.md).


```python
from pypolymlp.core.interface_openmx import parse_openmx
from pypolymlp.core.utils import split_ids_train_test
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

# Parse openmx MD data files.
datafiles = ["sample.md"]
structures, energies, forces = parse_openmx(datafiles)

# Split dataset into training and test datasets automatically.
n_data = len(energies)
train_ids, test_ids = split_ids_train_test(n_data, train_ratio=0.9)
train_structures = [structures[i] for i in train_ids]
test_structures = [structures[i] for i in test_ids]
train_energies = energies[train_ids]
test_energies = energies[test_ids]
train_forces = [forces[i] for i in train_ids]
test_forces = [forces[i] for i in test_ids]

polymlp = Pypolymlp()
polymlp.set_params(
    elements=("Ag", "C"),
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=(4, 4),
    reg_alpha_params=(-5, -1, 5),
    n_gaussians=10,
    atomic_energy_unit="Hartree",
    atomic_energy=(-114.017822031176,-5.499002028934),
)

polymlp.set_datasets_structures(
    train_structures=train_structures,
    test_structures=test_structures,
    train_energies=train_energies,
    test_energies=test_energies,
    train_forces=train_forces,
    test_forces=test_forces,
)

polymlp.print_params()
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```

## MLP development using a hybrid model via Python API

A hybrid model can also be developed using the Python API.
To specify additional sets of parameters, use the `append_hybrid_params` function.

```python
polymlp = Pypolymlp()
polymlp.set_params(
    elements=("Ag", "C"),
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=(4, 4),
    reg_alpha_params=(-5, -1, 5),
    n_gaussians=10,
    atomic_energy_unit="Hartree",
    atomic_energy=(-114.017822031176,-5.499002028934),
)
polymlp.append_hybrid_params(
    elements=("Ag", "C"),
    cutoff=4.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=(8, 8),
    n_gaussians=5,
)
polymlp.set_datasets_structures(
    train_structures=train_structures,
    test_structures=test_structures,
    train_energies=train_energies,
    test_energies=test_energies,
    train_forces=train_forces,
    test_forces=test_forces,
)

polymlp.print_params()
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```
Two polymlp files `polymlp.yaml.1` and `polymlp.yaml.2` will be generated.
