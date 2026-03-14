# Interface to OpenMX Datasets

`pypolymlp` can be used for OpenMX datasets in a manner similar to VASP datasets. See [VASP (Command line interface)](mlpdev_command.md) and [VASP (Python API)](mlpdev_api.md).
When using command-line interface, the only difference is that `dataset_type` must be set to `openmx` in the input file.

Version 0.19.0 or later.

## MLP development using a single model via command-line interface

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

    gaussian_params1 1.0 1.0 1
    gaussian_params2 0.0 7.0 10

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
More details on input parameter and dataset settings can be found in [Notes on parameter and dataset settings](mlpdev_params.md).

## MLP development using a hybrid model via command-line interface

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

    gaussian_params1 1.0 1.0 1
    gaussian_params2 0.0 7.0 10

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

    gaussian_params1 1.0 1.0 1
    gaussian_params2 0.0 3.0 5
```
Two polymlp files `polymlp.yaml.1` and `polymlp.yaml.2` will be generated.


## MLP development using a single model via Python API

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
    elements=["Ag", "C"],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=(4, 4),
    reg_alpha_params=(-5, -1, 5),
    gaussian_params2=(0.0, 7.0, 10),
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

```python
polymlp = Pypolymlp()
polymlp.set_params(
    elements=["Ag", "C"],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=(4, 4),
    reg_alpha_params=(-5, -1, 5),
    gaussian_params2=(0.0, 7.0, 10),
    atomic_energy_unit="Hartree",
    atomic_energy=(-114.017822031176,-5.499002028934),
)
polymlp.append_hybrid_params(
    elements=["Ag", "C"],
    cutoff=4.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=(8, 8),
    gaussian_params2=(0.0, 3.0, 5),
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
