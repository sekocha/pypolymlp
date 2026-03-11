# Interface for OpenMX Datasets

`pypolymlp` can be used for OpenMX datasets in a manner similar to VASP datasets. See [VASP (Command line interface)](mlpdev_command.md) and [VASP (Python API)](mlpdev_api.md).
The only difference is that `dataset_type` must be set to `openmx`.

Version 0.18.10 or later.

## MLP development using a single model from command line interface
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

## MLP development using hybrid models

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
