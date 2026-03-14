# Polynomial MLP development using command-line interface (VASP datasets)

## MLP development using a single polymlp model
Once datasets are prepared, a polynomial MLP can be developed using the command-line interface.
In this procedure, an input parameter file must be prepared.
If the file is named `polymlp.in`, simply run `pypolymlp -i polymlp.in` and and the potential file will be generated.
Details on input parameter and dataset settings can be found in [Notes on parameter and dataset settings](mlpdev_params.md).

```shell
> pypolymlp -i polymlp.in
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
    include_stress True
```
If version <= 0.8.0 is used, polymlp files are generated in a text format as `polymlp.lammps`. If a newer version (>= 0.9.0) is used, polymlp files are generated in a yaml format as `polymlp.yaml`.

When multiple datasets are specified and prediction errors are evaluated separately for each dataset, they can be listed on multiple lines in the input file as follows.

```shell
train_data vaspruns/train1/vasprun-*.xml.polymlp
train_data vaspruns/train2/vasprun-*.xml.polymlp
test_data vaspruns/test1/vasprun-*.xml.polymlp
test_data vaspruns/test2/vasprun-*.xml.polymlp
```

When datasets are automatically divided into training and test datasets, the `data` tag can be used.

```shell
data vaspruns/*1/vasprun-*.xml.polymlp
data vaspruns/*2/vasprun-*.xml.polymlp
```

## MLP development using a hybrid polymlp model

A hybrid polynomial MLP represents the potential energy as the sum of multiple MLPs.
All coefficients of the hybrid model are simultaneously estimated using regression and the given datasets.

When a hybrid model with two polymlps is considered, the corresponding two input files must be specified as follows.

```shell
> pypolymlp -i polymlp1.in polymlp2.in
> cat polymlp1.in

    n_type 2 elements Mg O
    feature_type gtinv
    cutoff 8.0
    model_type 3
    max_p 2

    gtinv_order 3
    gtinv_maxl 4 4

    gaussian_params1 1.0 1.0 1
    gaussian_params2 0.0 7.0 8

    reg_alpha_params -3 1 5

    atomic_energy -0.00040000 -1.85321219

    train_data vaspruns/train1/vasprun-*.xml.polymlp
    train_data vaspruns/train2/vasprun-*.xml.polymlp
    test_data vaspruns/test1/vasprun-*.xml.polymlp
    test_data vaspruns/test2/vasprun-*.xml.polymlp

    include_force True
    include_stress True

> cat polymlp2.in

    n_type 2
    elements Mg O

    feature_type gtinv
    cutoff 4.0
    model_type 3
    max_p 2

    gtinv_order 3
    gtinv_maxl 4 4

    gaussian_params1 1.0 1.0 1
    gaussian_params2 0.0 3.0 4
```
Two polymlp files, `polymlp.yaml.1` and `polymlp.yaml.2`, will be generated.
They must be used together as a set of polymlps to calculate properties.
