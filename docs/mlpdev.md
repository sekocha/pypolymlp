# MLP development

If version <= 0.8.0 is used, polymlp files are generated in a text format as `polymlp.lammps`.
If a newer version (>= 0.9.0) is used, polymlp files are generated in a yaml format as `polymlp.yaml`.

```
> pypolymlp -i polymlp.in
> cat polymlp.in

    n_type 2
    elements Mg O

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

    train_data vaspruns/train/vasprun-*.xml.polymlp
    test_data vaspruns/test/vasprun-*.xml.polymlp

    (if using multiple datasets)
    train_data vaspruns/train1/vasprun-*.xml.polymlp
    train_data vaspruns/train2/vasprun-*.xml.polymlp
    test_data vaspruns/test1/vasprun-*.xml.polymlp
    test_data vaspruns/test2/vasprun-*.xml.polymlp

    include_force True
    include_stress True
```

## MLP development using hybrid models

```
> pypolymlp -i polymlp1.in polymlp2.in
> cat polymlp1.in

    n_type 2
    elements Mg O

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

## Dataset settings

When both the training and test datasets are explicitly provided, they can be included in the input file as follows:

```
train_data vaspruns/train1/vasprun-*.xml.polymlp
train_data vaspruns/train2/vasprun-*.xml.polymlp
test_data vaspruns/test1/vasprun-*.xml.polymlp
test_data vaspruns/test2/vasprun-*.xml.polymlp
```

In cases where the datasets are automatically divided into training and test sets, they can be included in the input file as follows:

```
data vaspruns1/vasprun-*.xml.polymlp
data vaspruns2/vasprun-*.xml.polymlp
```

When the datasets contain property entries for multiple structures, such as those derived from molecular dynamics (MD) simulations, they can be specified in the input file as follows:

```
data_md vasprun-md1.xml
data_md vasprun-md2.xml
data_md vasprun-md-*.xml
```

These datasets will be automatically divided into training and test sets.
