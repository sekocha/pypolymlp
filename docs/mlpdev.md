# MLP development

```
> pypolymlp -i polymlp.in --no_sequential
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

## MLP development using a memory-efficient sequential implementation

```
> pypolymlp -i polymlp.in
```

## MLP development using hybrid models

```
> pypolymlp -i polymlp1.in polymlp2.in (--no_sequential)
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
