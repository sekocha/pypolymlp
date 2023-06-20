# A generator of polynomial machine learning potentials

## MLP development using a single training dataset and a single test dataset

```
> $(pypolymlp)/mlpgen/generator.py --in polymlp.in
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

    include_force True
    include_stress True

```

## MLP development using multiple datasets

```
> $(pypolymlp)/mlpgen/multi_datasets/generator.py --in polymlp.in
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
    
    train_data vaspruns/train1/vasprun-*.xml.polymlp
    train_data vaspruns/train2/vasprun-*.xml.polymlp
    test_data vaspruns/test1/vasprun-*.xml.polymlp
    test_data vaspruns/test2/vasprun-*.xml.polymlp

    include_force True
    include_stress True

```

## MLP development using a memory-efficient sequential implementation

```
> $(pypolymlp)/mlpgen/multi_datasets/generator_sequential.py --in polymlp.in

```

## MLP development using additive models

```
> $(pypolymlp)/mlpgen/multi_datasets/additive/generator.py --in polymlp1.in polymlp2.in
> $(pypolymlp)/mlpgen/multi_datasets/additive/generator_sequential.py --in polymlp1.in polymlp2.in

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

## Computation of structural features

```
> $(pypolymlp)/tools/compute_features.py
    --infile polymlp.in --poscars poscars/poscar-000*

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

```
