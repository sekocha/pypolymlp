# A generator of polynomial machine learning potentials

## Required libraries and python modules

- numpy
- scipy
- Eigen3
- pybind11
- phonopy (if using phonon detasets and/or computing force constants)
- phono3py (if using phonon detasets and/or computing force constants)
- symfc (if computing force constants)
- spglib (optional)
- joblib (optional)

## Conda package management system
```
> conda create -n pypolymlp 
> conda activate pypolymlp

> conda install -c conda-forge pybind11
> conda install -c omnia eigen3
(optional)
> conda install -c anaconda joblib
> conda install -c conda-forge phonopy
> conda install -c conda-forge phono3py
> conda install -c conda-forge spglib
```

## Building a shared library (mlpcpp) required for pypolymlp

```
> cd $(pypolymlp)/c++
> make
```

## MLP development 

```
> $(pypolymlp)/run_polymlp.py -i polymlp.in
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

### MLP development using a memory-efficient sequential implementation

```
> $(pypolymlp)/run_polymlp.py -i polymlp.in --sequential
```

### MLP development using additive models

```
> $(pypolymlp)/run_polymlp.py -i polymlp1.in polymlp2.in (--sequential)
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

## Calculators

### Computation of properties (energies, forces, and stress tensors)

```
> $(pypolymlp)/run_polymlp.py --properties --pot polymlp.lammps --poscars */POSCAR
> $(pypolymlp)/run_polymlp.py --properties --pot polymlp.lammps --vaspruns vaspruns/vasprun.xml.polymlp.*
> $(pypolymlp)/run_polymlp.py --properties --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AgI.yaml.xz
```

### Computation of polynomial structural features

```
> $(pypolymlp)/run_polymlp.py --features --pot polymlp.lammps --poscars */POSCAR
> $(pypolymlp)/run_polymlp.py --features -i polymlp.in --poscars */POSCAR
```

### Computation of force constants 
(phonopy, phono3py, and symfc are required.)

```
> $(pypolymlp)/run_polymlp.py --force_constants --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AgI.yaml.xz
```

## Generation of random structures 

```
> $(pypolymlp)/str_gen/structure_generator.py -p ideals/poscar-* --n_structures 50 5 --max_disp 0.5 1.5
```
