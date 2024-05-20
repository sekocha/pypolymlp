# A generator of polynomial machine learning potentials

## Polynomial machine learning potentials

- [Polynomial machine learning potential repository](http://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/index.html)
- [A. Seko, J. Appl. Phys. 133, 011101 (2023)](https://doi.org/10.1063/5.0129045)

## Required libraries and python modules

- numpy
- scipy
- Eigen3 (if building pypolymlp)
- pybind11 (if building pypolymlp)
- phonopy (if using phonon datasets and/or computing force constants)
- phono3py (if using phonon datasets and/or computing force constants)
- symfc (if computing force constants)
- spglib (optional)
- joblib (optional)

## Installing pypolymlp using pip package

- Intel Linux
- Compatible with python 3.9, 3.10, and 3.11

```
> pip install $(pypolymlp)/dist/pypolymlp-0.1-py3-none-any.whl
```

## Manually installing pypolymlp

1. Conda package management system

```
> conda create -n pypolymlp 
> conda activate pypolymlp

> conda install numpy scipy pybind11 eigen cmake
(optional)
> conda install spglib
> conda install phono3py
> conda install joblib
```

2. Building a shared library (libmlpcpp)

The process of building libmlpcpp may take approximately five minutes to one hour. 
```
> cd $(pypolymlp)/src/pypolymlp/cxx
> cmake -S . -B build
> cmake --build build
> cmake --install build
```
or
```
> cd $(pypolymlp)/src/pypolymlp/cxx
> make
```

If necessary, the stack size may need to be set to unlimited.
```
ulimit -s unlimited
```

3. Install pypolymlp using pip

```
> cd $(pypolymlp)
> pip install .
```

## MLP development 

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

### MLP development using a memory-efficient sequential implementation

```
> pypolymlp -i polymlp.in --sequential
```

### MLP development using hybrid models

```
> pypolymlp -i polymlp1.in polymlp2.in (--sequential)
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

### Properties (energies, forces, and stress tensors)

```
> pypolymlp --properties --pot polymlp.lammps --poscars */POSCAR
> pypolymlp --properties --pot polymlp.lammps --vaspruns vaspruns/vasprun.xml.polymlp.*
> pypolymlp --properties --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AgI.yaml.xz
```

### Polynomial structural features

```
> pypolymlp --features --pot polymlp.lammps --poscars */POSCAR
> pypolymlp --features -i polymlp.in --poscars */POSCAR
```

### Force constants 

(phonopy, phono3py, and symfc are required.)
```
> pypolymlp --force_constants --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AlN.yaml.xz
> pypolymlp --force_constants --pot polymlp.lammps --str_yaml polymlp_str.yaml --fc_n_samples 1000
> pypolymlp --force_constants --pot polymlp.lammps --poscar POSCAR-unitcell --supercell 3 3 2 --fc_n_samples 1000
```

### Phonon calculations

(phonopy is required.)
```
> pypolymlp --phonon --pot polymlp.lammps --poscar POSCAR-unitcell --supercell 3 3 2
```

- [Utilities](docs/utilities.md)
- [Python API (MLP development)](docs/api_mlpdev.md)
- [Python API (Property calculations)](docs/api_calc.md)

