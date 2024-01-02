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

### Conda package management system

```
> conda create -n pypolymlp 
> conda activate pypolymlp

> conda install numpy scipy pybind11 eigen cmake
(optional)
> conda install spglib
> conda install phono3py
> conda install joblib
```

### Building a shared library (libmlpcpp)

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

### Install pypolymlp using pip

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

### MLP development using additive models

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

## Utilities

### Structure generation for DFT calculations

1. Prototype structure selection
```
> pypolymlp-structure --prototype --n_types 1
> pypolymlp-structure --prototype --n_types 2 --comp 0.25 0.75
> pypolymlp-structure --prototype --n_types 2 --comp 1 3
> pypolymlp-structure --prototype --n_types 3 
```
Only alloy structure types are available. 
Selected prototypes are listed in polymlp_prototypes.yaml.

2. Random structure generation
- Generation from prototype structures
```
> pypolymlp-structure --random --poscars prototypes/* --n_str 10 --low_density 2 --high_density 2
```
Structures for are listed in polymlp_str_samples.yaml. 
Structures are generated in "poscar" directory.

- Generation from a given structure
```
> pypolymlp-structure --random --poscars POSCAR --n_str 10 --low_density 2 --high_density 2
```

- Random displacements for phonon calculations
(phonopy is required.)
```
> pypolymlp-structure --random_phonon --supercell 3 3 2 --n_str 20 --disp 0.03 -p POSCAR
```
Structures are generated in "poscar_phonon" directory.

3. DFT calculations for structures 

### Compression of vasprun.xml files

```
> pypolymlp-utils --vasprun_compress vaspruns/vasprun-*.xml
```
Compressed vasprun.xml is generated as vasprun.xml.polymlp.

### Automatic division of DFT dataset

```
> pypolymlp-utils --auto_dataset dataset1/*/vasprun.xml dataset2/*/vasprun.xml
> cat polymlp.in.append >> polymlp.in
```
A given DFT dataset is automatically divided into some sets, depending on the values of the energy, the forces acting on atoms, and the volume.
A generated file "polymlp.in.append" can be appended in your polymlp.in, which will be used for developing MLPs. 
Datasets identified with "train1" and "test1" are composed of structures with low energy and small force values.
The predictive power for them is more important than the other structures for the successive calculations using polynomial MLPs, so the prediction errors for "train1" and "test1" datasets should be accuracy measures for polynomial MLPs.

### Atomic energies
(Experimental: Only for VASP calculations using PBE and PBEsol functionals)

```
> pypolymlp-utils --atomic_energy_elements Mg O --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula MgO --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula Al2O3 --atomic_energy_functional PBEsol
```

A standard output can be appended in your polymlp.in, which will be used for developing MLPs. 
The polynomial MLP has no constant term, which means that the energy for isolated atoms is set to zero.
The energy values for the isolated atoms must be subtracted from the energy values for structures.

### Enumeration of optimal MLPs on convex hull

```
> pypolymlp-utils --find_optimal Ti-Pb/* --key test-disp1
```
