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

## How to use pypolymlp

- [Polynomial MLP development](docs/mlpdev.md)
- [Property calculators](docs/calc.md)
  - Energy, forces on atoms, and stress tensor
  - Force constants
  - Elastic constants
  - Equation of states
  - Structural features (Polynomial invariants)
  - Local geometry optimization
  - Phonon properties, Quasi-harmonic approximation
  - Self-consistent phonon calculations
- [Utilities](docs/utilities.md)
  - Random structure generation
  - Estimation of computational costs
  - Enumeration of optimal MLPs
  - Compression of vasprun.xml files
  - Automatic division of DFT dataset
  - Atomic energies
- [Python API (MLP development)](docs/api_mlpdev.md)
- [Python API (Property calculations)](docs/api_calc.md)

