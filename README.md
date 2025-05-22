# A generator of polynomial machine learning potentials

## Polynomial machine learning potentials

- [Polynomial machine learning potential repository](http://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/index.html)

## Citation of pypolymlp

“Tutorial: Systematic development of polynomial machine learning potentials for elemental and alloy systems”, [A. Seko, J. Appl. Phys. 133, 011101 (2023)](https://doi.org/10.1063/5.0129045)

```
@article{pypolymlp,
    author = {Seko, Atsuto},
    title = "{"Tutorial: Systematic development of polynomial machine learning potentials for elemental and alloy systems"}",
    journal = {J. Appl. Phys.},
    volume = {133},
    number = {1},
    pages = {011101},
    year = {2023},
    month = {01},
}
```

## Required libraries and python modules

- python >= 3.9
- numpy != 2.0.*
- scipy
- pyyaml
- setuptools
- eigen3
- pybind11
- openmp (recommended)

[Optional]
- phonopy (if using phonon datasets and/or computing force constants)
- phono3py (if using phonon datasets and/or computing force constants)
- symfc (if computing force constants)
- sparse_dot_mkl (if computing force constants)
- spglib
- pymatgen
- ase

## How to install pypolymlp

- Install from conda-forge

| Version | Last Update | Downloads | Platform | License |
| ---- | ---- | ---- | ---- | ---- |
| ![badge](https://anaconda.org/conda-forge/pypolymlp/badges/version.svg) | ![badge](https://anaconda.org/conda-forge/pypolymlp/badges/latest_release_date.svg) | ![badge](https://anaconda.org/conda-forge/pypolymlp/badges/downloads.svg)| ![badge](https://anaconda.org/conda-forge/pypolymlp/badges/platforms.svg) | ![badge](https://anaconda.org/conda-forge/pypolymlp/badges/license.svg) |

```
conda create -n pypolymlp-env
conda activate pypolymlp-env
conda install -c conda-forge pypolymlp
```

- Install from PyPI
```
conda create -n pypolymlp-env
conda activate pypolymlp-env
conda install -c conda-forge numpy scipy pybind11 eigen cmake cxx-compiler
pip install pypolymlp
```
Building C++ codes in pypolymlp may require a significant amount of time.

- Install from GitHub
```
git clone https://github.com/sekocha/pypolymlp.git
cd pypolymlp
conda create -n pypolymlp-env
conda activate pypolymlp-env
conda install -c conda-forge numpy scipy pybind11 eigen cmake cxx-compiler
pip install . -vvv
```
Building C++ codes in pypolymlp may require a significant amount of time.

## How to use pypolymlp

- [Polynomial MLP development](docs/mlpdev.md)
- [Property calculators](docs/calc.md)
  - Energy, forces on atoms, and stress tensor
  - Force constants
  - Elastic constants
  - Equation of states
  - Structural features (Polynomial invariants)
  - Phonon properties, Quasi-harmonic approximation
  - Local geometry optimization
  - Molecular dynamics
  - Thermodynamic integration using MD
- [DFT structure generator](docs/strgen.md)
  - Random atomic displacements with constant magnitude
  - Random atomic displacements with sequential magnitudes and volume changes
  - Random atomic displacements, cell expansion, and distortion
- [Utilities](docs/utilities.md)
  - Compression of vasprun.xml files
  - Automatic division of DFT dataset
  - Atomic energies
  - Enumeration of optimal MLPs
  - Estimation of computational costs
- [Python API (MLP development)](docs/api_mlpdev.md)
- [Python API (Property calculations)](docs/api_calc.md)
  - Energy, forces on atoms, and stress tensor
  - Force constants
  - Elastic constants
  - Equation of states
  - Structural features (Polynomial invariants)
  - Phonon properties, Quasi-harmonic approximation
  - Local geometry optimization
  - Molecular dynamics
  - Thermodynamic integration using MD
  - Self-consistent phonon calculations
- [How to use polymlp in other calculator tools](docs/api_other_calc.md)
  - LAMMPS
  - Phonopy
  - ASE
