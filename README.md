# A generator of polynomial machine learning potentials

## Polynomial machine learning potentials

- [Polynomial machine learning potential repository](http://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/index.html)
- [A. Seko, J. Appl. Phys. 133, 011101 (2023)](https://doi.org/10.1063/5.0129045)

## Required libraries and python modules

- numpy
- scipy
- pyyaml
- eigen3
- pybind11
- openmp (recommended)
- phonopy (if using phonon datasets and/or computing force constants)
- phono3py (if using phonon datasets and/or computing force constants)
- symfc (if computing force constants)
- spglib (optional)
- joblib (optional)

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
