# A generator of polynomial machine learning potentials

`pypolymlp` is a Python code designed for the development of polynomial machine learning potentials (MLPs) based on datasets generated from density functional theory (DFT) calculations. The code provides functionalities for fitting polynomial models to energy, force, and stress data, enabling the construction of accurate and computationally efficient interatomic potentials.
In addition to potential development, `pypolymlp` allows users to compute various physical properties and perform atomistic simulations using the trained MLPs.

## Polynomial machine learning potentials
A polynomial MLP represents the potential energy as a polynomial function of linearly independent polynomial invariants of the O(3) group. Developed polynomial MLPs are available in [Polynomial Machine Learning Potential Repository](http://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/index.html).

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
- phonopy
- phono3py
- symfc
- sparse_dot_mkl
- spglib
- pymatgen
- ase
- joblib
- matplotlib, seaborn

## How to install pypolymlp

- Install from conda-forge (Recommended)

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

### Polynomial MLP development
To develop polynomial MLPs from datasets obtained from DFT calculations, both the command-line interface and the Python API are available.
Several procedures for generating structures used in DFT calculations are also supported.

- Tutorials
  1. [Development of a single on-the-fly MLP](docs/tutorial_onthefly.md)
  2. Development of a single general-purpose MLP
  3. [Development of convex hull (Pareto-optimal) MLPs](docs/utils_grid.md)
- MLP Developments
  - [VASP (Command line interface)](docs/mlpdev_command.md)
  - [VASP (Python API)](docs/mlpdev_api.md)
  - [Structure-Properties general datasets (Python API)](docs/mlpdev_dataset_api.md)
  - [Phono3py (Python API)](docs/mlpdev_phono3py.md)
  - [OpenMX](docs/mlpdev_openmx.md)
  - [Notes on parameter and dataset settings](docs/mlpdev_params.md)

- Utilities for MLP development
  - Dataset Generation
    - [Generator of DFT random structures](docs/utils_strgen.md)
    - [Compression of vasprun.xml files](docs/utils_compress.md)
    - [Automatic division of DFT dataset](docs/utils_dataset_div.md)
  - [Convex hull (Pareto-optimal) MLP search](docs/utils_grid.md)
  - [Atomic energies](docs/utils_atomic_energies.md)
  - [Generator of portable model for OpenKIM](docs/utils_kim.md)
  - [Other utilities](docs/utils.md)

- Experimental Features
  - [SSCHA free energy model](docs/experimental/mlpdev_sscha.md)
  - [Electronic free energy model](docs/experimental/mlpdev_electron.md)
  - [Substitutional disordered model](docs/experimental/mlpdev_disorder.md)


### Calculations using polynomial MLP
In version 0.8.0 or earlier, polymlp files are generated in a plain text format as `polymlp.lammps`.
Starting from version 0.9.0, the files are generated in YAML format as `polymlp.yaml`.
Both formats are supported by the command-line interface and the Python API.
The following calculations can be performed using **pypolymlp** with the polynomial MLP files `polymlp.yaml` or `polymlp.lammps`.

- [Notes on hybrid polynomial MLPs](docs/calc_hybrid.md)
- Property calculations and simulation
  - [Energy, forces, stress tensor](docs/calc_property.md)
  - [Formation energy](docs/calc_formation.md)
  - [Equation of states](docs/calc_eos.md)
  - [Local geometry optimization](docs/calc_geometry.md)
  - [Elastic constants](docs/calc_elastic.md)
  - [Phonon properties, Quasi-harmonic approximation](docs/calc_phonon.md)
  - [Force constants](docs/calc_fc.md)
  - [Lattice thermal conductivity](docs/calc_ltc.md)
  - [Polynomial invariants](docs/calc_features.md)
  - [Systematic property calculations](docs/calc_auto.md)
- Experimental features
  - [Self-consistent phonon calculations](docs/experimental/calc_sscha.md)
  - [Molecular dynamics](docs/experimental/calc_md.md)
  - [Thermodynamic integration using molecular dynamics](docs/experimental/calc_ti.md)
  - [Thermodynamic property calculation](docs/experimental/calc_thermodynamics.md)
  - Evaluation of atomic-configuration-dependent electronic free energy
  - Global structure optimization
  - Structure optimization at finite temperatures

- [How to use polymlp in other calculator tools](docs/api_other_calc.md)
  - LAMMPS
  - ASE
  - OpenKIM
  - phonopy and phonon3py
