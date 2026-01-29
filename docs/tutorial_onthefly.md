# Development of an On-the-Fly Polynomial MLP

The on-the-fly MLP is developed using a dataset generated from a single target structure. The model is expected to accurately predict the properties of this target structure; however, its transferability to other structures is not anticipated.

This tutorial explains how to develop an on-the-fly MLP using the example located in the `examples/mlp_develop/vasp/MgO/single_dataset` directory.

## 1. Generate Random Structures for DFT Calculations

First, random structures are generated based on a given single target structure.
Let us now consider the target structure specified by `POSCAR` in the `examples/MgO` directory.

```shell
> cat $(pypolymlp)/examples/mlp_develop/vasp/MgO/single_dataset/POSCAR

MgO-rocksalt
   1.00000000000000
     5.6542556743096712    0.0000000000000000    0.0000000000000000
     0.0000000000000000    5.6542556743096712    0.0000000000000000
     0.0000000000000000    0.0000000000000000    5.6542556743096712
   Mg  O
     4     4
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.0000000000000000  0.5000000000000000
  0.5000000000000000  0.5000000000000000  0.0000000000000000
  0.5000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000  0.5000000000000000
  0.5000000000000000  0.5000000000000000  0.5000000000000000
```

When only atomic displacements are introduced to the target structure in order to calculate phonon-related properties under a fixed cell, random structures with displaced atoms can be generated as follows:
```shell
# Constant magnitude of atomic displacements, fixed cell
> cd $(pypolymlp)/examples/mlp_develop/vasp/MgO/single_dataset
> pypolymlp-structure -p POSCAR --displacements 100 --const_distance 0.001 --supercell 2 2 2

# Sequential magnitudes of atomic displacements, fixed cell
> cd $(pypolymlp)/examples/mlp_develop/vasp/MgO/single_dataset
> pypolymlp-structure -p POSCAR --displacements 100 --max_distance 1.5 --supercell 2 2 2
```
In this example, 100 random structures with atomic displacements are generated from the target structure `POSCAR`, each expanded into a 2x2x2 supercell.


If random structures with atomic displacements, cell expansions, and distortions are to be generated, use the `--standard` option.
```shell
> cd $(pypolymlp)/examples/mlp_develop/vasp/MgO/single_dataset
> pypolymlp-structure --poscars POSCAR --standard 100 --max_distance 1.5
```

See [DFT Structure Generator](strgen.md) for other commands to generate structures.


## 2. Perform DFT Calculations for the Random Structures

DFT calculations are performed for the random structures generated from the target structure.
In addition, the energy values are computed for the isolated atoms composing the target structures.
In this tutorial, the case where the VASP code is used is considered.

DFT calculations are performed for the random structures generated from the target structure.
In this tutorial, the case where the VASP code is used is considered.

Furthermore, the energies of the isolated atoms constituting the target structure must be evaluated.
In the case of VASP, the atomic energies calculated under standard settings can be obtained using the following command:

```shell
pypolymlp-utils --atomic_energy_elements Mg O --atomic_energy_functional PBE
```
Additional details are provided in [Utilities](utilities.md).


## 3. Estimate coefficients in polynomial MLP
Using a DFT-based dataset, a polynomial MLP is developed.
Let us now consider the case where the dataset located in the dataset in `examples/mlp_develop/vasp/MgO/single_dataset` directory is constructed from DFT calculations using the VASP.

Coefficients in the polynomial MLP model can be estimated using the following command:
```shell
> cd $(pypolymlp)/examples/mlp_develop/vasp/MgO/single_dataset
> polymlp -i polymlp.in
```
In this case, the polynomial MLP model is defined in the input file named `polymlp.in`.
After running the command, a polynomial MLP file named `polymlp.yaml` will be generated.

The polynomial MLP model in this example contains several parameters, as shown below:
```shell
> cat $(pypolymlp)/examples/mlp_develop/vasp/MgO/single_dataset/polymlp.in

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

data dataset/vasprun-*.xml.polymlp
```
In this model, the cutoff radius is set to 8.0 angstroms.
Polynomial invariants up to the third order are included.
The maximum degrees of the spherical harmonics for the second- and third-order terms are set to four.
Eight radial Gaussian functions are used to construct the polynomial invariants.
Furthermore, a second-order polynomial function of the polynomial invariants is used to model the energy.
For `model_type = 3`, only the second-order terms composed of pair invariants are included.

In this example, the dataset is automatically split into training and test sets.
If specific training and test datasets are provided, the `train_data` and `test_data` tags can be used.
For more details, see [Polynomial MLP development using the command line](mlpdev_command.md).

The parameters are defined as follows.
```python
"""
elements: Element species, (e.g., ['Mg','O'])
include_force: Considering force entries
include_stress: Considering stress entries
cutoff: Cutoff radius (Angstrom)
model_type: Polynomial function type
    model_type = 1: Linear polynomial of polynomial invariants
    model_type = 2: Polynomial of polynomial invariants
    model_type = 3: Polynomial of pair invariants
                    + linear polynomial of polynomial invariants
    model_type = 4: Polynomial of pair and second-order invariants
                    + linear polynomial of polynomial invariants
max_p: Order of polynomial function
feature_type: 'gtinv' or 'pair'
gaussian_params: Parameters for exp[- param1 * (r - param2)**2]
    Parameters are given as np.linspace(p[0], p[1], p[2]),
    where p[0], p[1], and p[2] are given by gaussian_params1
    and gaussian_params2.
reg_alpha_params: Parameters for penalty term in
    linear ridge regression. Parameters are given as
    np.linspace(p[0], p[1], p[2]).
gtinv_order: Maximum order of polynomial invariants.
gtinv_maxl: Maximum angular numbers of polynomial invariants.
    [maxl for order=2, maxl for order=3, ...]
atomic_energy: Atomic energies (in eV).
"""
```
