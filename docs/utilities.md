# Pypolymlp utilities

## Compression of vasprun.xml files

```shell
> pypolymlp-utils --vasprun_compress vaspruns/vasprun-*.xml
```
The compressed vasprun.xml files will be generated as `vasprun.xml.polymlp`.


## Calculation of electronic properties at finite temperatures from vasprun.xml files
(phonopy required.)
```shell
> pypolymlp-utils --electron_vasprun */vasprun.xml
```
The output files will be generated as `electron.yaml`.


## Automatic division of DFT dataset

```shell
> pypolymlp-utils --auto_dataset dataset1/*/vasprun.xml dataset2/*/vasprun.xml
> cat polymlp.in.append >> polymlp.in
```
A given DFT dataset is automatically divided into some sets, depending on the values of the energy, the forces acting on atoms, and the volume.
A generated file "polymlp.in.append" can be appended in your polymlp.in, which will be used for developing MLPs.
Datasets identified with "train1" and "test1" are composed of structures with low energy and small force values.
The predictive power for them is more important than the other structures for the successive calculations using polynomial MLPs, so the prediction errors for "train1" and "test1" datasets should be accuracy measures for polynomial MLPs.

## Atomic energies
(Experimental: Only for VASP calculations using PBE and PBEsol functionals)

```shell
> pypolymlp-utils --atomic_energy_elements Mg O --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula MgO --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula Al2O3 --atomic_energy_functional PBEsol
```

A standard output can be appended in your polymlp.in, which will be used for developing MLPs.
The polynomial MLP has no constant term, which means that the energy for isolated atoms is set to zero.
The energy values for the isolated atoms must be subtracted from the energy values for structures.

## Estimation of computational costs

calc_cost option generates a file 'polymlp_cost.yaml', which is required for finding optimal MLPs.

1. Single polynomial MLP

```shell
> pypolymlp-utils --calc_cost --pot polymlp.yaml

# hybrid polynomial MLP
> pypolymlp-utils --calc_cost --pot polymlp.yaml*
```

2. Multiple polynomial MLPs

Consider the following file and directory structures, which can be found in `examples/utils/mlp_opt`.

```shell
> ls Ag-Au
polymlp-00001/  polymlp-00003/  polymlp-00005/  polymlp-00007/  polymlp-00009/
polymlp-00002/  polymlp-00004/  polymlp-00006/  polymlp-00008/

> ls Ag-Au/polymlp-00*
test/polymlp-00001:
polymlp.yaml

test/polymlp-00002:
polymlp.yaml

test/polymlp-00003:
polymlp.yaml

test/polymlp-00004:
polymlp.yaml

test/polymlp-00005:
polymlp.yaml

test/polymlp-00006:
polymlp.yaml

test/polymlp-00007:
polymlp.yaml

test/polymlp-00008:
polymlp.yaml

test/polymlp-00009:
polymlp.yaml
```

In this case, computational costs for multiple polynomial MLPs can be estimated as follows.
```shell
> pypolymlp-utils --calc_cost -d Ag-Au/polymlp-00*
```

## Enumeration of optimal MLPs on convex hull
Consider the following file and directory structures, which can be found in `examples/utils/mlp_opt`.

```shell
> ls Ag-Au
polymlp-00001/  polymlp-00003/  polymlp-00005/  polymlp-00007/  polymlp-00009/
polymlp-00002/  polymlp-00004/  polymlp-00006/  polymlp-00008/

Ag-Au/polymlp-00001:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00002:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00003:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00004:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00005:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00006:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00007:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00008:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml

Ag-Au/polymlp-00009:
polymlp.yaml  polymlp_cost.yaml  polymlp_error.yaml
```
Files `polymlp_error.yaml` and `polymlp_cost.yaml` are needed for each MLP.

In this case, optimal MLPs on the convex hull can be found as follows.
```shell
> pypolymlp-utils --find_optimal Ag-Au/* --key test-disp1
```
