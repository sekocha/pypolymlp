# Pypolymlp utilities

## Compression of vasprun.xml files

```
> pypolymlp-utils --vasprun_compress vaspruns/vasprun-*.xml
```
Compressed vasprun.xml is generated as vasprun.xml.polymlp.

## Automatic division of DFT dataset

```
> pypolymlp-utils --auto_dataset dataset1/*/vasprun.xml dataset2/*/vasprun.xml
> cat polymlp.in.append >> polymlp.in
```
A given DFT dataset is automatically divided into some sets, depending on the values of the energy, the forces acting on atoms, and the volume.
A generated file "polymlp.in.append" can be appended in your polymlp.in, which will be used for developing MLPs.
Datasets identified with "train1" and "test1" are composed of structures with low energy and small force values.
The predictive power for them is more important than the other structures for the successive calculations using polynomial MLPs, so the prediction errors for "train1" and "test1" datasets should be accuracy measures for polynomial MLPs.

## Atomic energies
(Experimental: Only for VASP calculations using PBE and PBEsol functionals)

```
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

```
> pypolymlp-utils --calc_cost --pot polymlp.lammps

# hybrid polynomial MLP
> pypolymlp-utils --calc_cost --pot polymlp.lammps*
```

2. Multiple polynomial MLPs

```
> pypolymlp-utils --calc_cost -d $(path_mlps)/polymlp-00*
```

## Enumeration of optimal MLPs on convex hull

```
> pypolymlp-utils --find_optimal Ti-Pb/* --key test-disp1
```

Files 'polymlp_error.yaml' and 'polymlp_cost.yaml' are needed for each MLP.
