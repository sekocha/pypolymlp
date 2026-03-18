# Other Utilities

## Converter for Potential Files from Legacy to Latest Format

```shell
> pypolymlp-utils --yaml_converter polymlp.lammps
```

## Generation of Supercell

> **Note:** Requires Phonopy for the use of general matrix.

```shell
# Diagonal supercell matrix
> pypolymlp-utils --poscar POSCAR --supercell 2 2 2

# General supercell matrix, A' = A M
# M = [[2, 0, 0], [1, 2, 0], [1, 1, 2]]
> pypolymlp-utils --poscar POSCAR --supercell 2 0 0 1 2 0 1 1 2
```

## Calculation of Electronic Properties at Finite Temperatures from `vasprun.xml` Files

> **Note:** Requires Phonopy.

```shell
> pypolymlp-utils --electron_vasprun */vasprun.xml
```
The output files will be generated as `electron.yaml`.
