# Structure generator for DFT calculations

## Random atomic displacements from a single structure
- Constant magnitude of atomic displacements, Fixed cell
```shell
> pypolymlp-structure -p POSCAR --displacements 10 --const_distance 0.001 --supercell 3 3 2
```

- Sequential magnitudes of atomic displacements, Fixed cell
```shell
> pypolymlp-structure -p POSCAR --displacements 10 --max_distance 1.5 --supercell 3 3 2
```

- Sequential magnitudes of atomic displacements, Volume changes
```shell
> pypolymlp-structure --poscars POSCAR --displacements 10 --max_distance 1.5 --supercell 3 3 2 --n_volumes 5 --min_volume 0.8 --max_volume 1.2
```
The number of structures is n_volumes * n_displacements.

## Random structures from a single structures
- Random atomic displacements, cell expansions, and distortions are introduced by --standard option.
```shell
> pypolymlp-structure --poscars POSCAR --standard 100 --max_distance 1.5
```
The supercell size is automatically determined using --max_natom option.

- Structures with low densities and those with high densities can also be generated as follows:
```shell
> pypolymlp-structure --poscars POSCAR --standard 100 --max_distance 1.5 --low_density 10 --distance_density_mode 0.1 --high_density 10
```

## Random structures from multiple structures
```shell
> pypolymlp-structure --poscars prototypes/icsd-* --standard 30 --max_distance 1.5 --low_density 5 --distance_density_mode 0.1
```
Generated structures will be listed in polymlp_str_samples.yaml, and POSCAR files will be generated in "poscars" directory.

<!--
```
> pypolymlp-structure --prototype --n_types 1
> pypolymlp-structure --prototype --n_types 2 --comp 0.25 0.75
> pypolymlp-structure --prototype --n_types 2 --comp 1 3
> pypolymlp-structure --prototype --n_types 3
```
Only alloy structure types are available.
Selected prototypes are listed in polymlp_prototypes.yaml.
-->
