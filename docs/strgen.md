# Structure generator for DFT calculations

## Random atomic displacements from a single structure
- Constant magnitude of atomic displacements, Fixed cell
```
pypolymlp-structure -p POSCAR --displacements 10 --const_distance 0.001 --supercell 3 3 2
```

- Sequential magnitudes of atomic displacements, Fixed cell
```
pypolymlp-structure -p POSCAR --displacements 10 --max_distance 1.5 --supercell 3 3 2
```

- Sequential magnitudes of atomic displacements, Volume changes
```
pypolymlp-structure --poscars POSCAR-unitcell --displacements 10 --max_distance 1.5 --supercell 3 3 2 --n_volumes 5 --min_volume 0.8 --max_volume 1.2
```

## Random atomic displacements, cell expansions, and distortions from a single structures
```
pypolymlp-structure --poscars POSCAR --standard 100 --max_distance 1.5
```
The supercell size is automatically determined using --max_natom option.

Structures with low densities and those with high densities can also be generated as follows:
```
pypolymlp-structure --poscars POSCAR --standard 100 --max_distance 1.5 --low_density 10 --distance_density_mode 0.1 --high_density 10
```

## Structure generation from multiple prototype structures

1. Prototype structure selection

Prepare prototype structures in POSCAR format.

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
