# Pypolymlp calculators

If version 0.8.0 or earlier is used, polymlp files are generated in a text format as `polymlp.lammps`.
If a newer version (0.9.0 or later) is used, polymlp files are generated in YAML format  as `polymlp.yaml`, which can be utilized by replacing `polymlp.lammps` with `polymlp.yaml` in the following documentation.

## Properties (energies, forces, and stress tensors)

```shell
> pypolymlp-calc --properties --pot polymlp.lammps --poscars */POSCAR
> pypolymlp-calc --properties --pot polymlp.lammps --vaspruns vaspruns/vasprun.xml.polymlp.*
```

When using a hybrid polynomial MLP, multiple MLP files should be given for --pot option.
```shell
--pot polymlp.lammps*
or
--pot polymlp.lammps.1 polymlp.lammps.2
```

## Polynomial structural features

```shell
> pypolymlp-calc --features --pot polymlp.lammps --poscars */POSCAR
> pypolymlp-calc --features -i polymlp.in --poscars */POSCAR
```

## Force constants

(phonopy, phono3py, and symfc are required.)
```shell
> pypolymlp-calc --force_constants --pot polymlp.lammps --poscar POSCAR --supercell 3 3 2 --fc_n_samples 100 --disp 0.001 --fc_orders 2 3
```
If a cutoff radius is introduced to evaluate FC3s, use "--cutoff" option as follows.
```shell
> pypolymlp-calc --force_constants --pot polymlp.lammps --poscar POSCAR --geometry_optimization --fc_n_samples 300 --fc_orders 2 3 --disp 0.001 --batch_size 100 --supercell 3 3 2 --cutoff 6
```

## Local geometry optimization
```shell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --no_symmetry
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --fix_cell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --fix_atom
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --method CG
```

## Equation of states

(phonopy is required.)
```shell
> pypolymlp-calc --eos --poscar POSCAR --pot polymlp.lammps
```

## Elastic constant calculation

(pymatgen is required.)
```shell
> pypolymlp-calc --elastic --poscar POSCAR --pot polymlp.lammps
```

## Phonon calculations

(phonopy is required.)
```shell
> pypolymlp-calc --phonon --pot polymlp.lammps --poscar POSCAR --supercell 3 3 2 --ph_mesh 20 20 20
```

## Molecular dynamics calculations

(ASE and phonopy are required.)
```shell
> pypolymlp-md --poscar POSCAR --pot polymlp.yaml --supercell_size 3 3 3 --temp 300 --n_eq 5000 --n_steps 20000
```
