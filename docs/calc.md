# Pypolymlp calculators

If version 0.8.0 or earlier is used, polymlp files are generated in a text format as `polymlp.lammps`.
If a newer version (0.9.0 or later) is used, polymlp files are generated in YAML format  as `polymlp.yaml`, which can be utilized by replacing `polymlp.lammps` with `polymlp.yaml` in the following documentation.

## Properties (energies, forces, and stress tensors)

```
> pypolymlp-calc --properties --pot polymlp.lammps --poscars */POSCAR
> pypolymlp-calc --properties --pot polymlp.lammps --vaspruns vaspruns/vasprun.xml.polymlp.*
```

When using a hybrid polynomial MLP, multiple MLP files should be given for --pot option.
```
--pot polymlp.lammps*
or
--pot polymlp.lammps.1 polymlp.lammps.2
```

## Polynomial structural features

```
> pypolymlp-calc --features --pot polymlp.lammps --poscars */POSCAR
> pypolymlp-calc --features -i polymlp.in --poscars */POSCAR
```

## Force constants

(phonopy, phono3py, and symfc are required.)
```
> pypolymlp-calc --force_constants --pot polymlp.lammps --poscar POSCAR --supercell 3 3 2 --fc_n_samples 100 --disp 0.001 --fc_orders 2 3
```
If a cutoff radius is introduced to evaluate FC3s, use "--cutoff" option as follows.
```
pypolymlp-calc --force_constants --pot polymlp.lammps --poscar POSCAR --geometry_optimization --fc_n_samples 300 --fc_orders 2 3 --disp 0.001 --batch_size 100 --supercell 3 3 2 --cutoff 6
```

## Local geometry optimization
```
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --no_symmetry
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --fix_cell
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --fix_atom
> pypolymlp-calc --geometry_optimization --poscar POSCAR --pot polymlp.lammps --method CG
```

## Equation of states

(phonopy is required.)
```
> pypolymlp-calc --eos --poscar POSCAR --pot polymlp.lammps
```

## Elastic constants

(pymatgen is required.)
```
> pypolymlp-calc --elastic --poscar POSCAR --pot polymlp.lammps
```


<!--
> pypolymlp-calc --force_constants --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AlN.yaml.xz
-->

## Phonon calculations

(phonopy is required.)
```
> pypolymlp-calc --phonon --pot polymlp.lammps --poscar POSCAR --supercell 3 3 2 --ph_mesh 20 20 20
```
