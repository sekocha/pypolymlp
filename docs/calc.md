# Pypolymlp calculators

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

<!--
> pypolymlp-calc --force_constants --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AlN.yaml.xz
-->

## Phonon calculations

(phonopy is required.)
```
> pypolymlp-calc --phonon --pot polymlp.lammps --poscar POSCAR --supercell 3 3 2 --ph_mesh 20 20 20
```
