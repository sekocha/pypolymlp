# Pypolymlp calculators

## Properties (energies, forces, and stress tensors)

```
> pypolymlp-calc --properties --pot polymlp.lammps --poscars */POSCAR
> pypolymlp-calc --properties --pot polymlp.lammps --vaspruns vaspruns/vasprun.xml.polymlp.*
> pypolymlp-calc --properties --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AgI.yaml.xz
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
> pypolymlp-calc --force_constants --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AlN.yaml.xz
> pypolymlp-calc --force_constants --pot polymlp.lammps --str_yaml polymlp_str.yaml --fc_n_samples 1000
> pypolymlp-calc --force_constants --pot polymlp.lammps --poscar POSCAR-unitcell --supercell 3 3 2 --fc_n_samples 1000
```

## Phonon calculations

(phonopy is required.)
```
> pypolymlp-calc --phonon --pot polymlp.lammps --poscar POSCAR-unitcell --supercell 3 3 2
```
