# Calculators

## Properties (energies, forces, and stress tensors)

```
> pypolymlp --properties --pot polymlp.lammps --poscars */POSCAR
> pypolymlp --properties --pot polymlp.lammps --vaspruns vaspruns/vasprun.xml.polymlp.*
> pypolymlp --properties --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AgI.yaml.xz
```

## Polynomial structural features

```
> pypolymlp --features --pot polymlp.lammps --poscars */POSCAR
> pypolymlp --features -i polymlp.in --poscars */POSCAR
```

## Force constants 

(phonopy, phono3py, and symfc are required.)
```
> pypolymlp --force_constants --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AlN.yaml.xz
> pypolymlp --force_constants --pot polymlp.lammps --str_yaml polymlp_str.yaml --fc_n_samples 1000
> pypolymlp --force_constants --pot polymlp.lammps --poscar POSCAR-unitcell --supercell 3 3 2 --fc_n_samples 1000
```

## Phonon calculations

(phonopy is required.)
```
> pypolymlp --phonon --pot polymlp.lammps --poscar POSCAR-unitcell --supercell 3 3 2
```
