# How to use SSCHA utilities

```
python3 ~/git/pypolymlp/src/pypolymlp/calculator/sscha/utils/distribution.py --yaml sscha/2000/sscha_results.yaml --fc2 sscha/2000/fc2.hdf5
```

```
python3 ~/git/pypolymlp/src/pypolymlp/calculator/sscha/utils/compute_fc3.py --yaml sscha/2000/sscha_results.yaml --fc2 sscha/2000/fc2.hdf5
```

```
python3 ~/git/pypolymlp/src/pypolymlp/calculator/sscha/utils/summary.py
```

```
python3 ~/git/pypolymlp/src/pypolymlp/str_gen/strgen_volume.py --poscar poscar_eqm
```

```
python3 ~/git/pypolymlp/src/pypolymlp/calculator/sscha/utils/summary_eos.py --electronic electronic_free_energy.dat
```

```
python3 ~/git/pypolymlp/src/pypolymlp/calculator/sscha/utils/find_tc.py --yaml bcc/free_energy.yaml hcp/free_energy.yaml
```
```
python3 ~/git/pypolymlp/src/pypolymlp/calculator/sscha/utils/find_phase_boundary.py --yaml bcc/volume_dependence_with_elf_noimag.yaml hcp/volume_dependence_with_elf_noimag.yaml
```
