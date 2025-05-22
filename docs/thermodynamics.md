# Thermodynamics calculations using polynomial MLP

1. MLP development

2. SSCHA calculations at various volumes and temperatures.
SSCHA calculations at a single volume and multiple temperatures can be performed as follows.
```shell
> pypolymlp-sscha --poscar POSCAR --pot polymlp.yaml --supercell 3 3 2 --temp_min 50 --temp_max 2000 --temp_step 50 --mixing 0.5 --tol 0.005
```

3. (Optional) Electronic free energy calculations using DFT.

3-1. Electronic calculations at various volumes using DFT.

3-2. Free energy calculation from vasprun.xml files.
```shell
> pypolymlp-utils --electron_vasprun */vasprun.xml --temp_max 2000 --temp_step 50

(joblib required.)
> pypolymlp-utils --electron_vasprun */vasprun.xml --temp_max 2000 --temp_step 50 --n_jobs -1
```

4. (Optional) Thermodynamic integration calculations at various volumes and temperatures.
In this thermodynamic integration, converged states of SSCHA calculations are used as reference states.

5.

```shell
> pypolymlp-thermodynamics --sscha ./sscha_runs/0*/sscha/*/sscha_results.yaml

# Include electronic contribution or thermodynamic integration contribution
> pypolymlp-thermodynamics --sscha ./sscha_runs/*/sscha/*/sscha_results.yaml --electron electrons/*/electron.yaml --ti sscha_runs/*/sscha/*/polymlp_ti.yaml
```
