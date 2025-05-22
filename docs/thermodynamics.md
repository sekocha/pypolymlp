# Thermodynamics calculations using polynomial MLP

1. MLP development

2. Generate structures with various volumes.
```shell
> pypolymlp-structure -p POSCAR --isotropic 20 --min_volume 0.8 --max_volume 1.2
> mkdir -p sscha_results/{00001..00020}
> for i in {00001..00020};do mv poscars/poscar-00001 sscha_results/$i/POSCAR;done
```

3. Run SSCHA calculations at various volumes and temperatures.

SSCHA calculations at a single volume and multiple temperatures can be performed as follows.
```shell
> pypolymlp-sscha --poscar POSCAR --pot polymlp.yaml --supercell 3 3 2 --temp_min 50 --temp_max 2000 --temp_step 50 --mixing 0.5 --tol 0.005
```

4. (Optional) Electronic free energy calculations using DFT.

- DFT calculations at various volumes.

- Free energy and entropy calculation from vasprun.xml files.

```shell
> pypolymlp-utils --electron_vasprun */vasprun.xml --temp_max 2000 --temp_step 50

(joblib required.)
> pypolymlp-utils --electron_vasprun */vasprun.xml --temp_max 2000 --temp_step 50 --n_jobs -1
```

5. (Optional) Thermodynamic integration with MD calculations at various volumes and temperatures.

In this thermodynamic integration, converged states of SSCHA calculations are used as reference states.

6. Calculate thermodynamic properties from the precedent calculations.
```shell
> pypolymlp-thermodynamics --sscha ./sscha_results/0*/sscha/*/sscha_results.yaml

# Include electronic contribution or thermodynamic integration contribution
> pypolymlp-thermodynamics --sscha ./sscha_results/*/sscha/*/sscha_results.yaml --electron electrons/*/electron.yaml --ti sscha_runs/*/sscha/*/polymlp_ti.yaml
```
