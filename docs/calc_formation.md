# Formation energy calculation

## Using Python API

```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_poscars(["POSCAR", "POSCAR2", "POSCAR3"])

polymlp.init_formation_energy(end_structures=end_structures)
formation_energies = polymlp.run_formation_energy()
```
