"""Tests of thermodynamics_utils."""

from pathlib import Path

import pytest

from pypolymlp.calculator.thermodynamics.thermodynamics_utils import FittedModels
from pypolymlp.calculator.utils.eos_utils import EOS
from pypolymlp.calculator.utils.fit_utils import Polyfit

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/others/thermodynamics/"


def test_FittedModels(thermodynamics_grids_Cu):
    """Test FittedModels."""
    grid, _, _, _, _ = thermodynamics_grids_Cu
    fv_fits = grid.fit_free_energy_volume()
    sv_fits = grid.fit_entropy_volume()
    models = FittedModels(
        volumes=grid.volumes,
        temperatures=grid.temperatures,
        fv_fits=fv_fits,
        sv_fits=sv_fits,
    )
    fv, sv, cv = models.extract(3)
    assert isinstance(fv, EOS)
    assert isinstance(sv, Polyfit)
    assert cv is None
    assert not models._check_errors(3)

    assert models.eval_eq_free_energy(3) == pytest.approx(-4.0817623942196715)
    assert models.eval_eq_entropy(3) == pytest.approx(0.0003279106689341045)

    volumes = [20, 22]
    assert models.eval_helmholtz_free_energies(volumes).shape == (2, 11)
    assert models.eval_entropies(volumes).shape == (2, 11)
    gibbs = models.eval_gibbs_free_energies(volumes)
    assert gibbs.shape == (11, 2, 2)
    assert gibbs[3, 1, 1] == pytest.approx(-5.0137212766888)
