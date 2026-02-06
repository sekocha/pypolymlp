"""Tests of post-calculations of SSCHA using API."""

import shutil
from pathlib import Path

from pypolymlp.api.pypolymlp_sscha_post import PypolymlpSSCHAPost

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

pot = path_file + "mlps/polymlp.yaml.gtinv.Al"


def test_sscha_Al():
    """Test SSCHA calculations from polymlp using API."""

    path_sscha = path_file + "others/sscha_restart/"
    yaml = path_sscha + "sscha_results.yaml"
    fc2 = path_sscha + "fc2.hdf5"

    sscha = PypolymlpSSCHAPost(verbose=True)
    sscha.init_structure_distribution(yamlfile=yaml, fc2file=fc2, pot=pot)
    sscha.run_structure_distribution(n_samples=10)
    sscha.save_structure_distribution(path="tmp", save_poscars=True)
    shutil.rmtree("tmp")
