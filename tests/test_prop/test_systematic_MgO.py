"""Tests of property calculations in MgO."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.properties_utils import convert_stresses_in_gpa
from pypolymlp.calculator.validation.validation import PolymlpValidation
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

pot_list = [
    path_file + "/mlps_systematic/polymlp.yaml.MgO.1",
    path_file + "/mlps_systematic/polymlp.yaml.MgO.2",
    path_file + "/mlps_systematic/polymlp.yaml.MgO.3",
    path_file + "/mlps_systematic/polymlp.yaml.MgO.4",
    path_file + "/mlps_systematic/polymlp.yaml.MgO.5",
    path_file + "/mlps_systematic/polymlp.yaml.MgO.6",
]
api_list = [Properties(pot=pot) for pot in pot_list]

poscar = path_file + "poscars/POSCAR.RS.MgO"
unitcell = Poscar(poscar).structure

etol = 1e-8
ftol = 1e-7
stol = 1e-6

true_e = [
    -40.22054040319249,
    -40.221732529620766,
    -40.22169719857264,
    -40.22156720693283,
    -40.221438636737915,
    -40.221736494011495,
]
true_f = [
    [
        [-0.03946803, -0.01183927, -0.07896111],
        [-0.00348044, 0.00320334, 0.02135077],
        [0.01067793, -0.00104418, 0.02135215],
        [0.01067498, 0.00320266, -0.00695989],
        [0.00915053, 0.00187107, 0.01247323],
        [0.00623762, 0.00274416, 0.01247479],
        [0.00623427, 0.00187030, 0.01832294],
        [-0.00002686, -0.00000810, -0.00005288],
    ],
    [
        [-0.03910993, -0.01173186, -0.07824443],
        [-0.00320102, 0.00317373, 0.02115474],
        [0.01057924, -0.00096035, 0.02115584],
        [0.01057688, 0.00317318, -0.00640114],
        [0.00987424, 0.00184498, 0.01230022],
        [0.00615051, 0.00296134, 0.01230132],
        [0.00614815, 0.00184443, 0.01976909],
        [-0.00101806, -0.00030544, -0.00203565],
    ],
    [
        [-0.03915201, -0.0117445, -0.07832823],
        [-0.00316905, 0.00317385, 0.02115538],
        [0.01057962, -0.00095075, 0.02115647],
        [0.01057729, 0.00317331, -0.00633727],
        [0.00988631, 0.00184932, 0.01232919],
        [0.00616501, 0.00296496, 0.01233032],
        [0.00616258, 0.00184876, 0.01979306],
        [-0.00104975, -0.00031495, -0.00209893],
    ],
    [
        [-0.03911412, -0.01173311, -0.07825299],
        [-0.00315067, 0.00318434, 0.02122533],
        [0.01061459, -0.00094523, 0.0212264],
        [0.01061229, 0.00318381, -0.00630062],
        [0.00984644, 0.00184194, 0.01228031],
        [0.00614042, 0.00295298, 0.01228139],
        [0.00613812, 0.00184141, 0.01971371],
        [-0.00108707, -0.00032615, -0.00217353],
    ],
    [
        [-0.03914689, -0.01174296, -0.07831801],
        [-0.00315319, 0.0032045, 0.02135952],
        [0.01068181, -0.00094598, 0.0213607],
        [0.01067927, 0.00320391, -0.0063058],
        [0.00973702, 0.00183448, 0.01223023],
        [0.00611559, 0.00292019, 0.01223145],
        [0.00611296, 0.00183387, 0.01949427],
        [-0.00102657, -0.00030801, -0.00205236],
    ],
    [
        [-0.03903950, -0.01171080, -0.07810204],
        [-0.00320404, 0.00319834, 0.02131811],
        [0.01066118, -0.00096125, 0.02131916],
        [0.01065893, 0.00319782, -0.00640728],
        [0.00986028, 0.00182940, 0.01219664],
        [0.00609851, 0.00295713, 0.01219753],
        [0.00609661, 0.00182896, 0.01974156],
        [-0.00113197, -0.00033960, -0.00226368],
    ],
]
true_s = [
    [0.01523251, 0.01399421, 0.01931511, -0.00003230, -0.00006463, -0.00021545],
    [0.03072673, 0.02947710, 0.03484693, -0.00000334, -0.00000672, -0.00002243],
    [0.03010376, 0.02885184, 0.03423150, -0.00000094, -0.00000191, -0.00000642],
    [0.03154739, 0.03030193, 0.03565386, 0.00000116, 0.00000228, 0.00000757],
    [0.02977576, 0.02852980, 0.03388370, -0.00000401, -0.00000804, -0.00002684],
    [0.02346943, 0.02223697, 0.02753304, -0.00000183, -0.00000371, -0.00001240],
]


@pytest.mark.parametrize("api, te, tf, ts", zip(api_list, true_e, true_f, true_s))
def test_eval(api, te, tf, ts):
    """Test properties in MgO."""
    e, f, s = api.eval(unitcell)
    assert e == pytest.approx(te, abs=etol)
    np.testing.assert_allclose(f.T, tf, atol=ftol)
    s = convert_stresses_in_gpa(np.array([s]), [unitcell])[0]
    np.testing.assert_allclose(s, ts, atol=stol)


@pytest.mark.parametrize("api, te, tf, ts", zip(api_list, true_e, true_f, true_s))
def test_rotation(api, te, tf, ts):
    """Test rotational invariance."""
    val = PolymlpValidation(unitcell, api)
    val.check_rotation(etol=1e-12)


@pytest.mark.parametrize("api, te, tf, ts", zip(api_list, true_e, true_f, true_s))
def test_translation(api, te, tf, ts):
    """Test translational invariance."""
    val = PolymlpValidation(unitcell, api)
    val.check_translation(etol=1e-12, ftol=1e-12, stol=1e-12)
