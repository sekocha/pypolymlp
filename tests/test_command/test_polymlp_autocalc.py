"""Test func_calc for APIs."""

from pathlib import Path

from pypolymlp.api.run_polymlp_autocalc import _parse_args_pypolymlp_autocalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

pot = path_file + "polymlp.yaml.pair.Ag"


def test_parse_args_pypolymlp_autocalc():
    """Test _parse_args_pypolymlp_autocalc."""
    args = _parse_args_pypolymlp_autocalc(["--pot", pot])
    assert args.pot[0] == pot
