"""Tests of SSCHA calculations using API."""

from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"


def test_sscha_Al():
    """Test SSCHA calculations from polymlp using API."""

    sscha = PypolymlpSSCHA(verbose=True)
    sscha.load_poscar(poscar, (2, 2, 2))
    sscha.set_polymlp(pot)

    sscha.run(temp=700, tol=0.003, mixing=0.5)
    _ = sscha.sscha_params
    props = sscha.sscha_properties
    _ = sscha.sscha_logs
    fc2 = sscha.force_constants
    assert fc2.shape == (32, 32, 3, 3)

    assert props.static_potential == pytest.approx(-1322.92893961425, rel=1e-8)
    assert props.harmonic_potential == pytest.approx(34.50552229486996, rel=1e-2)
    assert props.harmonic_free_energy == pytest.approx(-59.06006839696035, rel=1e-2)
    assert props.average_potential == pytest.approx(32.438167553114006, rel=1e-2)
    assert props.anharmonic_free_energy == pytest.approx(-2.0673547417559606, rel=1e-1)
    assert props.free_energy == pytest.approx(-61.12742313871631, rel=1e-2)
    assert props.entropy == pytest.approx(185.98197787014593, rel=1e-2)
    assert props.harmonic_heat_capacity == pytest.approx(97.95462787508784, rel=1e-2)
    assert props.delta < 0.003
    assert props.converge
    assert not props.imaginary


def test_sscha_Al_restart():
    """Test SSCHA calculations from polymlp using API."""
    # sscha.load_restart(yaml=args.yaml, parse_fc2=True)
    # sscha.run(
    #     temp=700,
    #     temp_min=args.temp_min,
    #     temp_max=args.temp_max,
    #     temp_step=args.temp_step,
    #     n_temp=args.n_temp,
    #     ascending_temp=args.ascending_temp,
    #     n_samples_init=n_samples_init,
    #     n_samples_final=n_samples_final,
    #     tol=args.tol,
    #     max_iter=args.max_iter,
    #     mixing=args.mixing,
    #     mesh=args.mesh,
    #     init_fc_algorithm=args.init,
    #     init_fc_file=args.init_file,
    #     cutoff_radius=args.cutoff_fc2,
    #     use_temporal_cutoff=args.use_temporal_cutoff,
    # )
