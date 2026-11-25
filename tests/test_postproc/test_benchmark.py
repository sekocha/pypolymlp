"""Tests of benchmark."""

import os

from pypolymlp.postproc.benchmark.benchmark import PolymlpBenchmark


def test_benchmark():
    """Test PolymlpBenchmark."""
    bench = PolymlpBenchmark(use_test=True)
    bench.run(n_calc=1, size=(1, 1, 1))
    bench.save(filename="tmp.yaml")
    os.remove("tmp.yaml")
