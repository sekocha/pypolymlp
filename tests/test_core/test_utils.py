"""Tests of utils.py."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.utils import rmse, split_ids_train_test, split_train_test, strtobool

cwd = Path(__file__).parent


def test_strtobool():
    """Test for strtobool."""
    assert strtobool("True") == True
    assert strtobool("true") == True
    assert strtobool("t") == True
    assert strtobool("False") == False
    assert strtobool("false") == False
    assert strtobool("f") == False


def test_split_ids_train_test():
    """Test for split_ids_train_test."""
    train_ids, test_ids = split_ids_train_test(100, train_ratio=0.9)
    assert len(train_ids) == 90
    assert len(test_ids) == 10


def test_split_train_test():
    """Test for split_train_test."""
    files = ["file-" + str(i) for i in range(100)]
    train, test = split_train_test(files, train_ratio=0.8)
    assert len(train) == 80
    assert len(test) == 20


def test_rmse():
    """Test for rmse."""
    y1 = np.array([0.3, 0.2, 0.1])
    y2 = np.array([0.35, 0.18, 0.09])
    assert rmse(y1, y2) == pytest.approx(0.03162277660168379)
