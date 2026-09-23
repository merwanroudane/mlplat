import numpy as np
import pytest

from utils.datasets import DATASET_INFO, load_dataset, make_plr, toy_2d, xy


@pytest.mark.parametrize("name", list(DATASET_INFO))
def test_dataset_loads_deterministically(name):
    a, b = load_dataset(name), load_dataset(name)
    assert len(a) > 0 and a.equals(b)
    assert DATASET_INFO[name].target in a.columns


def test_supervised_xy_drops_truth_columns():
    X, y = xy("causal")
    assert "tau" not in X and "true_ps" not in X and y.name == "y"


def test_plr_truth_recoverable_by_oracle():
    df = make_plr(n=20000, nonlinearity=0.0, seed=1)
    X = np.c_[np.ones(len(df)), df["d"], df.filter(like="x")]
    beta = np.linalg.lstsq(X, df["y"], rcond=None)[0]
    assert abs(beta[1] - 0.5) < 0.03  # linear DGP: OLS with controls is consistent


def test_no_exact_collinearity_in_classification():
    X, _ = xy("classification")
    assert np.linalg.cond(X.to_numpy()) < 1e4


@pytest.mark.parametrize("kind", ["moons", "circles", "linear", "xor", "blobs"])
def test_toy_2d(kind):
    X, y = toy_2d(kind, n=100)
    assert X.shape == (100, 2) and len(y) == 100
