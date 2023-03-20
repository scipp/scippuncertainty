# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc

from scippuncertainty import pearson_correlation


def test_pearson_correlation_same_sizes_as_input_variable():
    cov = sc.ones(sizes={"kwk": 4, "o al": 4})
    r = pearson_correlation(cov)
    assert r.sizes == cov.sizes


def test_pearson_correlation_same_sizes_as_input_data_array():
    cov = sc.DataArray(sc.ones(sizes={"t 2": 3, ",": 3}))
    r = pearson_correlation(cov)
    assert r.sizes == cov.sizes


def test_pearson_correlation_preserves_coords():
    cov = sc.DataArray(
        sc.ones(sizes={"c1": 4, "c/2": 4}),
        coords={"c1": sc.arange("c1", 4, unit="m"), "c/2": 2 * sc.arange("c/2", 4)},
    )
    r = pearson_correlation(cov)
    assert r.coords == cov.coords


def test_pearson_correlation_result_is_dimensionless():
    cov = sc.ones(sizes={"551": 5, "y": 5}, unit="kg^2")
    r = pearson_correlation(cov)
    assert r.unit == "one"


def test_pearson_correlation_diagonal_is_one():
    rng = np.random.default_rng(7891789)
    cov = sc.array(dims=["d1", "d2"], values=rng.uniform(0.01, 2.3, (11, 11)))
    r = pearson_correlation(cov)
    np.testing.assert_allclose(np.diag(r.values), np.ones(11))


def test_pearson_correlation_requires_square_array():
    with pytest.raises(sc.DimensionError):
        pearson_correlation(sc.ones(sizes={"a": 3, "b": 4}))


def test_pearson_correlation_requires_2d_array():
    with pytest.raises(sc.DimensionError):
        pearson_correlation(sc.ones(sizes={"j": 3}))
    with pytest.raises(sc.DimensionError):
        pearson_correlation(sc.ones(sizes={"l": 2, "a": 2, "b": 2}))
