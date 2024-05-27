# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
import scipp.testing

from scippuncertainty.mc import CovarianceAccum, VarianceAccum


def test_variance_accum_returns_single_sample():
    da = sc.DataArray(
        sc.arange("w", 6.0, unit="kg"), coords={"h": sc.arange("w", 6) * 0.1}
    )
    accum = VarianceAccum()
    accum.add(da)

    res = accum.get()
    data, n_samples = res.data, res.n_samples

    assert n_samples == 1
    assert sc.identical(sc.values(data.data), da.data)
    assert sc.identical(sc.values(data.coords["h"]), da.coords["h"])
    # Cannot have variances with one sample
    assert sc.all(sc.isnan(sc.variances(data.data)))


def test_variance_accum_returns_expected_result():
    rng = np.random.default_rng(912)
    da = sc.DataArray(
        sc.array(dims=["observation", "p"], values=rng.normal(0.0, 1.0, (15, 7)))
    )

    accum = VarianceAccum()
    for i in range(15):
        accum.add(da["observation", i])

    res = accum.get().data
    np.testing.assert_allclose(res.values, sc.mean(da, dim="observation").values)
    np.testing.assert_allclose(res.variances, np.var(da.values, axis=0, ddof=1))


def test_variance_accum_returns_expected_result_2d():
    rng = np.random.default_rng(8341)
    da = sc.DataArray(
        sc.array(
            dims=["observation", "a", "b"], values=rng.normal(0.0, 1.0, (11, 3, 4))
        )
    )

    accum = VarianceAccum()
    for i in range(11):
        accum.add(da["observation", i])

    res = accum.get().data
    np.testing.assert_allclose(res.values, sc.mean(da, dim="observation").values)
    np.testing.assert_allclose(res.variances, np.var(da.values, axis=0, ddof=1))


def test_variance_accum_returns_number_of_samples():
    da = sc.DataArray(
        sc.arange("u", 3.0, unit="s"), coords={"o": sc.arange("u", 3) * 0.1}
    )
    accum = VarianceAccum()
    accum.add(da)
    accum.add(da)
    accum.add(da)
    accum.add(da)

    assert accum.get().n_samples == 4


def test_variance_accum_can_return_samples():
    da = sc.DataArray(
        sc.arange("u", 3.0, unit="s"), coords={"o": sc.arange("u", 3) * 0.1}
    )
    expected0 = da.copy()
    expected1 = 2 * da.copy()

    accum = VarianceAccum(keep_samples=True)
    accum.add(da)
    da *= 2  # doing in-place modification to test copy behavior
    accum.add(da)

    samples = accum.get().samples
    assert len(samples) == 2
    sc.testing.assert_identical(samples[0], expected0)
    sc.testing.assert_identical(samples[1], expected1)


def test_variance_accum_does_not_return_samples_if_disabled():
    da = sc.DataArray(
        sc.arange("u", 3.0, unit="s"), coords={"o": sc.arange("u", 3) * 0.1}
    )

    accum = VarianceAccum(keep_samples=False)
    accum.add(da)
    da *= 2  # doing in-place modification to test copy behavior
    accum.add(da)

    assert accum.get().samples is None


def test_variance_accum_preserves_metadata():
    rng = np.random.default_rng(83)
    da = sc.DataArray(
        sc.array(dims=["variable", "observation"], values=rng.normal(0.0, 1.0, (9, 2))),
        coords={
            "x": sc.arange("variable", 9, unit="kg"),
            "y": -sc.arange("variable", 9),
        },
        masks={"m": sc.arange("variable", 9) < 7},
    )
    da.coords.set_aligned("y", aligned=False)

    accum = VarianceAccum()
    accum.add(da["observation", 0])
    accum.add(da["observation", 1])
    res = accum.get().data

    assert set(res.coords.keys()) == {"x", "y"}
    assert sc.identical(res.coords["x"], sc.arange("variable", 9, unit="kg"))
    assert sc.identical(res.coords["y"], -sc.arange("variable", 9))
    assert res.coords["x"].aligned
    assert not res.coords["y"].aligned

    assert set(res.masks.keys()) == {"m"}
    assert sc.identical(res.masks["m"], sc.arange("variable", 9) < 7)


def test_variance_accum_add_from():
    rng = np.random.default_rng(83)
    da = sc.DataArray(
        sc.array(dims=["observation", "variable"], values=rng.normal(0.0, 1.0, (14, 8)))
    )
    chunk0 = da["observation", :5]
    chunk1 = da["observation", 5:7]
    chunk2 = da["observation", 7:]

    accum_total = VarianceAccum()
    accum0 = VarianceAccum()
    accum1 = VarianceAccum()
    accum2 = VarianceAccum()

    for i in range(14):
        accum_total.add(da["observation", i])
    for i in range(5):
        accum0.add(chunk0["observation", i])
    for i in range(2):
        accum1.add(chunk1["observation", i])
    for i in range(7):
        accum2.add(chunk2["observation", i])

    accum0.add_from(accum1)
    accum0.add_from(accum2)

    a = accum0.get().data
    b = accum_total.get().data
    np.testing.assert_allclose(a.values, b.values)
    np.testing.assert_allclose(a.variances, b.variances)


def test_variance_accum_new_does_not_return_stored_samples():
    da = sc.DataArray(
        sc.arange("u", 3.0, unit="s"), coords={"o": sc.arange("u", 3) * 0.1}
    )

    accum = VarianceAccum()
    accum.add(da)

    new = accum.new()
    with pytest.raises(RuntimeError):
        new.get()  # has no samples to get


@pytest.mark.parametrize("keep_samples", [True, False])
def test_variance_accum_new_passes_keep_samples_along(keep_samples):
    da = sc.DataArray(
        sc.arange("u", 3.0, unit="s"), coords={"o": sc.arange("u", 3) * 0.1}
    )

    accum = VarianceAccum(keep_samples=keep_samples)
    new = accum.new()
    new.add(da)

    assert (new.get().samples is not None) == keep_samples


def test_covariance_accum_returns_expected_result():
    rng = np.random.default_rng(512)
    da = sc.DataArray(
        sc.array(dims=["variable", "observation"], values=rng.normal(0.0, 1.0, (5, 19)))
    )

    accum = CovarianceAccum()
    for i in range(19):
        accum.add(da["observation", i])

    res = accum.get().data
    expected = np.cov(da.values)
    np.testing.assert_allclose(res.values, expected)
    assert res.variances is None


def test_covariance_accum_returns_expected_result_1_sample():
    rng = np.random.default_rng(32)
    da = sc.DataArray(sc.array(dims=["variable"], values=rng.normal(0.0, 1.0, 6)))

    accum = CovarianceAccum()
    accum.add(da)

    res = accum.get()
    assert sc.all(sc.isnan(res.data)).value
    assert res.n_samples == 1


def test_covariance_accum_returns_number_of_samples():
    rng = np.random.default_rng(823)
    da = sc.DataArray(sc.array(dims=["variable"], values=rng.normal(0.0, 1.0, 6)))

    accum = CovarianceAccum()
    accum.add(da)
    accum.add(da)
    accum.add(da)
    accum.add(da)
    accum.add(da)

    assert accum.get().n_samples == 5


def test_covariance_accum_erases_metadata():
    rng = np.random.default_rng(44)
    da = sc.DataArray(
        sc.array(dims=["variable", "observation"], values=rng.normal(0.0, 1.0, (9, 2))),
        coords={"x": sc.arange("variable", 9, unit="kg")},
        masks={"m": sc.arange("variable", 9) < 7},
        attrs={"a": sc.index(4), "b": sc.scalar("a string")},
    )

    accum = CovarianceAccum()
    accum.add(da["observation", 0])
    accum.add(da["observation", 1])
    res = accum.get().data

    assert not res.coords.keys()
    assert not res.masks.keys()


def test_covariance_accum_add_from():
    rng = np.random.default_rng(231)
    da = sc.DataArray(
        sc.array(dims=["variable", "observation"], values=rng.normal(0.0, 1.0, (5, 19)))
    )
    chunk0 = da["observation", :2]
    chunk1 = da["observation", 2:10]
    chunk2 = da["observation", 10:]

    accum_total = CovarianceAccum()
    accum0 = CovarianceAccum()
    accum1 = CovarianceAccum()
    accum2 = CovarianceAccum()

    for i in range(19):
        accum_total.add(da["observation", i])
    for i in range(2):
        accum0.add(chunk0["observation", i])
    for i in range(8):
        accum1.add(chunk1["observation", i])
    for i in range(9):
        accum2.add(chunk2["observation", i])

    accum0.add_from(accum1)
    accum0.add_from(accum2)

    a = accum0.get().data
    b = accum_total.get().data
    np.testing.assert_allclose(a.values, b.values)


def test_covariance_accum_new_does_not_return_stored_samples():
    rng = np.random.default_rng(412)
    da = sc.DataArray(sc.array(dims=["variable"], values=rng.normal(0.0, 1.0, 7)))
    accum = CovarianceAccum()
    accum.add(da)

    new = accum.new()
    with pytest.raises(RuntimeError):
        new.get()  # has no samples to get


@pytest.mark.parametrize("dims", [None, ("x", "y")])
def test_covariance_accum_new_passes_dims_along(dims):
    rng = np.random.default_rng(9)
    da = sc.DataArray(sc.array(dims=["variable"], values=rng.normal(0.0, 1.0, 8)))
    accum = CovarianceAccum(dims=dims)
    new = accum.new()

    accum.add(da)
    new.add(da)

    assert accum.get().data.dims == new.get().data.dims
