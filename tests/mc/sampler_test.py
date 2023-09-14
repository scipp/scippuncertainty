# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
import scipp.testing

from scippuncertainty.mc import NormalDenseSampler, PoissonDenseSampler


@pytest.mark.parametrize("sampler_type", [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_produce_no_variances(sampler_type):
    rng = np.random.default_rng(67176)
    da = sc.DataArray(
        sc.array(
            dims=["xx"],
            values=rng.uniform(0.0, 1.0, 100),
            variances=rng.uniform(0.001, 0.1, 100),
        )
    )
    sampler = sampler_type(da)
    sample = sampler.sample_once(rng)

    assert sample.variances is None


@pytest.mark.parametrize("sampler_type", [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_reproduce_input_parameters(sampler_type):
    rng = np.random.default_rng(8183)
    da = sc.DataArray(
        sc.array(
            dims=["some-dim"],
            values=rng.uniform(0.0, 1.0, 100),
            variances=rng.uniform(0.001, 0.1, 100),
            unit="m",
        )
    )
    sampler = sampler_type(da)
    sample = sampler.sample_once(rng)

    assert sample.dtype == da.dtype
    assert sample.sizes == da.sizes
    assert sample.unit == da.unit


@pytest.mark.parametrize("sampler_type", [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_reproduce_input_metadata(sampler_type):
    rng = np.random.default_rng(512)
    da = sc.DataArray(
        sc.array(
            dims=["xx"],
            values=rng.uniform(0.0, 1.0, 10),
            variances=rng.uniform(0.001, 0.1, 10),
        ),
        coords={
            "xx": sc.arange("xx", 10, unit="s"),
            "a": -sc.arange("xx", 10, unit="s"),
        },
        masks={"m": sc.arange("xx", 10) < 5},
    )
    da.coords.set_aligned("a", aligned=False)
    sampler = sampler_type(da)
    sample = sampler.sample_once(rng)

    sc.testing.assert_identical(sample.coords, da.coords)
    sc.testing.assert_identical(sample.masks, da.masks)


@pytest.mark.parametrize("sampler_type", [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_copy_results(sampler_type):
    rng = np.random.default_rng(2241)
    da = sc.DataArray(
        sc.array(
            dims=["xx"],
            values=rng.uniform(0.0, 1.0, 100),
            variances=rng.uniform(0.001, 0.1, 100),
        )
    )
    sampler = sampler_type(da)

    sample0 = sampler.sample_once(rng)
    sample0_copy = sample0.copy(deep=True)
    sample1 = sampler.sample_once(rng)
    # sample0 has not been overwritten
    assert sc.identical(sample0, sample0_copy)
    # a new sample has been produced
    assert not sc.identical(sample1, sample0)


@pytest.mark.parametrize("sampler_type", [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_disable_copy(sampler_type):
    rng = np.random.default_rng(561)
    da = sc.DataArray(
        sc.array(
            dims=["xx"],
            values=rng.uniform(0.0, 1.0, 100),
            variances=rng.uniform(0.001, 0.1, 100),
        )
    )
    sampler = sampler_type(da, copy=False)

    sample0 = sampler.sample_once(rng)
    sample0_copy = sample0.copy(deep=True)
    sample1 = sampler.sample_once(rng)
    # sample0 has been overwritten
    assert not sc.identical(sample0, sample0_copy)
    assert sc.identical(sample0, sample1)
    # a new sample has been produced
    assert not sc.identical(sample1, sample0_copy)


@pytest.mark.parametrize("sampler_type", [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_copy_input(sampler_type):
    rng = np.random.default_rng(2241)
    da = sc.DataArray(
        sc.array(
            dims=["xx"],
            values=rng.uniform(0.0, 1.0, 100),
            variances=rng.uniform(0.001, 0.1, 100),
        )
    )
    sampler = sampler_type(da)
    orig = da.copy()
    da.values += 100
    samples = np.stack([sampler.sample_once(rng).values for _ in range(1000)])

    mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(mean, orig.values, atol=0.2)


@pytest.mark.parametrize("sampler_type", [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_disable_input_copy(sampler_type):
    rng = np.random.default_rng(512)
    da = sc.DataArray(
        sc.array(
            dims=["xx"],
            values=rng.uniform(0.0, 1.0, 100),
            variances=rng.uniform(0.001, 0.1, 100),
        )
    )
    sampler = sampler_type(da, copy_in=False)
    da.values += 100
    samples = np.stack([sampler.sample_once(rng).values for _ in range(1000)])

    mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(mean, da.values, atol=1)


def test_normal_dense_sampler_produces_expected_values():
    rng = np.random.default_rng(51232)
    da = sc.DataArray(
        sc.array(
            dims=["xx"],
            values=rng.uniform(0.0, 1.0, 100),
            variances=rng.uniform(0.0001, 0.01, 100),
        )
    )

    sampler = NormalDenseSampler(da)
    samples = np.stack([sampler.sample_once(rng).values for _ in range(1000)])

    mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(mean, da.values, atol=0.01)

    var = np.var(samples, axis=0, ddof=1)
    np.testing.assert_allclose(var, da.variances, atol=0.01)


def test_poisson_dense_sampler_produces_expected_values():
    rng = np.random.default_rng(2231)
    da = sc.DataArray(sc.array(dims=["xx"], values=rng.poisson(2, 100)))

    sampler = PoissonDenseSampler(da)
    samples = np.stack([sampler.sample_once(rng).values for _ in range(1000)])

    mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(mean, da.values, atol=0.2)

    var = np.mean(samples, axis=0)
    np.testing.assert_allclose(var, da.values, atol=0.2)


@pytest.mark.parametrize("sampler_type", [PoissonDenseSampler, NormalDenseSampler])
@pytest.mark.parametrize("copy_in", [True, False])
def test_cloned_dense_sampler_produces_same_values_as_original(sampler_type, copy_in):
    rng = np.random.default_rng(82646)
    da = sc.DataArray(
        sc.array(
            dims=["xx"],
            values=rng.uniform(0.0, 1.0, 100),
            variances=rng.uniform(0.001, 0.1, 100),
        )
    )

    orig_sampler = sampler_type(da, copy_in=copy_in)
    cloned_sampler = orig_sampler.clone()

    rng_orig = np.random.default_rng(41828)
    rng_clone = np.random.default_rng(41828)
    orig_sample0 = orig_sampler.sample_once(rng_orig)
    cloned_sample0 = cloned_sampler.sample_once(rng_clone)
    cloned_sample1 = cloned_sampler.sample_once(rng_clone)
    orig_sample1 = orig_sampler.sample_once(rng_orig)

    assert sc.identical(orig_sample0, cloned_sample0)
    assert sc.identical(orig_sample1, cloned_sample1)


@pytest.mark.parametrize("sampler_type", [PoissonDenseSampler, NormalDenseSampler])
@pytest.mark.parametrize("copy_in", [True, False])
@pytest.mark.parametrize("copy_out", [True, False])
def test_cloned_dense_samplers_output_is_independent_of_orig(
    sampler_type, copy_in, copy_out
):
    # Data returned by the cloned sampler does not share memory with
    # data returned by the original.
    rng = np.random.default_rng(98283)
    da = sc.DataArray(
        sc.array(
            dims=["xx"],
            values=rng.uniform(2.0, 3.0, 69),
            variances=rng.uniform(0.01, 0.1, 69),
        )
    )

    orig_sampler = sampler_type(da, copy_in=copy_in, copy=copy_out)
    cloned_sampler = orig_sampler.clone()

    orig_sample = orig_sampler.sample_once(rng)
    cloned_sample = cloned_sampler.sample_once(rng)

    orig_sample_copy = orig_sample.copy(deep=True)
    cloned_sample[4] = 6.412
    cloned_sample.unit = "kg"

    assert not sc.identical(orig_sample, cloned_sample)
    assert sc.identical(orig_sample, orig_sample_copy)
