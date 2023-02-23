# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc

from scipp_uncertainty.mc import NormalDenseSampler, PoissonDenseSampler


@pytest.mark.parametrize('sampler_type',
                         [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_produce_no_variances(sampler_type):
    rng = np.random.default_rng(67176)
    da = sc.DataArray(
        sc.array(dims=['xx'],
                 values=rng.uniform(0.0, 1.0, 100),
                 variances=rng.uniform(0.001, 0.1, 100)))
    sampler = sampler_type(da)
    sample = sampler.sample_once(rng)

    assert sample.variances is None


@pytest.mark.parametrize('sampler_type',
                         [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_reproduce_input_parameters(sampler_type):
    rng = np.random.default_rng(8183)
    da = sc.DataArray(
        sc.array(dims=['some-dim'],
                 values=rng.uniform(0.0, 1.0, 100),
                 variances=rng.uniform(0.001, 0.1, 100),
                 unit='m'))
    sampler = sampler_type(da)
    sample = sampler.sample_once(rng)

    assert sample.dtype == da.dtype
    assert sample.sizes == da.sizes
    assert sample.unit == da.unit


@pytest.mark.parametrize('sampler_type',
                         [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_reproduce_input_metadata(sampler_type):
    rng = np.random.default_rng(512)
    da = sc.DataArray(sc.array(dims=['xx'],
                               values=rng.uniform(0.0, 1.0, 10),
                               variances=rng.uniform(0.001, 0.1, 10)),
                      coords={'xx': sc.arange('xx', 10, unit='s')},
                      masks={'m': sc.arange('xx', 10) < 5},
                      attrs={
                          'a': sc.scalar(8),
                          'b': sc.scalar('a string'),
                          'c': sc.arange('xx', 10) * 2
                      })
    sampler = sampler_type(da)
    sample = sampler.sample_once(rng)

    assert set(sample.coords.keys()) == set(da.coords.keys())
    assert sc.identical(sample.coords['xx'], da.coords['xx'])

    assert set(sample.masks.keys()) == set(da.masks.keys())
    assert sc.identical(sample.masks['m'], da.masks['m'])

    assert set(sample.attrs.keys()) == set(da.attrs.keys())
    assert sc.identical(sample.attrs['a'], da.attrs['a'])
    assert sc.identical(sample.attrs['b'], da.attrs['b'])
    assert sc.identical(sample.attrs['c'], da.attrs['c'])


@pytest.mark.parametrize('sampler_type',
                         [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_copy_results(sampler_type):
    rng = np.random.default_rng(2241)
    da = sc.DataArray(
        sc.array(dims=['xx'],
                 values=rng.uniform(0.0, 1.0, 100),
                 variances=rng.uniform(0.001, 0.1, 100)))
    sampler = sampler_type(da)

    sample0 = sampler.sample_once(rng)
    sample0_copy = sample0.copy(deep=True)
    sample1 = sampler.sample_once(rng)
    # sample0 has not been overwritten
    assert sc.identical(sample0, sample0_copy)
    # a new sample has been produced
    assert not sc.identical(sample1, sample0)


@pytest.mark.parametrize('sampler_type',
                         [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_disable_copy(sampler_type):
    rng = np.random.default_rng(561)
    da = sc.DataArray(
        sc.array(dims=['xx'],
                 values=rng.uniform(0.0, 1.0, 100),
                 variances=rng.uniform(0.001, 0.1, 100)))
    sampler = sampler_type(da, copy=False)

    sample0 = sampler.sample_once(rng)
    sample0_copy = sample0.copy(deep=True)
    sample1 = sampler.sample_once(rng)
    # sample0 has been overwritten
    assert not sc.identical(sample0, sample0_copy)
    assert sc.identical(sample0, sample1)
    # a new sample has been produced
    assert not sc.identical(sample1, sample0_copy)


@pytest.mark.parametrize('sampler_type',
                         [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_copy_input(sampler_type):
    rng = np.random.default_rng(2241)
    da = sc.DataArray(
        sc.array(dims=['xx'],
                 values=rng.uniform(0.0, 1.0, 100),
                 variances=rng.uniform(0.001, 0.1, 100)))
    sampler = sampler_type(da)
    orig = da.copy()
    da.values += 100
    samples = np.stack([sampler.sample_once(rng).values for _ in range(1000)])

    mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(mean, orig.values, atol=0.2)


@pytest.mark.parametrize('sampler_type',
                         [PoissonDenseSampler, NormalDenseSampler])
def test_dense_samplers_disable_input_copy(sampler_type):
    rng = np.random.default_rng(512)
    da = sc.DataArray(
        sc.array(dims=['xx'],
                 values=rng.uniform(0.0, 1.0, 100),
                 variances=rng.uniform(0.001, 0.1, 100)))
    sampler = sampler_type(da, copy_in=False)
    da.values += 100
    samples = np.stack([sampler.sample_once(rng).values for _ in range(1000)])

    mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(mean, da.values, atol=1)


def test_normal_dense_sampler_produces_expected_values():
    rng = np.random.default_rng(51232)
    da = sc.DataArray(
        sc.array(dims=['xx'],
                 values=rng.uniform(0.0, 1.0, 100),
                 variances=rng.uniform(0.0001, 0.01, 100)))

    sampler = NormalDenseSampler(da)
    samples = np.stack([sampler.sample_once(rng).values for _ in range(1000)])

    mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(mean, da.values, atol=0.01)

    var = np.var(samples, axis=0, ddof=1)
    np.testing.assert_allclose(var, da.variances, atol=0.01)


def test_poisson_dense_sampler_produces_expected_values():
    rng = np.random.default_rng(2231)
    da = sc.DataArray(sc.array(dims=['xx'], values=rng.poisson(2, 100)))

    sampler = PoissonDenseSampler(da)
    samples = np.stack([sampler.sample_once(rng).values for _ in range(1000)])

    mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(mean, da.values, atol=0.2)

    var = np.mean(samples, axis=0)
    np.testing.assert_allclose(var, da.values, atol=0.2)
