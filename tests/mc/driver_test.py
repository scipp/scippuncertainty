# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from scipp_uncertainty.mc import (
    NormalDenseSampler,
    PoissonDenseSampler,
    SkipSample,
    VarianceAccum,
    resample,
    resample_n,
    run,
)


def test_resample_draws_from_samplers():
    rng = np.random.default_rng(8283)
    da = sc.DataArray(sc.array(dims=["uy"], values=rng.poisson(2, 11)))
    samplers = {"hse": PoissonDenseSampler(da), "olk": PoissonDenseSampler(da)}

    rng = np.random.default_rng(129)
    r = iter(resample(samplers=samplers, rng=rng))
    samples0 = next(r)
    samples1 = next(r)

    # This checks if the RNG is used in the expected order.
    rng = np.random.default_rng(129)
    expected0 = {
        "hse": samplers["hse"].sample_once(rng),
        "olk": samplers["olk"].sample_once(rng),
    }
    expected1 = {
        "hse": samplers["hse"].sample_once(rng),
        "olk": samplers["olk"].sample_once(rng),
    }

    assert samples0.keys() == expected0.keys()
    assert sc.identical(samples0["hse"], expected0["hse"])
    assert sc.identical(samples0["olk"], expected0["olk"])
    assert samples1.keys() == expected1.keys()
    assert sc.identical(samples1["hse"], expected1["hse"])
    assert sc.identical(samples1["olk"], expected1["olk"])


def test_resample_n_produces_expected_samples():
    rng = np.random.default_rng(231)
    da = sc.DataArray(sc.array(dims=["uy"], values=rng.poisson(3, 10)))
    samplers = {"qq": PoissonDenseSampler(da), "hah": PoissonDenseSampler(da)}

    rng = np.random.default_rng(9)
    samples0, samples1 = resample_n(samplers=samplers, n=2, rng=rng)

    # This checks if the RNG is used in the expected order.
    rng = np.random.default_rng(9)
    expected0 = {
        "qq": samplers["qq"].sample_once(rng),
        "hah": samplers["hah"].sample_once(rng),
    }
    expected1 = {
        "qq": samplers["qq"].sample_once(rng),
        "hah": samplers["hah"].sample_once(rng),
    }

    assert samples0.keys() == expected0.keys()
    assert sc.identical(samples0["qq"], expected0["qq"])
    assert sc.identical(samples0["hah"], expected0["hah"])
    assert samples1.keys() == expected1.keys()
    assert sc.identical(samples1["qq"], expected1["qq"])
    assert sc.identical(samples1["hah"], expected1["hah"])


@pytest.mark.parametrize("n_threads", range(1, 5))
def test_run_produces_expected_results(n_threads):
    def f(d):
        # Does not introduce correlations, so calling this function with variances
        # produces the same result as using MC.
        return {"r": d / d.coords["s"]}

    rng = np.random.default_rng(512)
    n = 10
    da = sc.DataArray(
        sc.array(
            dims=["x"],
            values=rng.normal(4.0, 0.5, n),
            variances=rng.uniform(0.01, 0.1, n),
            unit="m",
        ),
        coords={"x": sc.arange("x", n, dtype="float64"), "s": sc.scalar(2.0, unit="s")},
    )

    expected = f(da)["r"]
    mc_dict = run(
        f,
        n_samples=1000,
        samplers={"d": NormalDenseSampler(da)},
        accumulators={"r": VarianceAccum()},
        seed=9381,
        n_threads=n_threads,
        progress=False,
    )
    assert mc_dict.keys() == {"r"}
    mc = mc_dict["r"]

    assert set(mc.coords.keys()) == set(expected.coords.keys())
    assert sc.identical(mc.coords["x"], expected.coords["x"])
    assert sc.identical(mc.coords["s"], expected.coords["s"])
    assert mc.sizes == expected.sizes
    assert mc.unit == expected.unit
    # Tolerance is not expected to be better than 1/sqrt(n_samples).
    np.testing.assert_allclose(mc.values, expected.values, atol=1e-2)
    np.testing.assert_allclose(mc.variances, expected.variances, atol=1e-2)


def test_run_can_skip_samples():
    i = 0

    def f(d):
        nonlocal i
        i += 1
        if i % 2 == 0:
            return SkipSample
        return {"r": d}

    rng = np.random.default_rng(31)
    n = 10
    da = sc.DataArray(
        sc.array(
            dims=["x"],
            values=rng.normal(4.0, 0.5, n),
            variances=rng.uniform(0.01, 0.1, n),
        )
    )

    res = run(
        f,
        n_samples=20,
        samplers={"d": NormalDenseSampler(da)},
        accumulators={"r": VarianceAccum()},
        seed=412,
        n_threads=1,
        progress=False,
    )

    assert res["r"].attrs["n_samples"] == sc.index(10)
