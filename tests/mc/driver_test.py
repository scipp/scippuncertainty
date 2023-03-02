# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from scipp_uncertainty.mc import PoissonDenseSampler, resample, resample_n


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
