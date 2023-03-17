# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np

from scippuncertainty.random import make_rngs


def test_make_rngs_results_are_independent():
    rngs = make_rngs(1460, n=4)
    numbers = [rng.integers(0, 1000, 10) for rng in rngs]
    assert not np.array_equal(numbers[0], numbers[1])
    assert not np.array_equal(numbers[0], numbers[2])
    assert not np.array_equal(numbers[0], numbers[3])
    assert not np.array_equal(numbers[1], numbers[2])
    assert not np.array_equal(numbers[1], numbers[3])
    assert not np.array_equal(numbers[2], numbers[3])


def test_make_rngs_returns_given_rng():
    rng0 = np.random.default_rng(761)
    rng0_copy = np.random.default_rng(761)
    rng1 = make_rngs(rng0, n=1)[0]

    # rng1 produces the same numbers as the input rng0 would have.
    a = rng0_copy.integers(0, 1000, 100)
    b = rng1.integers(0, 1000, 100)
    assert np.array_equal(a, b)

    # The state of rng0 has been advanced.
    c = rng0.integers(0, 1000, 100)
    assert not np.array_equal(b, c)
