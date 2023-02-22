# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Random number generators."""

from typing import List, Optional, Union

import numpy as np

from .logging import get_logger


def make_rngs(seed: Optional[Union[np.random.Generator,
                                   List[np.random.Generator], int, List[int],
                                   np.random.SeedSequence]], *,
              n: int) -> List[np.random.Generator]:
    """Instantiate new random number generators.

    Creates the given number of random generators using
    :func:`numpy.random.default_rng` with :class:`numpy.random.SeedSequence`.
    ``n`` separate seed sequences are spawned from ``seed`` and generators constructed
    from those.
    These generators are independent with high probability which makes them
    usable in a multithreaded context.

    The function logs the initial entropy and details of the RNGs
    so the RNGs can be re-created later.

    See `Parallel Random Number Generation
    <https://numpy.org/doc/stable/reference/random/parallel.html>`_
    in numpy for details.

    Note
    ----
    ``make_rngs(x, n=1)[0]`` is *not* the same as
    ``np.random.default_rng(x)`` but
    ``np.random.default_rng(np.random.SeedSequence(x).spawn(1)[0])``

    Parameters
    ----------
    seed:
        If a ``np.random.Generator`` or list thereof, return those generators.
        Otherwise, seed ``n`` generators with this.
    n:
        Number of generators to make.

    Returns
    -------
    :
        List of ``n`` random generators.
    """
    if isinstance(seed, np.random.Generator):
        seed = [seed]
    if isinstance(seed, list) and isinstance(seed[0], np.random.Generator):
        if len(seed) != n:
            raise ValueError(
                f'Got {len(seed)} random generators for {n} threads. '
                'Need exactly one RNG per thread.')
        get_logger().info('Using %d provided random generators', n)
        return seed

    if seed is None:
        seed = np.random.SeedSequence()
    elif not isinstance(seed, np.random.SeedSequence):
        seed = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in seed.spawn(n)]
    get_logger().info(
        'Seeding %d random generators with entropy %s.\n'
        'The generators are of type %s from numpy.random version %s', n,
        seed.entropy, rngs[0], np.__version__)
    return rngs
