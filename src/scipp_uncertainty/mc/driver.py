# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Control bootstrap resampling."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import Dict, Generator, List, Optional, Union

import numpy as np
import scipp as sc

from .._progress import Progress, SilentProgress, progress_bars
from ..random import make_rngs
from .accumulator import VarianceAccum
from .sampler import Sampler


def _n_samples_per_thread(n_samples, n_thread):
    base = [n_samples // n_thread] * n_thread
    for i in range(n_samples % n_thread):
        base[i] += 1
    if sum(base) != n_samples:
        # This should not happen, only an internal debug check
        raise RuntimeError('Cannot distribute samples over threads')
    return base


class Bootstrap:
    """Perform a bootstrap analysis."""

    def __init__(self,
                 samplers: dict,
                 *,
                 seed: Optional[Union[np.random.Generator,
                                      List[np.random.Generator], int,
                                      List[int],
                                      np.random.SeedSequence]] = None,
                 n_threads: Optional[int] = None,
                 keep_samples=None,
                 accumulators: Optional[dict] = None):
        if n_threads is None:
            n_threads = max(multiprocessing.cpu_count(), 4)
        self._rngs = make_rngs(seed, n=n_threads)

        self._samplers = samplers
        self._keep_samples = set() if not keep_samples else set(keep_samples)
        self._accumulator_types = accumulators

    def _make_accum(self, name, **kwargs):
        if self._accumulator_types is None:
            return VarianceAccum(**kwargs)
        return self._accumulator_types[name](**kwargs)

    def _sample_once(self, rng):
        return {
            key: sampler.sample_once(rng)
            for key, sampler in self._samplers.items()
        }

    def _run_batch(self, fn, *, n_samples, rng, progress_bar) -> dict:
        accumulators = None
        for _ in progress_bar.track(range(n_samples),
                                    description='Bootstrapping...'):
            res = fn(**self._sample_once(rng))
            if res is None:
                continue
            if accumulators is None:
                accumulators = {
                    name: self._make_accum(name,
                                           keep_samples=name
                                           in self._keep_samples)
                    for name in res.keys()
                }
            for n, r in res.items():
                accumulators[n].add(r)
        return accumulators

    def run(self, fn, *, n_samples):
        """Run resampling."""
        results = []
        with progress_bars(visible=None).prepare(len(self._rngs)) as prog:
            with ThreadPoolExecutor(max_workers=len(self._rngs)) as executor:
                for rng, n_thread_samples, progress_bar in zip(
                        self._rngs,
                        _n_samples_per_thread(n_samples, len(self._rngs)),
                        prog):
                    results.append(
                        executor.submit(self._run_batch,
                                        fn,
                                        n_samples=n_thread_samples,
                                        rng=rng,
                                        progress_bar=progress_bar))

        accumulators = results[0].result()
        for res in results[1:]:
            for a, b in zip(accumulators.values(), res.result().values()):
                a.add_from(b)
        return {name: accum.get() for name, accum in accumulators.items()}


def _resample_once(samplers: Dict[str, Sampler],
                   rng: np.random.Generator) -> Dict[str, sc.DataArray]:
    return {key: sampler.sample_once(rng) for key, sampler in samplers.items()}


def resample(
    *, samplers: Dict[str, Sampler], rng: np.random.Generator
) -> Generator[Dict[str, sc.DataArray], None, None]:
    """Draw samples from given samplers forever."""
    while True:
        yield _resample_once(samplers, rng)


def resample_n(
    *,
    samplers: Dict[str, Sampler],
    rng: np.random.Generator,
    n: int,
    progress: Optional[Progress] = None,
    description: str = 'Monte-Carlo'
) -> Generator[Dict[str, sc.DataArray], None, None]:
    """Draw n samples.

    Passes the RNG to samplers in the following order:

    .. code:: python

        for _ in range(n):
            for sampler in samplers.values():
                sampler.sample_once(rng)

    Parameters
    ----------
    samplers:
        dict of samplers to draw from.
    rng:
        Random number generator to pass to the samplers.
    n:
        Number of samples to draw from each sampler.
    progress:
        Progress bar. Disabled if set to ``None``.
    description:
        Message to show on the progress bar.

    Yields
    ------
    :
        dicts of samples indexed by the keys of ``samplers``.
    """
    if progress is None:
        progress = SilentProgress()
    yield from progress.track(islice(resample(samplers=samplers, rng=rng), n),
                              total=n,
                              description=description)
