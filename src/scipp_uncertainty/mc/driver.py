# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Control bootstrap resampling."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import numpy as np
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..logging import get_logger
from .accumulator import VarianceAccum


def _n_samples_per_thread(n_samples, n_thread):
    base = [n_samples // n_thread] * n_thread
    for i in range(n_samples % n_thread):
        base[i] += 1
    if sum(base) != n_samples:
        # This should not happen, only an internal debug check
        raise RuntimeError('Cannot distribute samples over threads')
    return base


def _make_progress_bars(n):
    return [
        Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
        ) for _ in range(n)
    ]


class Bootstrap:
    """Perform a bootstrap analysis."""

    def __init__(self,
                 samplers: dict,
                 *,
                 seed: Optional[Union[int, np.random.SeedSequence]] = None,
                 n_threads: Optional[int] = None,
                 keep_samples=None,
                 accumulators: Optional[dict] = None):
        if n_threads is None:
            n_threads = multiprocessing.cpu_count() // 4
        if seed is None:
            seed = np.random.SeedSequence()
        elif isinstance(seed, int):
            seed = np.random.SeedSequence(seed)
        get_logger().info('Seeding bootstrap RNG with %s', seed.entropy)
        self._rngs = [np.random.default_rng(s) for s in seed.spawn(n_threads)]

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
        progress_bars = _make_progress_bars(len(self._rngs))
        with Live(Group(*progress_bars)):
            with ThreadPoolExecutor(max_workers=len(self._rngs)) as executor:
                for rng, n_thread_samples, progress_bar in zip(
                        self._rngs,
                        _n_samples_per_thread(n_samples, len(self._rngs)),
                        progress_bars):
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
