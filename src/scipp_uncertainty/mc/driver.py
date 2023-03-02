# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Control bootstrap resampling."""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from itertools import islice, starmap
from typing import Callable, Dict, Generator, List, Optional, Union

import numpy as np
import scipp as sc

from .._progress import Progress, SilentProgress, progress_bars
from .._util import distribute_evenly
from ..random import make_rngs
from .accumulator import Accumulator
from .sampler import Sampler


def _n_samples_per_thread(n_samples: int, n_thread: int) -> List[int]:
    return distribute_evenly(n_samples, n_thread)


def _pick_thread_count(n_threads: Optional[int]) -> int:
    if n_threads is None:
        return max(multiprocessing.cpu_count(), 4)
    return max(n_threads, 1)


def _resample_once(
    samplers: Dict[str, Sampler], rng: np.random.Generator
) -> Dict[str, sc.DataArray]:
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
    description: str = "Monte-Carlo",
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
    yield from progress.track(
        islice(resample(samplers=samplers, rng=rng), n),
        total=n,
        description=description,
    )


def run(
    fn: Callable[..., Union[Dict[str, sc.DataArray], SkipSampleType]],
    *,
    n_samples: int,
    samplers: Dict[str, Sampler],
    accumulators: Optional[Dict[str, Accumulator]] = None,
    n_threads: Optional[int] = None,
    seed: Optional[
        Union[
            np.random.Generator,
            List[np.random.Generator],
            int,
            List[int],
            np.random.SeedSequence,
        ]
    ] = None,
    progress: Optional[bool] = None,
    description: str = "Monte-Carlo",
) -> Dict[str, sc.DataArray]:
    """TODO."""
    rngs = make_rngs(seed=seed, n=_pick_thread_count(n_threads))
    results = []
    with progress_bars(visible=progress).prepare(len(rngs)) as p_bars:
        with ThreadPoolExecutor(max_workers=len(rngs)) as executor:
            for i, (rng, n_thread_samples, progress_bar) in enumerate(
                zip(rngs, _n_samples_per_thread(n_samples, len(rngs)), p_bars)
            ):
                job = _Job(
                    id_=i,
                    base_samplers=samplers,
                    base_accumulators=accumulators,
                    rng=rng,
                    progress_bar=progress_bar,
                    description=description,
                )
                results.append(executor.submit(job, fn, n_samples=n_thread_samples))

    accumulators = results[0].result()
    for res in results[1:]:
        for a, b in zip(accumulators.values(), res.result().values()):
            a.add_from(b)
    return {name: accum.get() for name, accum in accumulators.items()}


# TODO
# - clone samplers per thread to make sure buffers don't clash
# - pass accumulator instances and create new instances for each thread


class _Job:
    def __init__(
        self,
        *,
        id_: int,
        base_samplers: Dict[str, Sampler],
        base_accumulators: Dict[str, Accumulator],
        rng: np.random.Generator,
        progress_bar: Progress,
        description: str,
    ) -> None:
        self._id = id_
        self._samplers = {
            key: sampler.clone() for key, sampler in base_samplers.items()
        }
        self._accumulators = {
            key: accum.new() for key, accum in base_accumulators.items()
        }
        self._rng = rng
        self._progress_bar = progress_bar
        self._description = description + f"[{self._id}]"

    def __call__(
        self,
        fn: Callable[..., Union[Dict[str, sc.DataArray], SkipSampleType]],
        *,
        n_samples: int,
    ) -> Dict[str, Accumulator]:
        for samples in starmap(
            fn,
            resample_n(
                samplers=self._samplers,
                rng=self._rng,
                n=n_samples,
                progress=self._progress_bar,
                description=self._description,
            ),
        ):
            if samples is SkipSample:
                continue
            for n, r in samples.items():
                self._accumulators[n].add(r)
        return self._accumulators


class SkipSampleType:
    """See :attr:`scipp_uncertainty.mc.SkipSample`."""

    def __repr__(self) -> str:
        """repr."""
        return "SkipSample"


SkipSample = SkipSampleType()
"""Return from a function to indicate that a sample should be skipped."""
