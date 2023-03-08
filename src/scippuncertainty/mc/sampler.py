# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Data (re-)sampling."""
from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
import scipp as sc

S = TypeVar("S", bound="Sampler")


class Sampler(Protocol):
    """Protocol for MC data re-samplers."""

    def sample_once(self, rng: np.random.Generator) -> sc.DataArray:
        """Return a single new sample.

        Parameters
        ----------
        rng:
            Use this as the only source of randomness to ensure reproducibility.

        Returns
        -------
        :
            A new sample.
        """

    def clone(self: S) -> S:
        """Return a new sampler of the same type.

        The returned sampler is independent of ``self`` and samples from the same data.
        """


class PoissonDenseSampler:
    """Resample dense data from a Poisson distribution.

    Generates data arrays with the same metadata as the input and ``values``
    drawn from a Poisson distribution with ``mean = input.values``.

    The input data must be dense, i.e. not binned.
    """

    def __init__(
        self, data: sc.DataArray, copy: bool = True, copy_in: bool = True
    ) -> None:
        """Initialize a PoissonDenseSampler.

        Parameters
        ----------
        data:
            Input data to sample from.
        copy:
            If ``True``, :meth:`PoissonDenseSampler.sample_once` returns a new array.
            If ``False``, it returns a reference to the same array every time but
            updates the values in-place.
            Use with caution!
        copy_in:
            If ``True``, ``data`` is copied during initialization.
            If ``False``, the sampler keeps a reference to the input object.
        """
        self._base_hist = data.copy() if copy_in else data
        self._result_buffer = sc.empty_like(data)
        if data.dtype in ("float64", "float32"):
            self._result_buffer.variances = None
        self._copy = copy

    def sample_once(self, rng: np.random.Generator) -> sc.DataArray:
        """Return a new sample."""
        self._result_buffer.values = rng.poisson(self._base_hist.values)
        return self._result_buffer.copy() if self._copy else self._result_buffer

    def clone(self) -> PoissonDenseSampler:
        """Return a new PoissonDenseSampler.

        The returned sampler is independent of ``self`` and samples from the same data.
        """
        return PoissonDenseSampler(self._base_hist, copy_in=True, copy=self._copy)


class NormalDenseSampler:
    """Resample dense data from a Normal distribution.

    Generates data arrays with the same metadata as the input and ``values``
    drawn from a normal distribution with ``mean = input.values`` and
    ``standard_deviation = sqrt(input.variances)``.

    The input data must be dense, i.e. not binned.
    """

    def __init__(
        self, data: sc.DataArray, copy: bool = True, copy_in: bool = True
    ) -> None:
        """Initialize a NormalDenseSampler.

        Parameters
        ----------
        data:
            Input data to sample from.
        copy:
            If ``True``, :meth:`NormalDenseSampler.sample_once` returns a new array.
            If ``False``, it returns a reference to the same array every time but
            updates the values in-place.
            Use with caution!
        copy_in:
            If ``True``, ``data`` is copied during initialization.
            If ``False``, the sampler keeps a reference to the input object.
        """
        self._base_hist = data.copy() if copy_in else data
        self._stds = sc.stddevs(self._base_hist).values
        self._result_buffer = sc.empty_like(data)
        self._result_buffer.variances = None
        self._copy = copy

    def sample_once(self, rng: np.random.Generator) -> sc.DataArray:
        """Return a new sample."""
        self._result_buffer.values = rng.normal(
            loc=self._base_hist.values, scale=self._stds
        )
        return self._result_buffer.copy() if self._copy else self._result_buffer

    def clone(self) -> NormalDenseSampler:
        """Return a new NormalDenseSampler.

        The returned sampler is independent of ``self`` and samples from the same data.
        """
        return NormalDenseSampler(self._base_hist, copy_in=True, copy=self._copy)
