# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Data (re-)sampling."""

from typing import Protocol

import numpy as np
import scipp as sc


class Sampler(Protocol):
    """Resample a data array."""

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


class PoissonDenseSampler:
    """Resample dense data from a Poisson distribution.

    Assumes and does not check that the input data is poisson distributed.
    The uncertainties of the input are ignored.

    The data must be dense, i.e. not binned.
    """

    def __init__(self,
                 data: sc.DataArray,
                 copy: bool = True,
                 copy_in: bool = True) -> None:
        self._base_hist = data.copy() if copy_in else data
        self._result_buffer = sc.empty_like(data)
        if data.dtype in ('float64', 'float32'):
            self._result_buffer.variances = None
        self._copy = copy

    def sample_once(self, rng: np.random.Generator) -> sc.DataArray:
        """Return a new sample."""
        self._result_buffer.values = rng.poisson(self._base_hist.values)
        return self._result_buffer.copy(
        ) if self._copy else self._result_buffer


class NormalDenseSampler:
    """Resample dense data from a Normal distribution.

    Uses the ``values`` of the input as means and ``sqrt(variances)`` as
    standard deviations.

    The data must be dense, i.e. not binned.
    """

    def __init__(self,
                 data: sc.DataArray,
                 copy: bool = True,
                 copy_in: bool = True) -> None:
        self._base_hist = data.copy() if copy_in else data
        self._stds = sc.stddevs(self._base_hist).values
        self._result_buffer = sc.empty_like(data)
        self._result_buffer.variances = None
        self._copy = copy

    def sample_once(self, rng: np.random.Generator) -> sc.DataArray:
        """Return a new sample."""
        self._result_buffer.values = rng.normal(loc=self._base_hist.values,
                                                scale=self._stds)
        return self._result_buffer.copy(
        ) if self._copy else self._result_buffer
