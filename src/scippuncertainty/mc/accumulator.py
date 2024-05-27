# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Compute desired statistics on processed data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

import scipp as sc

A = TypeVar("A", bound="Accumulator")


@dataclass(frozen=True)
class Accumulated:
    """Accumulated data and metadata."""

    data: sc.DataArray
    n_samples: int
    samples: tuple[sc.DataArray, ...] | None = None


class Accumulator(Protocol):
    """Compute statistics on bootstrap results."""

    def add(self, sample: sc.DataArray) -> None:
        """Register a single sample."""

    def add_from(self: A, other: A) -> None:
        """Merge results from ``other`` into ``self``."""

    def get(self) -> Accumulated:
        """Return the current result."""

    def new(self: A) -> A:
        """Return a new accumulator of the same type as ``self``.

        The new instance does not contain any samples even if ``self`` does.
        """


class VarianceAccum:
    r"""Compute the mean and variance of bootstrap samples.

    The mean :math:`\mu_i` and variance :math:`\sigma_i^2` for array element
    :math:`i` are defined as

    .. math::

        \mu_i &= \frac{1}{N} \sum_{s=1}^{N} x_{i s} \\
        \sigma_i^2 &= \frac{1}{N-1} \sum_{s=1}^{N} (x_{i s} - \mu_i)^2, \\

    where the sums run over the Monte-Carlo samples.

    The computation of variances uses an algorithm based on :cite:`welford:1962`
    and :cite:`chan:1982`. This reduces the risk of catastrophic cancellations
    from sums of squares compared to a naive implementation.
    """

    # See also the article on Wikipedia for an overview:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # This code uses what they call 'Welford's algorithm' and the
    # 'parallel algorithm'.

    def __init__(self, *, keep_samples: bool = False) -> None:
        """Initialize a CovarianceAccum instance.

        Parameters
        ----------
        keep_samples:
            If ``True``, all samples are kept and returned as an attribute called
            ``samples`` with dimension ``monte_carlo``.
        """
        self._mean: sc.DataArray | None = None
        self._m2_dist: sc.Variable | None = None
        self._n_samples = 0
        self._samples: list[sc.DataArray] | None = [] if keep_samples else None

    def add(self, sample: sc.DataArray) -> None:
        """Register a single sample."""
        self._n_samples += 1
        if self._mean is None:
            self._mean = sample.copy()
            self._m2_dist = sc.zeros(sizes=sample.sizes, unit=sample.unit**2)
        else:
            delta = sample - self._mean
            self._mean += delta / self._n_samples
            delta2 = sample - self._mean
            self._m2_dist += delta * delta2
        if self._samples is not None:
            self._samples.append(sample.copy())

    def add_from(self, other: VarianceAccum) -> None:
        """Merge results from ``other`` into ``self``."""
        if self._mean is None:
            self._mean = other._mean.copy()
            self._m2_dist = other._m2_dist.copy()
        else:
            delta = other._mean - self._mean
            self._m2_dist = (
                self._m2_dist
                + other._m2_dist
                + delta**2
                * self._n_samples
                * other._n_samples
                / (self._n_samples + other._n_samples)
            )
            self._mean += (
                delta * other._n_samples / (self._n_samples + other._n_samples)
            )
        self._n_samples += other._n_samples

        if (self._samples is None) ^ (other._samples is None):
            raise RuntimeError("Both accumulators must have values")
        if self._samples is not None:
            self._samples.extend(other._samples)

    def get(self) -> Accumulated:
        """Return the current result."""
        if self._n_samples == 0:
            raise RuntimeError("There are no results to get.")
        mean = self._mean
        var = self._m2_dist / (self._n_samples - 1)

        res = mean
        res.variances = var.values

        return Accumulated(
            data=res,
            n_samples=self._n_samples,
            samples=tuple(self._samples) if self._samples is not None else None,
        )

    def new(self) -> VarianceAccum:
        """Return a new VarianceAccum.

        The new instance does not contain any samples even if ``self`` does.
        """
        return VarianceAccum(keep_samples=self._samples is not None)


class CovarianceAccum:
    r"""Compute the covariance matrix of a 1d array with itself.

    The covariance :math:`\Sigma_{ij}^2` of array element :math:`i`
    with element :math:`j` is defined as

    .. math::

        \Sigma_{ij}^2 = \frac{1}{N-1} \sum_{s=1}^N (x_{i s} - \mu_i)
          (x_{j s} - \mu_j)

    where the sums run over the Monte-Carlo samples and :math:`\mu_i` is the
    mean of element :math:`i`.

    The computation uses an algorithm based on :cite:`schubert:2018` which
    reduces the risk of catastrophic cancellations from sums of squares
    compared to a naive implementation.

    The covariance matrix is encoded as the values of a 2d data array.
    Dimensions are named ``dim_0`` and ``dim_1`` by default, but this can be
    customized using the ``dims`` argument.
    The returned data array does not have any coordinates, attributes, or masks
    (except for an attribute ``n-samples``).
    """

    # See also the article on Wikipedia for an overview:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance
    # In particular the 'Online' section.

    def __init__(self, *, dims: list[str] | tuple[str, str] | None = None) -> None:
        """Initialize a CovarianceAccum instance.

        Parameters
        ----------
        dims:
            Dimension names for the covariance matrix.
            Must be length-2.
        """
        self._mean = None
        self._c = None
        self._n_samples = 0
        self._dims = ("dim_0", "dim_1") if dims is None else tuple(dims)

    def add(self, sample: sc.DataArray) -> None:
        """Register a single sample."""
        if sample.ndim != 1:
            raise sc.DimensionError("Can only handle 1-d values")
        sample = sample.rename({sample.dims[0]: self._dims[0]})
        for key in list(sample.coords):
            del sample.coords[key]
        for key in list(sample.masks):
            del sample.masks[key]

        self._n_samples += 1
        if self._mean is None:
            self._mean = sample.copy()
            self._c = sc.zeros(
                sizes={self._dims[0]: sample.shape[0], self._dims[1]: sample.shape[0]},
                unit=sample.unit**2,
            )
        else:
            delta_old = sample - self._mean
            self._mean += delta_old / self._n_samples
            delta_new = (sample - self._mean).rename({self._dims[0]: self._dims[1]})
            self._c += delta_old * delta_new

    def add_from(self, other: CovarianceAccum) -> None:
        """Merge results from ``other`` into ``self``."""
        if self._mean is None:
            self._mean = other._mean.copy()
            self._c = other._c.copy()
        else:
            n = self._n_samples + other._n_samples
            self._c += other._c + (self._n_samples * other._n_samples / n) * (
                self._mean - other._mean
            ) * (self._mean - other._mean).rename({self._dims[0]: self._dims[1]})
            self._mean = self._mean * (self._n_samples / n) + other._mean * (
                other._n_samples / n
            )
        self._n_samples += other._n_samples

    def get(self) -> Accumulated:
        """Return the current result."""
        if self._n_samples == 0:
            raise RuntimeError("There are no results to get.")
        cov = self._c / (self._n_samples - 1)
        if isinstance(cov, sc.Variable):
            # This happens when there is only 1 sample.
            cov = sc.DataArray(
                cov,
                coords=self._mean.coords,
                masks=self._mean.masks,
            )
        return Accumulated(
            data=cov,
            n_samples=self._n_samples,
        )

    def new(self) -> CovarianceAccum:
        """Return a new CovarianceAccum.

        The new instance does not contain any samples even if ``self`` does.
        """
        return CovarianceAccum(dims=self._dims)
