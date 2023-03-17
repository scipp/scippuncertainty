# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Computing correlations."""

from typing import TypeVar, Union

import numpy as np
import scipp as sc

T = TypeVar("T", bound=Union[sc.Variable, sc.DataArray])


def pearson_correlation(cov: T) -> T:
    r"""Compute the Pearson correlation coefficient from a covariance matrix.

    The Pearson correlation coefficient is a measure of linear correlation.
    Given a variance-covariance matrix :math:`\Sigma`, it is ddefined as:

    .. math::

        r_{ij} = \Sigma_{ij} / \sqrt{\Sigma_{ii} \Sigma_{jj}}

    It ranges from 0 for no correlation to 1 for full correlation.
    The diagonal is always 1.

    Parameters
    ----------
    cov:
        A variance-covariance matrix.
        Must have exactly 2 dimensions that correspond to rows and columns.

    Returns
    -------
    :
        The pearson correlation coefficient.
        Has the same sizes and coords as ``cov``.
    """
    if cov.ndim != 2:
        # Manual check because otherwise this results in ValueError.
        raise sc.DimensionError("cov must be 2-dimensional")

    var = sc.array(dims=[cov.dims[0]], values=np.diag(cov.values), unit=cov.unit)
    std0 = sc.sqrt(var)
    std1 = std0.rename({cov.dims[0]: cov.dims[1]})
    return cov / (std0 * std1)
