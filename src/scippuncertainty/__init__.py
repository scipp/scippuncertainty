# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Advanced uncertainty propagation with Scipp."""
# ruff: noqa: E402, F401

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .correlation import pearson_correlation

__all__ = ["pearson_correlation"]
