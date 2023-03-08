# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Utilities for logging in ScippUncertainty."""

import logging


def logger_name() -> str:
    """Return the name of ScippUncertainty's logger."""
    return "scipp.uncertainty"


def get_logger() -> logging.Logger:
    """Return the logger used by ScippUncertainty."""
    return logging.getLogger(logger_name())
