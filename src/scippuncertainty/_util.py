# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Miscellaneous utilities."""


def distribute_evenly(total: int, n_parts: int) -> list[int]:
    """Divide total into n parts, spreading the remainder evenly."""
    parts = [total // n_parts] * n_parts
    for i in range(total % n_parts):
        parts[i] += 1
    return parts
