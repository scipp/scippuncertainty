# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest

from scippuncertainty._util import distribute_evenly


@pytest.mark.parametrize("total", range(0, 30))
@pytest.mark.parametrize("n_parts", range(1, 10))
def test_distribute_evenly(total: int, n_parts: int):
    parts = distribute_evenly(total, n_parts)
    assert sum(parts) == total
    assert max(parts) in (min(parts), min(parts) + 1)
