# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Progress bars."""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import (
    Any,
    Protocol,
    TypeVar,
)

ProgressType = TypeVar("ProgressType")


class Progress(Protocol):
    """Progress bar to track iteration over a sequence.

    Only works if the display has been initialized with ``MultiProgress.prepare``.
    """

    def track(
        self,
        sequence: Iterable[ProgressType] | Sequence[ProgressType],
        description: str,
        total: float | None = None,
    ) -> Iterable[ProgressType]:
        """Display and update a progress bar based on the provided sequence.

        Parameters
        ----------
        sequence:
            Iterable to iterate over and track progress.
        description:
            Message to display with the progress bar.
        total:
            Number of items to iterate over.

        Returns
        -------
        :
            An iterable over the same elements as ``sequence``.
        """


class SilentProgress:
    """'Progress bar' that shows nothing."""

    @staticmethod
    def track(
        sequence: Iterable[ProgressType] | Sequence[ProgressType],
        description: str,
        total: float | None = None,
    ) -> Iterable[ProgressType]:
        """Return the input sequence.

        Parameters
        ----------
        sequence:
            Iterable to iterate over and track progress.
        description:
            *Ignored*.
        total:
            *Ignored*.

        Returns
        -------
        :
            Same as ``sequence``.
        """
        return sequence


class RichProgress:
    """Progress bar using ``rich``."""

    def __init__(self) -> None:
        import rich.progress
        from rich.progress import (
            BarColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._progress = rich.progress.Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            refresh_per_second=1,
        )

    def track(
        self,
        sequence: Iterable[ProgressType] | Sequence[ProgressType],
        description: str,
        total: float | None = None,
    ) -> Iterable[ProgressType]:
        """Display and update a progress bar based on the provided sequence.

        Parameters
        ----------
        sequence:
            Iterable to iterate over and track progress.
        description:
            Message to display with the progress bar.
        total:
            Number of items to iterate over.

        Returns
        -------
        :
            An iterable over the same elements as ``sequence``.
        """
        return self._progress.track(
            sequence, total=total, description=description, update_period=1
        )

    def underlying(self) -> Any:
        """Return the underlying rich progress bar."""
        return self._progress


class MultiProgress(Protocol):
    """Manager for multiple progress bars."""

    @contextmanager
    def prepare(self, n: int) -> Iterator[list[Progress]]:
        """Create individual progress bars and set up display.

        The bars may only be used while this contextmanager is active.
        """


class SilentMultiProgress:
    """Manager for multiple silent progress bars."""

    @contextmanager
    def prepare(self, n: int) -> Iterator[list[Progress]]:
        """Create individual silent progress bars and set up display."""
        yield [SilentProgress() for _ in range(n)]


class RichMultiProgress:
    """Manager for multiple rich progress bars."""

    @contextmanager
    def prepare(self, n: int) -> Iterator[list[Progress]]:
        """Create individual rich progress bars and set up display.

        The bars may only be used while this contextmanager is active.
        """
        from rich.console import Group
        from rich.live import Live

        progresses = [RichProgress() for _ in range(n)]
        with Live(Group(*(p.underlying() for p in progresses))):
            yield progresses


def _rich_is_installed() -> bool:
    try:
        import rich  # noqa: F401
    except ImportError:
        return False
    return True


def progress_bars(visible: bool | None) -> MultiProgress:
    """Construct a multi progress bar.

    Parameters
    ----------
    visible:
        - If ``None``, pick an implementation automatically.
        - If ``True``, use ``RichMultiProgress``,
          `rich <https://rich.readthedocs.io/en/stable/index.html>`_ must be installed
          in this case.
        - If ``False``, use ``SilentMultiProgress``.

    Returns
    -------
    :
        A new multi progress bar.

    Raises
    ------
    ImportError
        If ``rich`` is not installed and ``visible is True``.
    """
    if visible is None:
        return RichMultiProgress() if _rich_is_installed() else SilentMultiProgress()
    if visible:
        if not _rich_is_installed():
            raise ImportError(
                "Cannot show a progress bar because `rich` is not "
                "available. Either switch off progress bars or install "
                "rich with `pip install rich` or "
                "`conda install -c conda-forge rich`."
            )
        return RichMultiProgress()
    return SilentMultiProgress()
