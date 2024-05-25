"""pool.py: Provides classes and methods for a smart process pool."""

import multiprocessing as mp
import os
import signal
from typing import Any, Callable, Iterable, Optional, TypeVar, cast

_T = TypeVar("_T")
_RT = TypeVar("_RT")


# NOTE: Throughout this file there are several `type: ignore` comments due to
# the multiprocessing library being known to not have very many, or very good,
# type annotations/stubs.


def set_signal_handler() -> None:
    """Set signal handler to ignore SIGINT."""
    # ignore sigint signal use pool stop instead
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class SmartPool:
    """A process pool that creates a process per CPU core."""

    def __init__(
        self,
        min_empty_cores: int = 1,
        max_used_cores: Optional[int] = None,
        warm_start: bool = True,
    ) -> None:
        """Initialize.

        Args:
            min_empty_cores: Min number of cores to leave empty
            max_used_cores: Max number of cores for the pool to use

        Returns:
            None
        """
        # print("Starting smart pool...")
        num_cores = cast(int, os.cpu_count())
        # print("Core count: ", num_cores)

        if max_used_cores is not None:
            num_processes = max(min(num_cores - min_empty_cores, max_used_cores), 1)
        else:
            num_processes = max(num_cores - min_empty_cores, 1)
        # print("Process count: ", num_processes)

        self._pool = mp.Pool(  # pylint: disable=consider-using-with
            processes=num_processes, initializer=set_signal_handler
        )
        if warm_start:
            self._pool.map(_warm_start, range(0, 100), chunksize=1)

    def map(self, func: Callable[[_T], _RT], iterable: Iterable[Any]) -> Iterable[_RT]:
        """Use the smart pool to run a function on every element in iterable.

        Args:
            func: A function that maps elements
            iterable: An iterable of elements

        Returns:
            An iterable of the results of each function call
        """
        return self._pool.map(func, iterable)

    def imap(
        self, func: Callable[[_T], _RT], iterable: Iterable[Any], chunk_size: int = 10
    ) -> Iterable[_RT]:
        """Use the smart pool to run a function on every element in iterable.

        Args:
            func: A function that maps elements
            iterable: An iterable of elements
            chunk_size: Size of chunks given to each process

        Returns:
            An iterable of the results of each function call
        """
        return self._pool.imap(func, iterable, chunksize=chunk_size)

    def imap_unordered(
        self, func: Callable[[_T], _RT], iterable: Iterable[Any], chunk_size: int = 10
    ) -> Iterable[_RT]:
        """Use the smart pool to run a function on every element in iterable.

        Results are unordered.

        Args:
            func: A function that maps elements
            iterable: An iterable of elements
            chunk_size: Size of chunks given to each process

        Returns:
            An iterable of the results of each function call
        """
        return self._pool.imap_unordered(func, iterable, chunksize=chunk_size)

    def stop(self) -> None:
        """Stop the smart pool child processes."""
        self._pool.close()
        self._pool.join()


def _warm_start(value: int) -> int:
    return value + value
