"""pool.py: Provides classes and methods for a smart process pool."""

import logging
import multiprocessing as mp
import os
import signal
import typing


logger = logging.getLogger(__name__)


_T = typing.TypeVar("_T")
_RT = typing.TypeVar("_RT")


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
        max_processes: typing.Optional[int] = None,
        warm_start: bool = True,
    ) -> None:
        """Initialize.

        Args:
            min_empty_cores: Min number of cores to leave empty
            max_processes: Max number of processes for the pool to use

        Returns:
            None
        """
        logger.debug("Starting smart pool...")
        num_cores = os.cpu_count()
        if num_cores is None:
            num_cores = 1
        logger.debug("Core count: %d", num_cores)

        num_processes = 2 * num_cores

        if max_processes is not None:
            num_processes = min(num_processes, max_processes)
        logger.debug("Process count: %d", num_processes)

        self._pool = mp.Pool(  # pylint: disable=consider-using-with
            processes=num_processes, initializer=set_signal_handler
        )
        if warm_start:
            self._pool.map(_warm_start, range(0, 100), chunksize=1)

    def map(
        self, func: typing.Callable[[_T], _RT], iterable: typing.Iterable[typing.Any]
    ) -> typing.Iterable[_RT]:
        """Use the smart pool to run a function on every element in iterable.

        Args:
            func: A function that maps elements
            iterable: An iterable of elements

        Returns:
            An iterable of the results of each function call
        """
        return self._pool.map(func, iterable)

    def imap(
        self,
        func: typing.Callable[[_T], _RT],
        iterable: typing.Iterable[typing.Any],
        chunk_size: int = 10,
    ) -> typing.Iterable[_RT]:
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
        self,
        func: typing.Callable[[_T], _RT],
        iterable: typing.Iterable[typing.Any],
        chunk_size: int = 10,
    ) -> typing.Iterable[_RT]:
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
