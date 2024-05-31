"""Implement a worker that class methods in the child process."""

import multiprocessing.connection as mp_conn
import traceback
import typing


STOP = "__STOP__"


Command = typing.Tuple[
    typing.Callable[..., typing.Any],
    typing.Tuple[typing.Any, ...],
    typing.Dict[str, typing.Any],
]


def class_worker(
    conn: mp_conn.Connection,
    cls: typing.Type[typing.Any],
    cls_args: typing.Tuple[typing.Any],
    cls_kwargs: typing.Dict[str, typing.Any],
) -> None:
    cls_instance = cls(*cls_args, **cls_kwargs)
    for val in iter(conn.recv, STOP):
        try:
            conn.send(val[0](cls_instance, *val[1], **val[2]))
        except:  # noqa: E722
            traceback.print_exc()
    conn.send(STOP)
    conn.close()
