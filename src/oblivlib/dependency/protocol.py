"""The client/server wire protocol: every message the client and ``StorageServer`` exchange.

A message is a frozen dataclass of primitives (ints, str, bytes, None, lists, and nested ops), framed as
the msgpack array ``[ClassName, *fields in declaration order]``. Both ends are this library, so decoding
only rebuilds the dataclasses; an unknown tag or a wrong field count raises.
"""

from dataclasses import dataclass, fields
from typing import Any, get_args

import msgpack

from oblivlib.dependency.errors import (
    DuplicateLabelError,
    OblivlibError,
    ProtocolError,
    RowSizeError,
    ScaleDownError,
    ServerError,
    StorageError,
    UnknownLabelError,
)
from oblivlib.dependency.types import ListPopBack, ListPushFront, ListWrite

PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReadPath:
    label: str
    leaves: list[int]


@dataclass(frozen=True, slots=True)
class WritePath:
    label: str
    leaves: list[int]
    rows: list[bytes]


@dataclass(frozen=True, slots=True)
class ReadList:
    label: str
    indices: list[int] | None


@dataclass(frozen=True, slots=True)
class ListOps:
    label: str
    ops: list[ListWrite | ListPushFront | ListPopBack]


@dataclass(frozen=True, slots=True)
class Hello:
    version: int


@dataclass(frozen=True, slots=True)
class CreateTree:
    session: str
    label: str
    level: int
    row_bytes: int | None


@dataclass(frozen=True, slots=True)
class CreateList:
    session: str
    label: str


@dataclass(frozen=True, slots=True)
class AttachTree:
    session: str
    label: str
    filename: str
    level: int
    row_bytes: int


@dataclass(frozen=True, slots=True)
class Resize:
    session: str
    label: str
    level: int


@dataclass(frozen=True, slots=True)
class WriteRange:
    session: str
    label: str
    start: int
    rows: list[bytes]


@dataclass(frozen=True, slots=True)
class Batch:
    session: str
    writes: list[WritePath | ListOps]
    reads: list[ReadPath | ReadList]


@dataclass(frozen=True, slots=True)
class Close:
    session: str


@dataclass(frozen=True, slots=True)
class Shutdown:
    pass


@dataclass(frozen=True, slots=True)
class HelloReply:
    session: str


@dataclass(frozen=True, slots=True)
class Ok:
    pass


@dataclass(frozen=True, slots=True)
class BatchReply:
    results: list[list[bytes]]


@dataclass(frozen=True, slots=True)
class ErrorReply:
    kind: str
    message: str


WriteOp = WritePath | ListOps
ReadOp = ReadPath | ReadList
Request = Hello | CreateTree | CreateList | AttachTree | Resize | WriteRange | Batch | Close | Shutdown
Reply = HelloReply | Ok | BatchReply | ErrorReply


def _tag_table[T](classes: tuple[type[T], ...]) -> dict[str, type[T]]:
    table: dict[str, type[T]] = {}
    for cls in classes:
        if cls.__name__ in table:
            raise ValueError(f"protocol: duplicate tag {cls.__name__!r}.")
        table[cls.__name__] = cls
    return table


ERROR_KINDS: dict[str, type[OblivlibError]] = _tag_table(
    (UnknownLabelError, DuplicateLabelError, RowSizeError, ProtocolError, ScaleDownError, StorageError, ServerError)
)


_MESSAGES: dict[str, type] = _tag_table(
    (
        ListWrite,
        ListPushFront,
        ListPopBack,
        ReadPath,
        WritePath,
        ReadList,
        ListOps,
        *get_args(Request),
        *get_args(Reply),
    )
)
_FIELD_NAMES: dict[type, tuple[str, ...]] = {cls: tuple(f.name for f in fields(cls)) for cls in _MESSAGES.values()}


def _to_wire(value: Any) -> Any:
    if type(value) is list:
        return [_to_wire(item) for item in value]
    names = _FIELD_NAMES.get(type(value))
    if names is None:
        return value
    return [type(value).__name__, *(_to_wire(getattr(value, name)) for name in names)]


def _from_wire(value: Any) -> Any:
    if type(value) is not list:
        return value
    if value and type(value[0]) is str:
        return _MESSAGES[value[0]](*(_from_wire(item) for item in value[1:]))
    return [_from_wire(item) for item in value]


def encode(message: Request | Reply) -> bytes:
    return msgpack.packb(_to_wire(message))


def decode_request(data: bytes) -> Request:
    message = _from_wire(msgpack.unpackb(data))
    if not isinstance(message, get_args(Request)):
        raise ProtocolError(f"expected a request, got {type(message).__name__}.")
    return message


def decode_reply(data: bytes) -> Reply:
    return _from_wire(msgpack.unpackb(data))


def error_reply(exc: Exception) -> ErrorReply:
    for cls in type(exc).__mro__:
        if ERROR_KINDS.get(cls.__name__) is cls:
            return ErrorReply(kind=cls.__name__, message=str(exc))
    if isinstance(exc, OSError):
        return ErrorReply(kind=StorageError.__name__, message=str(exc))
    return ErrorReply(kind=ServerError.__name__, message=f"{type(exc).__name__}: {exc}")


def expect_reply[R](reply: Reply, kind: type[R]) -> R:
    if isinstance(reply, ErrorReply):
        error = ERROR_KINDS.get(reply.kind)
        if error is None:
            raise ProtocolError(f"unknown error kind {reply.kind!r}: {reply.message}")
        raise error(reply.message)
    if not isinstance(reply, kind):
        raise ProtocolError(f"expected {kind.__name__}, got {type(reply).__name__}.")
    return reply
