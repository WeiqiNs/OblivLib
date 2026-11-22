"""The client/server wire protocol: every message the client and ``StorageServer`` exchange.

A message is a frozen ``msgspec.Struct`` of primitives (ints, str, bytes, None, lists, and nested ops), sent
as the MessagePack array ``[ClassName, *fields in declaration order]``. Decoding checks every message
against its declared fields, so a malformed message raises ``ProtocolError``.
"""

import msgspec

from oblivlib.dependency.errors import (
    DuplicateLabelError,
    OblivlibError,
    ProtocolError,
    RowSizeError,
    ServerError,
    StorageError,
    UnknownLabelError,
)


class _Message(msgspec.Struct, frozen=True, tag=True, array_like=True):
    pass


PROTOCOL_VERSION = 2


class ReadPath(_Message, frozen=True):
    label: str
    leaves: list[int]


class WritePath(_Message, frozen=True):
    label: str
    leaves: list[int]
    rows: list[bytes]


class Hello(_Message, frozen=True):
    version: int


class CreateTree(_Message, frozen=True):
    session: str
    label: str
    level: int
    row_bytes: int | None


class AttachTree(_Message, frozen=True):
    session: str
    label: str
    filename: str
    level: int
    row_bytes: int


class WriteRange(_Message, frozen=True):
    session: str
    label: str
    start: int
    rows: list[bytes]


class Batch(_Message, frozen=True):
    session: str
    writes: list[WritePath]
    reads: list[ReadPath]


class Close(_Message, frozen=True):
    session: str


class Shutdown(_Message, frozen=True):
    pass


class HelloReply(_Message, frozen=True):
    session: str


class Ok(_Message, frozen=True):
    pass


class BatchReply(_Message, frozen=True):
    results: list[list[bytes]]


class ErrorReply(_Message, frozen=True):
    kind: str
    message: str


SessionRequest = CreateTree | AttachTree | WriteRange | Batch | Close
Request = Hello | SessionRequest | Shutdown
Reply = HelloReply | Ok | BatchReply | ErrorReply


class Timed(_Message, frozen=True):
    server_ns: int
    reply: Reply


ERROR_KINDS: dict[str, type[OblivlibError]] = {
    cls.__name__: cls
    for cls in (UnknownLabelError, DuplicateLabelError, RowSizeError, ProtocolError, StorageError, ServerError)
}

_ENCODER = msgspec.msgpack.Encoder()
_REQUESTS: msgspec.msgpack.Decoder[Request] = msgspec.msgpack.Decoder(Request)
_TIMED: msgspec.msgpack.Decoder[Timed] = msgspec.msgpack.Decoder(Timed)


def encode(message: Request | Timed) -> bytes:
    return _ENCODER.encode(message)


def decode_request(data: bytes) -> Request:
    try:
        return _REQUESTS.decode(data)
    except msgspec.DecodeError as exc:
        raise ProtocolError(f"malformed request: {exc}") from exc


def decode_timed(data: bytes) -> Timed:
    try:
        return _TIMED.decode(data)
    except msgspec.DecodeError as exc:
        raise ProtocolError(f"malformed reply: {exc}") from exc


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
