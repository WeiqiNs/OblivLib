import pytest

from oblivlib.dependency import (
    DuplicateLabelError,
    ProtocolError,
    ServerError,
    StorageError,
    UnknownLabelError,
)
from oblivlib.dependency.protocol import (
    AttachTree,
    Batch,
    BatchReply,
    Close,
    CreateTree,
    ErrorReply,
    Hello,
    HelloReply,
    Ok,
    ReadPath,
    Shutdown,
    Timed,
    WritePath,
    WriteRange,
    decode_request,
    decode_timed,
    encode,
    error_reply,
    expect_reply,
)


def test_every_message_round_trips():
    requests = [
        Hello(version=1),
        CreateTree(session="s", label="t", level=3, row_bytes=64),
        CreateTree(session="s", label="t", level=3, row_bytes=None),
        AttachTree(session="s", label="t", filename="f.tree", level=2, row_bytes=8),
        WriteRange(session="s", label="t", start=5, rows=[b"a", b""]),
        Batch(
            session="s",
            writes=[
                WritePath(label="t", leaves=[0, 3], rows=[b"x", b"", b"z"]),
            ],
            reads=[
                ReadPath(label="t", leaves=[1]),
                ReadPath(label="t", leaves=[2, 3]),
            ],
        ),
        Batch(session="s", writes=[], reads=[]),
        Close(session="s"),
        Shutdown(),
    ]
    replies = [
        HelloReply(session="s"),
        Ok(),
        BatchReply(results=[[b"r", b""], []]),
        ErrorReply(kind="RowSizeError", message="m"),
    ]
    for request in requests:
        assert decode_request(encode(request)) == request
    for reply in replies:
        assert decode_timed(encode(Timed(server_ns=7, reply=reply))) == Timed(server_ns=7, reply=reply)


def test_error_replies_map_to_error_classes():
    for exc, kind in [
        (UnknownLabelError("tree 't' is not hosted"), UnknownLabelError),
        (DuplicateLabelError("dup"), DuplicateLabelError),
        (FileNotFoundError("gone"), StorageError),
        (RuntimeError("boom"), ServerError),
    ]:
        reply = error_reply(exc)
        with pytest.raises(kind, match=str(exc)):
            expect_reply(reply, Ok)

    assert expect_reply(Ok(), Ok) == Ok()
    with pytest.raises(ProtocolError, match="unknown error kind"):
        expect_reply(ErrorReply(kind="KeyboardInterrupt", message="m"), Ok)
    with pytest.raises(ProtocolError, match="expected BatchReply"):
        expect_reply(Ok(), BatchReply)
