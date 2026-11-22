import os

import pytest

from oblivlib.dependency import (
    DuplicateLabelError,
    ProtocolError,
    RowSizeError,
    StorageError,
    UnknownLabelError,
)
from oblivlib.dependency.protocol import (
    AttachTree,
    Batch,
    BatchReply,
    Close,
    CreateTree,
    Hello,
    HelloReply,
    Ok,
    ReadPath,
    Reply,
    WritePath,
    WriteRange,
    expect_reply,
)
from oblivlib.dependency.storage_server import StorageServer

ROW = 4


@pytest.fixture(params=["memory", "disk"])
def server(request, tmp_path):
    storage = StorageServer(tmp_path / "store" if request.param == "disk" else None)
    yield storage
    storage.close()


def _open(server: StorageServer) -> str:
    return expect_reply(server.handle(Hello(version=1)), HelloReply).session


def _ok(reply: Reply) -> None:
    expect_reply(reply, Ok)


def _read(server: StorageServer, session: str, *reads: ReadPath) -> list[list[bytes]]:
    return expect_reply(server.handle(Batch(session=session, writes=[], reads=list(reads))), BatchReply).results


def _row(tag: int) -> bytes:
    return bytes([tag]) * ROW


def test_hello_opens_sessions_and_checks_version(server):
    reply = expect_reply(server.handle(Hello(version=1)), HelloReply)
    assert len(reply.session) == 32

    with pytest.raises(ProtocolError, match="version 2"):
        expect_reply(server.handle(Hello(version=2)), HelloReply)
    with pytest.raises(ProtocolError, match="unknown session"):
        _ok(server.handle(CreateTree(session="nope", label="t", level=2, row_bytes=ROW)))


def test_create_tree_checks_shape(server):
    session = _open(server)
    for level, row_bytes in [(0, ROW), (3, 0)]:
        with pytest.raises(ProtocolError, match="must be >= 1"):
            _ok(server.handle(CreateTree(session=session, label="t", level=level, row_bytes=row_bytes)))


def test_write_range_then_read_path_root_first_deduped(server):
    session = _open(server)
    _ok(server.handle(CreateTree(session=session, label="t", level=3, row_bytes=ROW)))
    _ok(server.handle(WriteRange(session=session, label="t", start=0, rows=[_row(1), _row(2), b"", _row(4)])))

    results = _read(server, session, ReadPath(label="t", leaves=[1, 0, 1]), ReadPath(label="t", leaves=[3]))
    assert results == [[_row(1), _row(2), _row(4), b""], [_row(1), b"", b""]]

    with pytest.raises(ProtocolError, match="outside"):
        _ok(server.handle(WriteRange(session=session, label="t", start=6, rows=[_row(1), _row(1)])))


def test_batch_is_validated_before_any_write(server):
    session = _open(server)
    for label in ("good", "t"):
        _ok(server.handle(CreateTree(session=session, label=label, level=2, row_bytes=ROW)))
    good = WritePath(label="good", leaves=[0], rows=[_row(9), _row(9)])

    cases: list[tuple[list[WritePath], list[ReadPath], type[Exception]]] = [
        ([WritePath(label="missing", leaves=[0], rows=[_row(1), _row(1)])], [], UnknownLabelError),
        ([WritePath(label="t", leaves=[2], rows=[_row(1), _row(1)])], [], ProtocolError),
        ([WritePath(label="t", leaves=[0], rows=[_row(1)])], [], ProtocolError),
        ([WritePath(label="t", leaves=[0], rows=[_row(1), b"short"])], [], RowSizeError),
        ([], [ReadPath(label="t", leaves=[-1])], ProtocolError),
        ([], [ReadPath(label="missing", leaves=[0])], UnknownLabelError),
    ]
    for writes, reads, error in cases:
        with pytest.raises(error):
            expect_reply(server.handle(Batch(session=session, writes=[good, *writes], reads=reads)), BatchReply)
        assert _read(server, session, ReadPath(label="good", leaves=[0])) == [[b"", b""]]


def test_variable_rows_are_memory_only(tmp_path):
    memory = StorageServer()
    session = _open(memory)
    _ok(memory.handle(CreateTree(session=session, label="t", level=2, row_bytes=None)))
    write = WritePath(label="t", leaves=[1], rows=[b"root bucket", b"x"])
    expect_reply(memory.handle(Batch(session=session, writes=[write], reads=[])), BatchReply)
    assert _read(memory, session, ReadPath(label="t", leaves=[0, 1])) == [[b"root bucket", b"", b"x"]]
    _ok(memory.handle(Close(session=session)))

    disk = StorageServer(tmp_path)
    with pytest.raises(ProtocolError, match="fixed-size"):
        _ok(disk.handle(CreateTree(session=_open(disk), label="t", level=2, row_bytes=None)))
    disk.close()


def test_attach_adopts_file_and_rejects_unsafe_names(tmp_path):
    store_dir = tmp_path / "store"
    server = StorageServer(store_dir)
    session = _open(server)
    image = store_dir / "image.tree"
    image.write_bytes(_row(1) + _row(2) + bytes(ROW))

    with pytest.raises(RowSizeError):
        _ok(server.handle(AttachTree(session=session, label="t", filename="image.tree", level=3, row_bytes=ROW)))
    for name in ("../image.tree", str(image), "a/image.tree", "..", "", session, "missing.tree"):
        with pytest.raises((ProtocolError, StorageError)):
            _ok(server.handle(AttachTree(session=session, label="t", filename=name, level=2, row_bytes=ROW)))
    open_fds = len(os.listdir("/proc/self/fd"))
    with pytest.raises(StorageError):
        _ok(server.handle(AttachTree(session=session, label="t" * 200, filename="image.tree", level=2, row_bytes=ROW)))
    assert image.exists() and len(os.listdir("/proc/self/fd")) == open_fds

    _ok(server.handle(AttachTree(session=session, label="t", filename="image.tree", level=2, row_bytes=ROW)))
    assert not image.exists()
    assert _read(server, session, ReadPath(label="t", leaves=[0, 1])) == [[_row(1), _row(2), b""]]

    memory = StorageServer()
    with pytest.raises(ProtocolError, match="in-memory"):
        _ok(memory.handle(AttachTree(session=_open(memory), label="t", filename="x.tree", level=2, row_bytes=ROW)))
    server.close()


def test_close_deletes_session_files_and_labels_are_per_session(tmp_path):
    store_dir = tmp_path / "store"
    server = StorageServer(store_dir)
    first, second = _open(server), _open(server)
    for session in (first, second):
        _ok(server.handle(CreateTree(session=session, label="t", level=2, row_bytes=ROW)))
    with pytest.raises(DuplicateLabelError):
        _ok(server.handle(CreateTree(session=first, label="t", level=2, row_bytes=ROW)))
    assert (store_dir / first).is_dir()

    _ok(server.handle(Close(session=first)))
    assert not (store_dir / first).exists() and (store_dir / second).is_dir()
    with pytest.raises(ProtocolError, match="unknown session"):
        _read(server, first, ReadPath(label="t", leaves=[0]))

    server.close()
    assert not (store_dir / second).exists()
