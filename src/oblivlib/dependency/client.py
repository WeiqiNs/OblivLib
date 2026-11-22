import secrets
import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self, override

from oblivlib.dependency.errors import MissingResultError, UnknownLabelError
from oblivlib.dependency.heap_index import path_indices, path_leaves
from oblivlib.dependency.metrics import Measurement, Metrics, Recorder
from oblivlib.dependency.protocol import (
    PROTOCOL_VERSION,
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
    Request,
    WritePath,
    WriteRange,
    decode_timed,
    encode,
    expect_reply,
)
from oblivlib.dependency.storage_server import StorageServer
from oblivlib.dependency.transport import DEFAULT_TIMEOUT_MS, Transport, ZmqTransport
from oblivlib.dependency.tree_builder import FileImage, TreeImage
from oblivlib.dependency.types import PathRows

WRITE_RANGE_CHUNK_BYTES = 32 * 2**20


@dataclass(frozen=True)
class ExecuteResult:
    """The reads of one ``Client.execute()``, keyed by label: each path read maps each node's heap index to
    its row (``b""`` for an absent node)."""

    results: dict[str, PathRows]

    def require(self, label: str) -> PathRows:
        if label not in self.results:
            raise MissingResultError(f"ExecuteResult has no result for label {label!r}.")
        return self.results[label]


class Backend(ABC):
    @abstractmethod
    def call(self, message: Request) -> Reply: ...

    @property
    @abstractmethod
    def bytes_sent(self) -> int: ...

    @property
    @abstractmethod
    def bytes_received(self) -> int: ...

    @property
    @abstractmethod
    def client_wire_ns(self) -> int: ...

    @property
    @abstractmethod
    def transport_ns(self) -> int: ...

    @property
    @abstractmethod
    def server_ns(self) -> int: ...

    @property
    @abstractmethod
    def storage_dir(self) -> Path | None:
        """Where the server keeps hosted files, when the client can place a file there directly."""

    @abstractmethod
    def close(self) -> None: ...


class LocalBackend(Backend):
    def __init__(self, server: StorageServer):
        self._server: StorageServer = server
        self._server_ns: int = 0

    @override
    def call(self, message: Request) -> Reply:
        start = time.perf_counter_ns()
        reply = self._server.handle(message)
        self._server_ns += time.perf_counter_ns() - start
        return reply

    @property
    @override
    def bytes_sent(self) -> int:
        return 0

    @property
    @override
    def bytes_received(self) -> int:
        return 0

    @property
    @override
    def client_wire_ns(self) -> int:
        return 0

    @property
    @override
    def transport_ns(self) -> int:
        return 0

    @property
    @override
    def server_ns(self) -> int:
        return self._server_ns

    @property
    @override
    def storage_dir(self) -> Path | None:
        return self._server.storage_dir

    @override
    def close(self) -> None:
        pass


class TransportBackend(Backend):
    def __init__(self, transport: Transport):
        self._transport: Transport = transport
        self._bytes_sent: int = 0
        self._bytes_received: int = 0
        self._client_wire_ns: int = 0
        self._transport_ns: int = 0
        self._server_ns: int = 0

    @override
    def call(self, message: Request) -> Reply:
        start = time.perf_counter_ns()
        data = encode(message)
        sent = time.perf_counter_ns()
        raw = self._transport.request(data)
        received = time.perf_counter_ns()
        timed = decode_timed(raw)
        self._client_wire_ns += sent - start + time.perf_counter_ns() - received
        self._transport_ns += received - sent
        self._server_ns += timed.server_ns
        self._bytes_sent += len(data)
        self._bytes_received += len(raw)
        return timed.reply

    @property
    @override
    def bytes_sent(self) -> int:
        return self._bytes_sent

    @property
    @override
    def bytes_received(self) -> int:
        return self._bytes_received

    @property
    @override
    def client_wire_ns(self) -> int:
        return self._client_wire_ns

    @property
    @override
    def transport_ns(self) -> int:
        return self._transport_ns

    @property
    @override
    def server_ns(self) -> int:
        return self._server_ns

    @property
    @override
    def storage_dir(self) -> None:
        return None

    @override
    def close(self) -> None:
        self._transport.close()


class Client:
    """The client-side handle to a storage server; every scheme does all its I/O through one.

    ``add_*`` calls only stage queries, keyed by storage label. ``execute()`` sends them as one batch
    whose writes (merged per label, a later write winning) apply before its reads (deduplicated per label), and returns their ``ExecuteResult``. A failed batch raises
    the server's error and leaves nothing staged. An ``execute()`` with nothing staged is still a round.
    ``send_writes()`` is ``execute()`` for a batch that stages no reads, so it returns nothing.

    With ``defer_writes`` (the default), an ``execute()`` that stages only writes sends nothing: its writes
    ride on the next batch, ahead of that batch's reads. ``flush()`` sends them on their own, every
    lifecycle call flushes first, and ``close()`` flushes. An error from a deferred write surfaces on the
    call that sends it.

    ``host_tree`` hands a built tree to the server: a file image is adopted in place when the server's
    storage directory is local, placed out of band by ``ship_file`` (which returns its name inside the
    server's storage directory) and attached, or else streamed; a successful hand-over leaves no local build
    file.

    A client is not thread-safe: use one per thread.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        defer_writes: bool = True,
        ship_file: Callable[[Path], str] | None = None,
    ) -> None:
        self._backend: Backend = backend
        self._defer_writes: bool = defer_writes
        self._ship_file: Callable[[Path], str] | None = ship_file

        self._session: str | None = None
        self._levels: dict[str, int] = {}

        self._read_paths: dict[str, set[int]] = {}
        self._write_paths: dict[str, PathRows] = {}

        self._recorder: Recorder = Recorder()
        self._rounds: int = 0
        self._requests: int = 0
        self._row_bytes_written: int = 0
        self._row_bytes_read: int = 0
        self._closed: bool = False

    @classmethod
    def local(cls, storage_dir: str | Path | None = None, *, defer_writes: bool = True) -> Self:
        return cls(LocalBackend(StorageServer(storage_dir)), defer_writes=defer_writes)

    @classmethod
    def connect(cls, endpoint: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS, defer_writes: bool = True) -> Self:
        return cls(TransportBackend(ZmqTransport(endpoint, timeout_ms=timeout_ms)), defer_writes=defer_writes)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    @property
    def recorder(self) -> Recorder:
        return self._recorder

    @property
    def metrics(self) -> Metrics:
        backend, recorder = self._backend, self._recorder
        return Metrics(
            rounds=self._rounds,
            requests=self._requests,
            row_bytes_written=self._row_bytes_written,
            row_bytes_read=self._row_bytes_read,
            bytes_sent=backend.bytes_sent,
            bytes_received=backend.bytes_received,
            client_wire_ns=backend.client_wire_ns,
            transport_ns=backend.transport_ns,
            server_ns=backend.server_ns,
            rows_sealed=recorder.rows_sealed,
            rows_opened=recorder.rows_opened,
            sealed_bytes=recorder.sealed_bytes,
            opened_bytes=recorder.opened_bytes,
            encode_ns=recorder.encode_ns,
            encrypt_ns=recorder.encrypt_ns,
            decrypt_ns=recorder.decrypt_ns,
            decode_ns=recorder.decode_ns,
        )

    def measure(self) -> Measurement:
        return Measurement(lambda: self.metrics)

    def level_of(self, label: str) -> int:
        self._require_label(label)
        return self._levels[label]

    def add_read_path(self, label: str, leaves: Iterable[int]) -> None:
        self._require_label(label)
        self._read_paths.setdefault(label, set()).update(leaves)

    def add_write_path(self, label: str, data: PathRows) -> None:
        """Stage rows for whole root-to-leaf paths: ``data`` must hold exactly the nodes of the paths to
        the leaves among its indices."""
        level = self.level_of(label)
        assert set(data) == set(path_indices(path_leaves(data, level), level))
        self._write_paths.setdefault(label, {}).update(data)

    def execute(self) -> ExecuteResult:
        reads = self._take_reads()
        if self._defer_writes and not reads and self._has_pending_writes():
            return ExecuteResult({})
        return self._send_batch(self._take_writes(), reads)

    def send_writes(self) -> None:
        assert not self._read_paths, "send_writes() sends no reads; stage reads only before execute()"
        if not (self._defer_writes and self._has_pending_writes()):
            self._send_writes(self._take_writes())

    def flush(self) -> None:
        if self._has_pending_writes():
            self._send_writes(self._take_writes())

    def create_tree(self, label: str, *, level: int, row_bytes: int | None) -> None:
        self.flush()
        self._expect_ok(CreateTree(session=self._require_session(), label=label, level=level, row_bytes=row_bytes))
        self._levels[label] = level

    def host_tree(self, label: str, image: TreeImage) -> None:
        if isinstance(image, FileImage):
            self._host_file(label, image)
            return
        self.create_tree(label, level=image.level, row_bytes=image.row_bytes)
        self._stream_rows(label, image.rows)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            try:
                if self._session is not None:
                    self._expect_ok(Close(session=self._session))
            finally:
                self._backend.close()

    def _require_session(self) -> str:
        if self._session is None:
            self._session = expect_reply(self._call(Hello(version=PROTOCOL_VERSION)), HelloReply).session
        return self._session

    def _require_label(self, label: str) -> None:
        if label not in self._levels:
            raise UnknownLabelError(f"Client: tree label {label!r} is not hosted by this client.")

    def _expect_ok(self, message: Request) -> None:
        _ = expect_reply(self._call(message), Ok)

    def _host_file(self, label: str, image: FileImage) -> None:
        self.flush()
        session = self._require_session()
        filename = self._hand_over(image.path)
        if filename is None:
            self.create_tree(label, level=image.level, row_bytes=image.row_bytes)
            self._stream_rows(label, image.rows())
            image.path.unlink()
            return
        attach = AttachTree(
            session=session, label=label, filename=filename, level=image.level, row_bytes=image.row_bytes
        )
        try:
            self._expect_ok(attach)
        except BaseException:
            adopted = None if self._ship_file is not None else self._backend.storage_dir
            if adopted is not None and (adopted / filename).is_file():
                _ = shutil.move(adopted / filename, image.path)
            raise
        self._levels[label] = image.level

    def _hand_over(self, path: Path) -> str | None:
        if self._ship_file is not None:
            filename = self._ship_file(path)
            path.unlink()
            return filename
        storage_dir = self._backend.storage_dir
        if storage_dir is None:
            return None
        filename = f"adopt-{secrets.token_hex(8)}.tree"
        _ = shutil.move(path, storage_dir / filename)
        return filename

    def _stream_rows(self, label: str, rows: Iterable[bytes]) -> None:
        start = 0
        chunk: list[bytes] = []
        size = 0
        for row in rows:
            if chunk and size + len(row) > WRITE_RANGE_CHUNK_BYTES:
                self._write_range(label, start, chunk)
                start += len(chunk)
                chunk, size = [], 0
            chunk.append(row)
            size += len(row)
        self._write_range(label, start, chunk)

    def _write_range(self, label: str, start: int, rows: list[bytes]) -> None:
        self._expect_ok(WriteRange(session=self._require_session(), label=label, start=start, rows=rows))

    def _has_pending_writes(self) -> bool:
        return bool(self._write_paths)

    def _write_path_op(self, label: str, data: PathRows) -> WritePath:
        level = self._levels[label]
        leaves = path_leaves(data, level)
        return WritePath(label=label, leaves=leaves, rows=[data[i] for i in path_indices(leaves, level)])

    def _take_writes(self) -> list[WritePath]:
        writes = [self._write_path_op(label, data) for label, data in self._write_paths.items()]
        self._write_paths = {}
        return writes

    def _take_reads(self) -> list[ReadPath]:
        reads = [ReadPath(label=label, leaves=sorted(leaves)) for label, leaves in self._read_paths.items()]
        self._read_paths = {}
        return reads

    def _send_writes(self, writes: list[WritePath]) -> None:
        _ = self._send_batch(writes, [])

    def _send_batch(self, writes: list[WritePath], reads: list[ReadPath]) -> ExecuteResult:
        batch = Batch(session=self._require_session(), writes=writes, reads=reads)
        reply = expect_reply(self._call(batch), BatchReply)
        results: dict[str, PathRows] = {}
        for op, rows in zip(reads, reply.results, strict=True):
            results[op.label] = dict(zip(path_indices(op.leaves, self._levels[op.label]), rows, strict=True))
        return ExecuteResult(results)

    def _call(self, message: Request) -> Reply:
        reply = self._backend.call(message)
        self._requests += 1
        match message:
            case Batch(writes=writes):
                self._rounds += 1
                self._row_bytes_written += sum(len(row) for op in writes for row in op.rows)
            case WriteRange(rows=rows):
                self._row_bytes_written += sum(len(row) for row in rows)
            case _:
                pass
        if isinstance(reply, BatchReply):
            self._row_bytes_read += sum(len(row) for rows in reply.results for row in rows)
        return reply
