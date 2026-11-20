import os
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from oblivlib.dependency.errors import DuplicateLabelError, ProtocolError, StorageError, UnknownLabelError
from oblivlib.dependency.heap_index import path_indices
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
    Reply,
    Request,
    Shutdown,
    WriteRange,
    error_reply,
)
from oblivlib.dependency.server_stores import FixedRowStore, TreeStore, VariableRowStore


@dataclass
class _Session:
    directory: Path | None
    trees: dict[str, TreeStore] = field(default_factory=dict)

    def require_tree(self, label: str) -> TreeStore:
        if label not in self.trees:
            raise UnknownLabelError(f"StorageServer: tree label {label!r} is not hosted.")
        return self.trees[label]

    def require_new_label(self, label: str) -> None:
        if label in self.trees:
            raise DuplicateLabelError(f"StorageServer: label {label!r} is already hosted.")

    def tree_path(self, label: str) -> Path | None:
        return None if self.directory is None else self.directory / f"{label.encode().hex()}.tree"

    def close(self) -> None:
        for tree in self.trees.values():
            tree.close()
        self.trees.clear()
        if self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)


def _check_tree_shape(level: int, row_bytes: int | None) -> None:
    if level < 1:
        raise ProtocolError(f"StorageServer: tree level must be >= 1, got {level}.")
    if row_bytes is not None and row_bytes < 1:
        raise ProtocolError(f"StorageServer: row_bytes must be >= 1, got {row_bytes}.")


def _path_of(tree: TreeStore, leaves: list[int]) -> list[int]:
    leaf_range = 1 << (tree.level - 1)
    for leaf in leaves:
        if not 0 <= leaf < leaf_range:
            raise ProtocolError(f"StorageServer: leaf {leaf} is outside [0, {leaf_range}).")
    return path_indices(leaves, tree.level)


def _validate_batch(session: _Session, batch: Batch) -> None:
    for write in batch.writes:
        tree = session.require_tree(write.label)
        indices = _path_of(tree, write.leaves)
        if len(write.rows) != len(indices):
            raise ProtocolError(
                f"StorageServer: write to {write.label!r} carries {len(write.rows)} rows for {len(indices)} path nodes."
            )
        for row in write.rows:
            tree.check_row(row)
    for read in batch.reads:
        _path_of(session.require_tree(read.label), read.leaves)


def _apply_batch(session: _Session, batch: Batch) -> BatchReply:
    for write in batch.writes:
        tree = session.trees[write.label]
        tree.write_rows(path_indices(write.leaves, tree.level), write.rows)
    results = []
    for read in batch.reads:
        tree = session.trees[read.label]
        results.append(tree.read_rows(path_indices(read.leaves, tree.level)))
    return BatchReply(results=results)


class StorageServer:
    """The storage engine behind every client: answers protocol messages, with no transport of its own.

    Each ``Hello`` opens a session whose labels are its own. ``storage_dir=None`` keeps every tree in
    memory; otherwise fixed-row trees live in files under ``storage_dir/<session>/`` and variable-row
    (plaintext) trees are refused. A ``Batch`` is validated in full before any of it is applied, and
    ``Close`` releases the session and deletes its files. ``handle`` never raises: failures come back as
    an ``ErrorReply``.
    """

    def __init__(self, storage_dir: str | Path | None = None):
        self._storage_dir = None if storage_dir is None else Path(storage_dir)
        self._sessions: dict[str, _Session] = {}

    @property
    def storage_dir(self) -> Path | None:
        return self._storage_dir

    def handle(self, message: Request) -> Reply:
        try:
            return self._dispatch(message)
        except Exception as exc:
            return error_reply(exc)

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()

    def _require_session(self, session_id: str) -> _Session:
        if session_id not in self._sessions:
            raise ProtocolError(f"StorageServer: unknown session {session_id!r}.")
        return self._sessions[session_id]

    def _hello(self, version: int) -> HelloReply:
        if version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"StorageServer: protocol version {version} is unsupported; expected {PROTOCOL_VERSION}."
            )
        session_id = secrets.token_hex(16)
        directory = None if self._storage_dir is None else self._storage_dir / session_id
        if directory is not None:
            directory.mkdir(parents=True)
        self._sessions[session_id] = _Session(directory=directory)
        return HelloReply(session=session_id)

    def _create_tree(self, session: _Session, label: str, level: int, row_bytes: int | None) -> None:
        session.require_new_label(label)
        _check_tree_shape(level, row_bytes)
        path = session.tree_path(label)
        if row_bytes is None:
            if path is not None:
                raise ProtocolError("StorageServer: a disk-backed server only hosts fixed-size rows.")
            session.trees[label] = VariableRowStore(level)
        elif path is None:
            session.trees[label] = FixedRowStore.in_memory(level, row_bytes)
        else:
            session.trees[label] = FixedRowStore.create_file(path, level, row_bytes)

    def _adoptable_file(self, filename: str) -> Path:
        if self._storage_dir is None:
            raise ProtocolError("StorageServer: an in-memory server cannot attach a file.")
        if filename in ("", "..") or "\\" in filename or Path(filename).name != filename:
            raise ProtocolError(f"StorageServer: {filename!r} is not a plain file name inside the storage directory.")
        source = self._storage_dir / filename
        if not source.is_file():
            raise StorageError(f"StorageServer: {filename!r} is not a file in the storage directory.")
        return source

    def _attach_tree(self, session: _Session, message: AttachTree) -> None:
        source = self._adoptable_file(message.filename)
        session.require_new_label(message.label)
        _check_tree_shape(message.level, message.row_bytes)
        target = session.tree_path(message.label)
        assert target is not None
        store = FixedRowStore.open_file(source, message.level, message.row_bytes)
        try:
            os.replace(source, target)
        except OSError:
            store.close()
            raise
        session.trees[message.label] = store

    def _write_range(self, session: _Session, message: WriteRange) -> None:
        tree = session.require_tree(message.label)
        if message.start < 0 or message.start + len(message.rows) > tree.size:
            raise ProtocolError(
                f"StorageServer: rows [{message.start}, {message.start + len(message.rows)}) "
                + f"fall outside {message.label!r}'s {tree.size} rows."
            )
        for row in message.rows:
            tree.check_row(row)
        tree.write_range(message.start, message.rows)

    def _dispatch(self, message: Request) -> Reply:
        match message:
            case Hello(version=version):
                return self._hello(version)
            case Shutdown():
                return Ok()
            case Batch(session=session_id):
                session = self._require_session(session_id)
                _validate_batch(session, message)
                return _apply_batch(session, message)
            case CreateTree(session=session_id, label=label, level=level, row_bytes=row_bytes):
                self._create_tree(self._require_session(session_id), label, level, row_bytes)
            case AttachTree(session=session_id):
                self._attach_tree(self._require_session(session_id), message)
            case WriteRange(session=session_id):
                self._write_range(self._require_session(session_id), message)
            case Close(session=session_id):
                self._require_session(session_id).close()
                del self._sessions[session_id]
        return Ok()
