import os
import random
import secrets
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, override

import pytest

from oblivlib.dependency import (
    AesGcm,
    Client,
    LocalBackend,
    LoopbackTransport,
    StorageServer,
    TransportBackend,
)
from oblivlib.dependency.protocol import Batch, Reply, Request
from oblivlib.graph import GraphOS, Grove
from oblivlib.omap import GroupOmap, OramOstOmap
from oblivlib.omap.ost_base_omap import OstBaseOmap
from oblivlib.oram import TreeBaseOram, freecursive_oram

DEFAULT_NUM_DATA = 2**12

AccessShape = tuple[dict[str, frozenset[int]], dict[str, tuple[int, ...]]]


class RecordingBackend(LocalBackend):
    def __init__(self, server: StorageServer):
        super().__init__(server)
        self.messages: list[Request] = []

    @override
    def call(self, message: Request) -> Reply:
        self.messages.append(message)
        return super().call(message)

    @property
    def batches(self) -> list[Batch]:
        return [message for message in self.messages if isinstance(message, Batch)]


def _access_shape(batch: Batch) -> AccessShape:
    writes = {op.label: frozenset(op.leaves) for op in batch.writes}
    reads = {op.label: tuple(op.leaves) for op in batch.reads}
    return writes, reads


def _coalesce(shapes: list[AccessShape]) -> list[AccessShape]:
    coalesced: list[AccessShape] = []
    pending: dict[str, frozenset[int]] = {}
    for writes, reads in shapes:
        if not writes and not reads and not pending:
            coalesced.append(({}, {}))
            continue
        for label, leaves in writes.items():
            pending[label] = pending.get(label, frozenset()) | leaves
        if reads:
            coalesced.append((pending, reads))
            pending = {}
    if pending:
        coalesced.append((pending, {}))
    return coalesced


def _read_all_blocks(storage):
    client = storage._client
    client.add_read_path(storage.name, range(storage.leaf_range))
    path = storage._cipher.open_path(client.execute().require(storage.name))
    return [data for bucket in path.values() for data in bucket]


def _storages(scheme):
    if isinstance(scheme, OramOstOmap):
        return [*_storages(scheme._oram), *_storages(scheme._ost)]
    if isinstance(scheme, GraphOS):
        return [*_storages(scheme._omap), scheme._value_tree]
    if isinstance(scheme, Grove):
        return [*_storages(scheme._index), scheme._graph]
    if isinstance(scheme, GroupOmap):
        return [*_storages(scheme._upper_oram), scheme._buckets]
    if isinstance(scheme, TreeBaseOram):
        levels: list[Any] = list(scheme._pos_map_levels())
        return [scheme, *levels]
    return [scheme]


def _duplicate_blocks(scheme):
    duplicates = {}
    for storage in _storages(scheme):
        held = _read_all_blocks(storage) + storage._stash
        if isinstance(storage, OstBaseOmap):
            held += storage._local.to_list()
        counts = Counter(data.key for data in held)
        duplicates.update({(storage.name, key): count for key, count in counts.items() if count > 1})
    return duplicates


def _seed(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    rng = random.Random(seed)
    monkeypatch.setattr(secrets, "randbelow", rng.randrange)
    monkeypatch.setattr(os, "urandom", rng.randbytes)
    monkeypatch.setattr(freecursive_oram, "_CSPRNG", random.Random(seed))


@pytest.fixture
def num_data() -> int:
    return int(os.environ.get("NUM_DATA", DEFAULT_NUM_DATA))


@pytest.fixture
def client():
    return Client.local()


@pytest.fixture
def recorded_client():

    def make(*, defer_writes: bool = True, server: StorageServer | None = None) -> tuple[Client, RecordingBackend]:
        recorder = RecordingBackend(server or StorageServer())
        return Client(recorder, defer_writes=defer_writes), recorder

    return make


@pytest.fixture
def handover_client(tmp_path):

    def make(mode: Literal["adopt", "attach", "stream"]) -> Client:
        server_dir = tmp_path / f"{mode}_server"
        match mode:
            case "adopt":
                return Client.local(storage_dir=server_dir)
            case "stream":
                return Client(TransportBackend(LoopbackTransport(StorageServer())))
            case "attach":

                def ship(path: Path) -> str:
                    shutil.copy(path, server_dir / path.name)
                    return path.name

                return Client(TransportBackend(LoopbackTransport(StorageServer(server_dir))), ship_file=ship)

    return make


@pytest.fixture
def seeded_access(monkeypatch, recorded_client):

    def run(workload: Callable[[Client], None], *, defer_writes: bool = True) -> tuple[list[AccessShape], int]:
        _seed(monkeypatch, 7)
        client, recorder = recorded_client(defer_writes=defer_writes)
        workload(client)
        rounds = client.metrics.rounds
        client.flush()
        return [_access_shape(batch) for batch in recorder.batches], rounds

    return run


@pytest.fixture
def assert_deferral_preserves_access(seeded_access):

    def run(workload: Callable[[Client], None]) -> tuple[int, int]:
        shapes_off, rounds_off = seeded_access(workload, defer_writes=False)
        shapes_on, rounds_on = seeded_access(workload)
        assert _coalesce(shapes_off) == shapes_on
        return rounds_off, rounds_on

    return run


@pytest.fixture
def duplicate_blocks():
    return _duplicate_blocks


@pytest.fixture
def read_all_blocks():
    return _read_all_blocks


@pytest.fixture
def encryptor():
    return AesGcm()


@pytest.fixture
def fail_once(monkeypatch):
    def patch(target, name):
        original = getattr(target, name)
        calls = []

        def failing(*args, **kwargs):
            if not calls:
                calls.append(1)
                raise RuntimeError("injected")
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, failing)

    return patch
