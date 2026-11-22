import secrets
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Protocol

from oblivlib.dependency.client import Client, ExecuteResult
from oblivlib.dependency.config import OramConfig
from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.errors import StashOverflowError
from oblivlib.dependency.heap_index import compute_level, empty_path, fill_data_to_path, leaf_count
from oblivlib.dependency.layout import Field, UInt
from oblivlib.dependency.path_cipher import PathCipher, make_path_cipher
from oblivlib.dependency.tree_builder import TreeImage, build_tree
from oblivlib.dependency.types import Block, PathData


def leaf_field(num_data: int) -> UInt:
    return UInt.covering(leaf_count(num_data) - 1)


class PathReader(Protocol):
    def stage_read(self, leaves: list[int]) -> None: ...

    def finish_read(self, result: ExecuteResult) -> None: ...


def read_together(client: Client, reads: Sequence[tuple[PathReader, list[int]]]) -> None:
    for store, leaves in reads:
        store.stage_read(leaves)
    result = client.execute()
    for store, _ in reads:
        store.finish_read(result)


class TreeStorageBase[ConfigT: OramConfig, BlockT: Block[object]](ABC):
    def __init__(self, config: ConfigT):
        self._config: ConfigT = config

        self._level: int = compute_level(config.num_data)
        self._stash_capacity: int = config.stash_scale * max(1, self._level - 1)

        self._stash: list[BlockT] = []
        self._max_stash: int = 0
        self._unwritten_leaves: list[int] = []

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self.name!r}"

    @property
    def _build_file(self) -> Path | None:
        return None if self._config.build_file is None else Path(self._config.build_file)

    @property
    def _bucket_size(self) -> int:
        return self._config.bucket_size

    @property
    def _encryptor(self) -> Encryptor | None:
        return self._config.encryptor

    @property
    def _client(self) -> Client:
        return self._config.client

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def num_data(self) -> int:
        return self._config.num_data

    @property
    def data_size(self) -> int:
        return self._config.data_size

    @property
    def leaf_range(self) -> int:
        return 1 << (self._level - 1)

    @property
    def max_stash(self) -> int:
        return self._max_stash

    @property
    def stash_size(self) -> int:
        return len(self._stash)

    def _get_new_leaf(self) -> int:
        return secrets.randbelow(self.leaf_range)

    @property
    def _stash_limit(self) -> int:
        return self._stash_capacity

    def _check_stash(self) -> None:
        self._max_stash = max(self._max_stash, self.stash_size)
        if len(self._stash) > self._stash_limit:
            raise StashOverflowError(
                f"{type(self).__name__} {self.name!r}: stash holds {len(self._stash)} blocks, "
                + f"capacity {self._stash_limit}"
            )

    def _require_in_stash(self, key: object) -> BlockT:
        found = [data for data in self._stash if data.key == key]
        assert found, f"key {key!r} is not in the stash"
        return found[0]

    def padded_leaves(self, leaves: Iterable[int], width: int) -> list[int]:
        paths = set(leaves)
        assert len(paths) <= width
        while len(paths) < width:
            paths.add(self._get_new_leaf())
        return sorted(paths)

    def _remap(self, moves: Iterable[tuple[object, int]]) -> None:
        for key, leaf in moves:
            self._require_in_stash(key).leaf = leaf

    def _stage_write_back(self) -> None:
        if not self._unwritten_leaves:
            return
        leaves = self._unwritten_leaves
        path: PathData[BlockT] = empty_path(level=self._level, leaves=leaves)
        remaining = [
            data
            for data in self._stash
            if not fill_data_to_path(data, path, leaves=leaves, level=self._level, bucket_size=self._bucket_size)
        ]
        self._client.add_write_path(label=self.name, data=self._cipher.seal_path(path))
        self._stash = remaining
        self._unwritten_leaves = []

    def stage_read(self, leaves: list[int]) -> None:
        self._stage_write_back()
        self._client.add_read_path(label=self.name, leaves=leaves)
        self._unwritten_leaves = list(leaves)

    def finish_read(self, result: ExecuteResult) -> None:
        self._absorb(self._cipher.open_path(result.require(self.name)))

    def _absorb(self, path: PathData[BlockT]) -> None:
        for bucket in path.values():
            self._stash.extend(bucket)
        self._check_stash()

    def _read_paths(self, leaves: list[int]) -> None:
        self.stage_read(leaves)
        self.finish_read(self._client.execute())

    def _write_paths(self) -> None:
        self._stage_write_back()
        self._client.send_writes()

    def _pass_paths(self, leaves: list[int], moves: Iterable[tuple[object, int]]) -> None:
        self._read_paths(leaves)
        self._remap(moves)
        self._write_paths()

    @cached_property
    def _leaf_field(self) -> UInt:
        return leaf_field(self.num_data)

    @property
    @abstractmethod
    def _block_layout(self) -> Field[BlockT]: ...

    @cached_property
    def _cipher(self) -> PathCipher[BlockT]:
        return make_path_cipher(
            self._block_layout, self._encryptor, bucket_size=self._bucket_size, recorder=self._client.recorder
        )

    def _build_tree(self, blocks: Iterable[BlockT]) -> TreeImage:
        result = build_tree(
            blocks, level=self._level, bucket_size=self._bucket_size, cipher=self._cipher, build_file=self._build_file
        )
        self._stash.extend(result.overflow)
        self._check_stash()
        return result.image

    def _host_tree(self, image: TreeImage) -> None:
        self._client.host_tree(self.name, image)
