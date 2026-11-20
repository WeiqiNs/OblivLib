import secrets
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any

from oblivlib.dependency.client import Client
from oblivlib.dependency.config import OramConfig
from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.errors import StashOverflowError
from oblivlib.dependency.heap_index import compute_level, empty_path, fill_data_to_path
from oblivlib.dependency.layout import Bytes, Field, Record, UInt, block_layout
from oblivlib.dependency.path_cipher import PathCipher, make_path_cipher
from oblivlib.dependency.tree_builder import TreeImage, build_tree
from oblivlib.dependency.types import Data, PathData


class TreeStorageBase[ConfigT: OramConfig]:
    def __init__(self, config: ConfigT):
        self._config: ConfigT = config

        self._level: int = compute_level(config.num_data)
        self._leaf_range: int = 1 << (self._level - 1)
        self._stash_capacity: int = config.stash_scale * max(1, self._level - 1)

        self._stash: list[Any] = []
        self._max_stash: int = 0
        self._unwritten_leaves: list[int] = []

    @property
    def _name(self) -> str:
        return self._config.name

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._name!r}"

    @property
    def _build_file(self) -> Path | None:
        return None if self._config.build_file is None else Path(self._config.build_file)

    @property
    def _num_data(self) -> int:
        return self._config.num_data

    @property
    def _data_size(self) -> int:
        return self._config.data_size

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
    def stash_size(self) -> int:
        return len(self._stash)

    def _get_new_leaf(self) -> int:
        return secrets.randbelow(self._leaf_range)

    def _check_stash(self) -> None:
        self._max_stash = max(self._max_stash, self.stash_size)
        if len(self._stash) > self._stash_capacity:
            raise StashOverflowError(
                f"{type(self).__name__} {self._name!r}: stash holds {len(self._stash)} blocks, "
                + f"capacity {self._stash_capacity}"
            )

    def _require_in_stash(self, key: int) -> Data:
        for data in self._stash:
            if data.key == key:
                return data
        raise KeyError(f"Key {key} not found.")

    def _update_stash_leaf(self, key: int | None, new_leaf: int | None) -> None:
        if key is not None:
            self._require_in_stash(key).leaf = new_leaf

    def _stage_write_back(self) -> None:
        if not self._unwritten_leaves:
            return
        leaves = self._unwritten_leaves
        path = empty_path(level=self._level, leaves=leaves)
        remaining = [
            data
            for data in self._stash
            if not fill_data_to_path(data, path, leaves=leaves, level=self._level, bucket_size=self._bucket_size)
        ]
        self._client.add_write_path(label=self._name, data=self._cipher.seal_path(path))
        self._stash = remaining
        self._unwritten_leaves = []

    def _open_paths(self, leaves: list[int]) -> PathData:
        self._stage_write_back()
        self._client.add_read_path(label=self._name, leaves=leaves)
        path = self._cipher.open_path(self._client.execute().require(self._name))
        self._unwritten_leaves = list(leaves)
        return path

    def _read_paths(self, leaves: list[int]) -> None:
        for bucket in self._open_paths(leaves).values():
            self._stash.extend(bucket)
        self._check_stash()

    def _write_paths(self) -> None:
        self._stage_write_back()
        self._client.execute()

    @cached_property
    def _key_field(self) -> Field[Any]:
        return UInt.covering(self._num_data - 1)

    @cached_property
    def _leaf_field(self) -> UInt:
        return UInt.covering(self._leaf_range - 1)

    @cached_property
    def _value_field(self) -> Field[Any]:
        return Bytes(self._data_size)

    @cached_property
    def _block_layout(self) -> Record:
        return block_layout(self._key_field, self._leaf_field, self._value_field)

    @cached_property
    def _cipher(self) -> PathCipher:
        return make_path_cipher(self._block_layout, self._encryptor, self._bucket_size)

    def _build_tree(self, blocks: Iterable[Data]) -> TreeImage:
        result = build_tree(
            blocks, level=self._level, bucket_size=self._bucket_size, cipher=self._cipher, build_file=self._build_file
        )
        self._stash.extend(result.overflow)
        self._check_stash()
        return result.image

    def _host_tree(self, image: TreeImage) -> None:
        self._client.host_tree(self._name, image)
