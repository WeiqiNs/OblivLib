from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from oblivlib.dependency import UNSET, ContractError, Data, InitData, OramConfig
from oblivlib.dependency.contract import require_oram_key, require_value
from oblivlib.dependency.tree_storage_base import TreeStorageBase


def _initial_pairs(data: InitData | None) -> Iterable[tuple[int, Any]]:
    if data is None:
        return ()
    if isinstance(data, Mapping):
        mapping: Mapping[Any, Any] = data
        return mapping.items()
    return data


def pos_map_level_count(*, num_data: int, on_chip: int, ratio: int) -> int:
    count, capacity = 0, on_chip
    while capacity < num_data:
        capacity *= ratio
        count += 1
    return count


def pos_map_keys(key: int, *, ratio: int, count: int) -> list[tuple[int, int]]:
    keys = []
    for _ in range(count):
        key, offset = divmod(key, ratio)
        keys.append((key, offset))
    keys.reverse()
    return keys


class TreeBaseOram[ConfigT: OramConfig, LevelT: TreeStorageBase[Any]](TreeStorageBase[ConfigT], ABC):
    def __init__(self, config: ConfigT):
        super().__init__(config)
        self._levels: list[LevelT] = []
        self._pending_leaves: list[int] = []

    @property
    def max_stash(self) -> int:
        return self._max_stash + sum(level._max_stash for level in self._levels)

    @abstractmethod
    def _initial_leaf(self, key: int) -> int: ...

    def _initial_blocks(self, data: InitData | None) -> Iterator[Data]:
        supplied = bytearray(self._num_data)
        for key, value in _initial_pairs(data):
            require_oram_key(self._identity, key, self._num_data)
            require_value(self._identity, value, self._data_size)
            if supplied[key]:
                raise ContractError(f"{self._identity}: initial key {key} is given twice.")
            supplied[key] = 1
            yield Data(key=key, leaf=self._initial_leaf(key), value=value)
        for key in range(self._num_data):
            if not supplied[key]:
                yield Data(key=key, leaf=self._initial_leaf(key), value=b"")

    def _read_block(self, key: int, value: Any, *, leaves: list[int], new_leaf: int) -> Any:
        self._read_paths(leaves)
        data = self._require_in_stash(key=key)
        read_value = data.value
        if value is not UNSET:
            data.value = value
        data.leaf = new_leaf
        self._pending_leaves = leaves
        return read_value

    def _write_back(self) -> None:
        self._write_paths(self._pending_leaves)
        self._pending_leaves = []

    @abstractmethod
    def init_server_storage(self, data: InitData | None = None) -> None: ...

    def _require_no_pending(self) -> None:
        if self._pending_leaves:
            raise ValueError(f"{self._identity}: an access is still waiting for its eviction.")

    def _require_pending(self) -> None:
        if not self._pending_leaves:
            raise ValueError(f"{self._identity}: no access is waiting for its eviction.")

    def _require_access(self, key: Any, value: Any) -> None:
        require_oram_key(self._identity, key, self._num_data)
        if value is not UNSET:
            require_value(self._identity, value, self._data_size)

    def operate_on_key(self, key: int, value: Any = UNSET) -> bytes:
        self._require_access(key, value)
        self._require_no_pending()
        read_value = self._operate_on_key_without_eviction(key, value)
        self._write_back()
        return read_value

    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> bytes:
        self._require_access(key, value)
        self._require_no_pending()
        return self._operate_on_key_without_eviction(key, value)

    def eviction_with_update_stash(self, key: int, value: bytes) -> None:
        require_oram_key(self._identity, key, self._num_data)
        require_value(self._identity, value, self._data_size)
        self._require_pending()
        self._require_in_stash(key).value = value
        self._write_back()

    @abstractmethod
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any: ...
