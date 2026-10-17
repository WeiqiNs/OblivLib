"""Abstract base class for binary-tree-based ORAMs.

Adds the ORAM-specific layer (position-map levels, eviction, the operate_on_key access template) on top
of the shared ``TreeStorageBase`` (config accessors, level/stash math, random leaves, path encryption).
Private members use single underscores (not name-mangled ``__``) so subclasses can access them.
"""

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
    """The fewest position-map levels ``L`` with ``on_chip * ratio**L >= num_data``."""
    if ratio < 2:
        raise ValueError(f"ratio must be >= 2 to compress {num_data} entries into {on_chip}, got {ratio}.")
    count, capacity = 0, on_chip
    while capacity < num_data:
        capacity *= ratio
        count += 1
    return count


def pos_map_keys(key: int, *, ratio: int, count: int) -> list[tuple[int, int]]:
    """For each of ``count`` position maps (outermost first), the (block key, offset within block) for key."""
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
        """Peak stash size observed, including every position-map level's, for sizing client storage.
        (An upper bound: the per-level peaks need not occur at the same time.)"""
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
        """Pull the paths to ``leaves`` into the stash, read key (optionally writing value), remap it to
        new_leaf, and leave those paths pending for ``_write_back``."""
        self._client.add_read_path(label=self._name, leaves=leaves)
        self._absorb_path(path=self._client.execute().require(self._name))
        data = self._require_in_stash(key=key)
        read_value = data.value
        if value is not UNSET:
            data.value = value
        data.leaf = new_leaf
        self._pending_leaves = leaves
        return read_value

    def _write_back(self) -> None:
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=self._pending_leaves))
        self._client.execute()
        self._pending_leaves = []

    @abstractmethod
    def init_server_storage(self, data: InitData | None = None) -> None:
        """Build and host this oram's server storage. ``data`` gives initial values, as a mapping or a
        one-shot stream of ``(key, value)`` pairs; every other key starts as ``b""``."""

    def _require_access(self, key: Any, value: Any) -> None:
        require_oram_key(self._identity, key, self._num_data)
        if value is not UNSET:
            require_value(self._identity, value, self._data_size)

    def operate_on_key(self, key: int, value: Any = UNSET) -> bytes:
        """Return key's current value (``b""`` until written); write ``value`` when one is given."""
        self._require_access(key, value)
        read_value = self._operate_on_key_without_eviction(key, value)
        self._write_back()
        return read_value

    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> bytes:
        """Like ``operate_on_key`` but leaves the path's write-back to ``eviction_with_update_stash``."""
        self._require_access(key, value)
        return self._operate_on_key_without_eviction(key, value)

    def eviction_with_update_stash(self, key: int, value: bytes) -> None:
        """Set key's value in the stash, then evict and write the path back."""
        require_oram_key(self._identity, key, self._num_data)
        require_value(self._identity, value, self._data_size)
        if not self._pending_leaves:
            raise ValueError(f"{self._identity}: no access is waiting for its eviction.")
        self._require_in_stash(key).value = value
        self._write_back()

    @abstractmethod
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any: ...
