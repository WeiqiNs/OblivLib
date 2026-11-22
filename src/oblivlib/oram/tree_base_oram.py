from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Protocol, override

from oblivlib.dependency import UNSET, ContractError, InitData, OramConfig
from oblivlib.dependency.contract import require_oram_key, require_value
from oblivlib.dependency.layout import Bytes, Col, Field, Record, UInt
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.dependency.types import OramBlock, Unset


class Oram(Protocol):
    @property
    def num_data(self) -> int: ...

    @property
    def data_size(self) -> int: ...

    def init_server_storage(self, data: InitData | None = None) -> None: ...

    def update[T](self, key: int, change: Callable[[bytes], tuple[bytes, T]]) -> T: ...


def require_oram_fits(owner: str, oram: Oram, *, num_data: int, data_size: int, unit: str, sizing: str) -> None:
    if oram.num_data != num_data:
        raise ValueError(
            f"{owner}: the ORAM has num_data={oram.num_data}, not {num_data}; it needs one slot per {unit}."
        )
    if oram.data_size < data_size:
        raise ValueError(
            f"{owner}: the ORAM has data_size={oram.data_size}, too small for the {data_size} bytes a {unit} needs; "
            + f"size it with {sizing}."
        )


class PosMapLevel(Protocol):
    @property
    def max_stash(self) -> int: ...


def _initial_pairs(data: InitData | None) -> Iterable[tuple[int, bytes]]:
    if data is None:
        return ()
    if isinstance(data, Mapping):
        return data.items()
    return data


def pos_map_level_count(*, num_data: int, on_chip: int, ratio: int) -> int:
    count, capacity = 0, on_chip
    while capacity < num_data:
        capacity *= ratio
        count += 1
    return count


def pos_map_keys(key: int, *, ratio: int, count: int) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    for _ in range(count):
        key, offset = divmod(key, ratio)
        keys.append((key, offset))
    keys.reverse()
    return keys


class TreeBaseOram[ConfigT: OramConfig](TreeStorageBase[ConfigT, OramBlock], ABC):
    def __init__(self, config: ConfigT):
        super().__init__(config)
        self._accessing: bool = False

    def _pos_map_levels(self) -> Sequence[PosMapLevel]:
        return ()

    @property
    @override
    def max_stash(self) -> int:
        return self._max_stash + sum(level.max_stash for level in self._pos_map_levels())

    @property
    @override
    def _block_layout(self) -> Field[OramBlock]:
        return Record(
            OramBlock,
            [
                Col(lambda block: block.key, UInt.covering(self.num_data - 1)),
                Col(lambda block: block.leaf, self._leaf_field),
                Col(lambda block: block.value, Bytes(self.data_size)),
            ],
        )

    @abstractmethod
    def _initial_leaf(self, key: int) -> int: ...

    def _initial_blocks(self, data: InitData | None) -> Iterator[OramBlock]:
        supplied = bytearray(self.num_data)
        for key, value in _initial_pairs(data):
            require_oram_key(self._identity, key, self.num_data)
            require_value(self._identity, value, self.data_size)
            if supplied[key]:
                raise ContractError(f"{self._identity}: initial key {key} is given twice.")
            supplied[key] = 1
            yield OramBlock(key=key, leaf=self._initial_leaf(key), value=value)
        for key in range(self.num_data):
            if not supplied[key]:
                yield OramBlock(key=key, leaf=self._initial_leaf(key), value=b"")

    def _read_block(self, key: int, value: bytes | Unset, *, leaves: list[int], new_leaf: int) -> bytes:
        self._read_paths(leaves)
        data = self._require_in_stash(key=key)
        read_value = data.value
        if value is not UNSET:
            data.value = value
        data.leaf = new_leaf
        return read_value

    def _write_back(self) -> None:
        self._write_paths()

    def init_server_storage(self, data: InitData | None = None) -> None:
        self._host_tree(self._build_tree(self._initial_blocks(data)))
        self._host_levels()

    def _host_levels(self) -> None:
        pass

    def _require_access(self, key: int, value: bytes | Unset) -> None:
        require_oram_key(self._identity, key, self.num_data)
        if value is not UNSET:
            require_value(self._identity, value, self.data_size)

    def _access[T](self, run: Callable[[], T]) -> T:
        if self._accessing:
            raise ContractError(f"{self._identity}: an access began while another one was still running.")
        self._accessing = True
        result = run()
        self._write_back()
        self._accessing = False
        return result

    def operate_on_key(self, key: int, value: bytes | Unset = UNSET) -> bytes:
        self._require_access(key, value)
        return self._access(lambda: self._operate_on_key_without_eviction(key, value))

    def update[T](self, key: int, change: Callable[[bytes], tuple[bytes, T]]) -> T:
        require_oram_key(self._identity, key, self.num_data)

        def run() -> T:
            value, result = change(self._operate_on_key_without_eviction(key, UNSET))
            require_value(self._identity, value, self.data_size)
            self._require_in_stash(key).value = value
            return result

        return self._access(run)

    @abstractmethod
    def _operate_on_key_without_eviction(self, key: int, value: bytes | Unset) -> bytes: ...
