import secrets
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Final, override

from oblivlib.dependency import Blake2Prf
from oblivlib.dependency.config import CounterOramConfig
from oblivlib.dependency.heap_index import compute_level
from oblivlib.dependency.layout import BitFields, Col, Field, Record, UInt
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.oram.tree_base_oram import PosMapLevel, TreeBaseOram, pos_map_level_count

GC = 0


@dataclass
class _CounterBlock:
    key: int
    leaf: int
    word: int


@cache
def _bit_fields(widths: tuple[int, ...]) -> BitFields:
    return BitFields(widths)


class Counters:
    def __init__(self, config: CounterOramConfig, prf: Blake2Prf, *, child_num_data: int, child_level: int):
        self.num_ic: int = config.num_ic
        self.ic_length: int = config.ic_length
        self.fields: BitFields = _bit_fields(config.counter_widths)
        self._gc_bytes: int = -(-config.gc_length // 8)
        self._ic_bytes: int = -(-config.ic_length // 8)
        self._prf: Blake2Prf = prf
        self._child_num_data: int = child_num_data
        self._child_level: int = child_level

    @staticmethod
    def ic(offset: int) -> int:
        return 1 + offset

    def prf_leaf(self, *, key: int, gc: int, ic: int, level: int, num_data: int) -> int:
        message = (
            num_data.to_bytes(8, "big")
            + key.to_bytes(8, "big")
            + gc.to_bytes(self._gc_bytes, "big")
            + ic.to_bytes(self._ic_bytes, "big")
        )
        return self._prf.digest_mod_n(message, 1 << (level - 1))

    def previous_leaf(self, key: int, gc: int, ic: int) -> int:
        return self.prf_leaf(key=key, gc=gc, ic=ic, level=self._child_level, num_data=self._child_num_data)

    def has_child(self, key: int) -> bool:
        return key < self._child_num_data

    def random_child_leaf(self) -> int:
        return secrets.randbelow(1 << (self._child_level - 1))


class CounterPosMapLevel[CountersT: Counters](TreeStorageBase[CounterOramConfig, _CounterBlock]):
    def __init__(self, config: CounterOramConfig, counters: CountersT):
        super().__init__(config)
        self._counters: Final[CountersT] = counters

    @property
    @override
    def _block_layout(self) -> Field[_CounterBlock]:
        return Record(
            _CounterBlock,
            [
                Col(lambda block: block.key, UInt.covering(self.num_data - 1)),
                Col(lambda block: block.leaf, self._leaf_field),
                Col(lambda block: block.word, UInt(self._config.count_length)),
            ],
        )

    def build_and_host(self) -> None:
        self._host_tree(
            self._build_tree(
                _CounterBlock(
                    key=key,
                    leaf=self._counters.prf_leaf(key=key, gc=0, ic=0, level=self._level, num_data=self.num_data),
                    word=0,
                )
                for key in range(self.num_data)
            )
        )


class CounterOram[ConfigT: CounterOramConfig, CountersT: Counters, LevelT: CounterPosMapLevel[Counters]](
    TreeBaseOram[ConfigT], ABC
):
    def __init__(self, config: ConfigT):
        if config.num_data <= config.on_chip_mem:
            raise ValueError(f"num_data ({config.num_data}) must be greater than on_chip_mem ({config.on_chip_mem}).")
        super().__init__(config)

        self._prf: Blake2Prf = Blake2Prf(key=config.prf_key)
        self._levels: list[LevelT] = []
        child_num_data, child_level = self.num_data, self._level
        count = pos_map_level_count(num_data=self.num_data, on_chip=config.on_chip_mem, ratio=config.num_ic) - 1
        for i in range(count):
            num_data = -(-child_num_data // config.num_ic)
            level_config = replace(config, num_data=num_data, name=f"{self.name}_pos_map_{count - i - 1}")
            counters = self._new_counters(child_num_data=child_num_data, child_level=child_level)
            self._levels.append(self._new_level(level_config, counters))
            child_num_data, child_level = num_data, compute_level(num_data)
        self._levels.reverse()
        self._counters: CountersT = self._new_counters(child_num_data=child_num_data, child_level=child_level)
        self._on_chip_storage: list[int] = [0] * -(-child_num_data // config.num_ic)

    @abstractmethod
    def _new_counters(self, *, child_num_data: int, child_level: int) -> CountersT: ...

    @abstractmethod
    def _new_level(self, config: ConfigT, counters: CountersT) -> LevelT: ...

    @override
    def _pos_map_levels(self) -> Sequence[PosMapLevel]:
        return self._levels

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._counters.prf_leaf(key=key, gc=0, ic=0, level=self._level, num_data=self.num_data)

    @override
    def _host_levels(self) -> None:
        for level in reversed(self._levels):
            level.build_and_host()
