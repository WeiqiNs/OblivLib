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
    def __init__(self, config: CounterOramConfig, prf: Blake2Prf, *, num_data: int, leaf_range: int):
        self.num_ic: int = config.num_ic
        self.ic_length: int = config.ic_length
        self.fields: BitFields = _bit_fields(config.counter_widths)
        self._gc_bytes: int = -(-config.gc_length // 8)
        self._ic_bytes: int = -(-config.ic_length // 8)
        self._prf: Blake2Prf = prf
        self.num_data: int = num_data
        self._leaf_range: int = leaf_range

    @staticmethod
    def ic(offset: int) -> int:
        return 1 + offset

    def leaf(self, key: int, gc: int, ic: int) -> int:
        message = (
            self.num_data.to_bytes(8, "big")
            + key.to_bytes(8, "big")
            + gc.to_bytes(self._gc_bytes, "big")
            + ic.to_bytes(self._ic_bytes, "big")
        )
        return self._prf.digest_mod_n(message, self._leaf_range)

    def contains(self, key: int) -> bool:
        return key < self.num_data

    def random_leaf(self) -> int:
        return secrets.randbelow(self._leaf_range)


class CounterPosMapLevel[CountersT: Counters](TreeStorageBase[CounterOramConfig, _CounterBlock]):
    def __init__(self, config: CounterOramConfig, *, own: CountersT, child: CountersT):
        super().__init__(config)
        self._own_counters: Final[CountersT] = own
        self._counters: Final[CountersT] = child

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
                    leaf=self._own_counters.leaf(key, gc=0, ic=0),
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
        self._own_counters: CountersT = self._new_counters(num_data=self.num_data, leaf_range=self.leaf_range)
        self._levels: list[LevelT] = []
        child = self._own_counters
        count = pos_map_level_count(num_data=self.num_data, on_chip=config.on_chip_mem, ratio=config.num_ic) - 1
        for i in range(count):
            num_data = -(-child.num_data // config.num_ic)
            own = self._new_counters(num_data=num_data, leaf_range=1 << (compute_level(num_data) - 1))
            level_config = replace(config, num_data=num_data, name=f"{self.name}_pos_map_{count - i - 1}")
            self._levels.append(self._new_level(level_config, own=own, child=child))
            child = own
        self._levels.reverse()
        self._counters: CountersT = child
        self._on_chip_storage: list[int] = [0] * -(-child.num_data // config.num_ic)

    @abstractmethod
    def _new_counters(self, *, num_data: int, leaf_range: int) -> CountersT: ...

    @abstractmethod
    def _new_level(self, config: ConfigT, *, own: CountersT, child: CountersT) -> LevelT: ...

    @override
    def _pos_map_levels(self) -> Sequence[PosMapLevel]:
        return self._levels

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._own_counters.leaf(key, gc=0, ic=0)

    @override
    def _host_levels(self) -> None:
        for level in reversed(self._levels):
            level.build_and_host()
