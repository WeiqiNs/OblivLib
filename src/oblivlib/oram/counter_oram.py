from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import cache
from typing import Protocol, override

from oblivlib.dependency import Blake2Prf, InitData
from oblivlib.dependency.config import CounterOramConfig
from oblivlib.dependency.heap_index import compute_level
from oblivlib.dependency.layout import BitFields, Field, Record, UInt
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.oram.tree_base_oram import PosMapLevel, TreeBaseOram, pos_map_level_count

GC = 0


@dataclass
class _CounterBlock:
    key: int
    leaf: int
    word: int


class _CounterLevel(PosMapLevel, Protocol):
    def build_and_host(self) -> None: ...


@cache
def _bit_fields(widths: tuple[int, ...]) -> BitFields:
    return BitFields(widths)


class CounterMixin[ConfigT: CounterOramConfig](Protocol):
    _config: ConfigT
    _prf: Blake2Prf
    _last_oram_data: int
    _last_oram_level: int

    @property
    def _num_ic(self) -> int:
        return self._config.num_ic

    @property
    def _ic_length(self) -> int:
        return self._config.ic_length

    @property
    def _gc_length(self) -> int:
        return self._config.gc_length

    @property
    def _counters(self) -> BitFields:
        return _bit_fields(self._config.counter_widths)

    @staticmethod
    def _ic(offset: int) -> int:
        return 1 + offset

    def _prf_leaf(self, *, key: int, gc: int, ic: int, level: int, num_data: int) -> int:
        message = (
            num_data.to_bytes(8, "big")
            + key.to_bytes(8, "big")
            + gc.to_bytes(-(-self._gc_length // 8), "big")
            + ic.to_bytes(-(-self._ic_length // 8), "big")
        )
        return self._prf.digest_mod_n(message, 1 << (level - 1))

    def _get_previous_leaf_from_prf(self, key: int, gc: int, ic: int) -> int:
        return self._prf_leaf(key=key, gc=gc, ic=ic, level=self._last_oram_level, num_data=self._last_oram_data)


class CounterPosMapLevel[ConfigT: CounterOramConfig](TreeStorageBase[ConfigT, _CounterBlock], CounterMixin[ConfigT]):
    def __init__(self, config: ConfigT, *, last_oram_data: int, last_oram_level: int):
        super().__init__(config)
        self._prf: Blake2Prf = Blake2Prf(key=config.prf_key)
        self._last_oram_data: int = last_oram_data
        self._last_oram_level: int = last_oram_level

    @property
    @override
    def _block_layout(self) -> Field[_CounterBlock]:
        return Record(
            _CounterBlock, (UInt.covering(self._num_data - 1), self._leaf_field, UInt(self._config.count_length))
        )

    def build_and_host(self) -> None:
        self._host_tree(
            self._build_tree(
                _CounterBlock(
                    key=key,
                    leaf=self._prf_leaf(key=key, gc=0, ic=0, level=self._level, num_data=self._num_data),
                    word=0,
                )
                for key in range(self._num_data)
            )
        )


class CounterOram[ConfigT: CounterOramConfig, LevelT: _CounterLevel](
    TreeBaseOram[ConfigT, LevelT], CounterMixin[ConfigT], ABC
):
    def __init__(self, config: ConfigT):
        if config.num_data <= config.on_chip:
            raise ValueError(f"num_data ({config.num_data}) must be greater than the on-chip size ({config.on_chip}).")
        super().__init__(config)

        self._prf: Blake2Prf = Blake2Prf(key=config.prf_key)
        levels, (last_oram_data, last_oram_level) = self._make_levels()
        self._levels: list[LevelT] = levels
        self._last_oram_data: int = last_oram_data
        self._last_oram_level: int = last_oram_level
        self._on_chip_storage: list[int] = [0] * -(-self._last_oram_data // self._num_ic)

    @abstractmethod
    def _new_level(self, config: ConfigT, *, last_oram_data: int, last_oram_level: int) -> LevelT: ...

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._prf_leaf(key=key, gc=0, ic=0, level=self._level, num_data=self._num_data)

    def _make_levels(self) -> tuple[list[LevelT], tuple[int, int]]:
        levels: list[LevelT] = []
        last_oram_data, last_oram_level = self._num_data, self._level
        count = pos_map_level_count(num_data=self._num_data, on_chip=self._config.on_chip, ratio=self._num_ic) - 1
        for i in range(count):
            num_data = -(-last_oram_data // self._num_ic)
            level = self._new_level(
                replace(
                    self._config,
                    prf_key=self._prf.key,
                    num_data=num_data,
                    name=f"{self._name}_pos_map_{count - i - 1}",
                ),
                last_oram_data=last_oram_data,
                last_oram_level=last_oram_level,
            )
            levels.append(level)
            last_oram_data, last_oram_level = num_data, compute_level(num_data)
        levels.reverse()
        return levels, (last_oram_data, last_oram_level)

    @override
    def init_server_storage(self, data: InitData | None = None) -> None:
        self._host_tree(self._build_tree(self._initial_blocks(data)))
        for level in reversed(self._levels):
            level.build_and_host()
