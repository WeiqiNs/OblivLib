from abc import ABC, abstractmethod
from dataclasses import replace
from functools import cache, cached_property
from typing import Any, Protocol, override

from oblivlib.dependency import Blake2Prf, Data, InitData
from oblivlib.dependency.config import CounterOramConfig
from oblivlib.dependency.layout import BitFields, Field, UInt
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.oram.tree_base_oram import TreeBaseOram, pos_map_level_count

GC = 0


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
        return self._prf.digest_mod_n(message, pow(2, level - 1))

    def _get_previous_leaf_from_prf(self, key: int, gc: int, ic: int) -> int:
        return self._prf_leaf(key=key, gc=gc, ic=ic, level=self._last_oram_level, num_data=self._last_oram_data)


class CounterPosMapLevel[ConfigT: CounterOramConfig](TreeStorageBase[ConfigT], CounterMixin[ConfigT]):
    def __init__(self, config: ConfigT, *, last_oram_data: int, last_oram_level: int):
        super().__init__(config)
        self._prf = Blake2Prf(key=config.prf_key)
        self._last_oram_data = last_oram_data
        self._last_oram_level = last_oram_level

    @cached_property
    @override
    def _value_field(self) -> Field[Any]:
        return UInt(self._config.count_length)

    def _build_and_host(self) -> None:
        self._host_tree(
            self._build_tree(
                Data(
                    key=key,
                    leaf=self._prf_leaf(key=key, gc=0, ic=0, level=self._level, num_data=self._num_data),
                    value=0,
                )
                for key in range(self._num_data)
            )
        )


class CounterOram[ConfigT: CounterOramConfig, LevelT: CounterPosMapLevel[Any]](
    TreeBaseOram[ConfigT, LevelT], CounterMixin[ConfigT], ABC
):
    def __init__(self, config: ConfigT):
        if config.num_data <= config.on_chip:
            raise ValueError(f"num_data ({config.num_data}) must be greater than the on-chip size ({config.on_chip}).")

        super().__init__(config)

        self._prf = Blake2Prf(key=config.prf_key)
        self._levels = self._make_levels()
        smallest = self._levels[0] if self._levels else self
        self._last_oram_data, self._last_oram_level = smallest._num_data, smallest._level
        self._on_chip_storage: list[int] = [0] * -(-self._last_oram_data // self._num_ic)

    @abstractmethod
    def _new_level(self, config: ConfigT, *, last_oram_data: int, last_oram_level: int) -> LevelT: ...

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._prf_leaf(key=key, gc=0, ic=0, level=self._level, num_data=self._num_data)

    def _make_levels(self) -> list[LevelT]:
        levels: list[LevelT] = []
        last_oram_data, last_oram_level = self._num_data, self._level
        count = pos_map_level_count(num_data=self._num_data, on_chip=self._config.on_chip, ratio=self._num_ic) - 1
        for i in range(count):
            level = self._new_level(
                replace(
                    self._config,
                    prf_key=self._prf.key,
                    num_data=-(-last_oram_data // self._num_ic),
                    name=f"{self._name}_pos_map_{count - i - 1}",
                ),
                last_oram_data=last_oram_data,
                last_oram_level=last_oram_level,
            )
            levels.append(level)
            last_oram_data, last_oram_level = level._num_data, level._level
        levels.reverse()
        return levels

    @override
    def init_server_storage(self, data: InitData | None = None) -> None:
        self._host_tree(self._build_tree(self._initial_blocks(data)))
        for level in reversed(self._levels):
            level._build_and_host()
