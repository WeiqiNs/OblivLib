from typing import NamedTuple, override

from oblivlib.dependency.config import DaOramConfig
from oblivlib.dependency.types import Unset
from oblivlib.oram.counter_oram import GC, CounterOram, CounterPosMapLevel, Counters
from oblivlib.oram.tree_base_oram import pos_map_keys


class _Reset(NamedTuple):
    cur_leaf: int
    pending: tuple[int, int] | None


class _DaStep(NamedTuple):
    cur_leaf: int
    new_leaf: int
    reset: _Reset


class _DaCounters(Counters):
    def _indicator(self, offset: int) -> int:
        return 1 + self.num_ic + offset

    def step(self, key: int, word: int, offset: int) -> tuple[int, int, int]:
        fields = self.fields
        gc = fields.get(word, GC)
        ic = fields.get(word, self.ic(offset))

        if fields.get(word, self._indicator(offset)):
            next_ic, next_gc = 0, gc
            gc -= 1
            word = fields.set(word, self._indicator(offset), 0)
        elif ic + 1 >= 1 << self.ic_length:
            next_ic, next_gc = 0, gc + 1
            word = fields.set(word, GC, next_gc)
            for other in range(self.num_ic):
                word = fields.set(word, self._indicator(other), int(other != offset))
        else:
            next_ic, next_gc = ic + 1, gc

        word = fields.set(word, self.ic(offset), next_ic)
        child = key * self.num_ic + offset
        cur_leaf = self.previous_leaf(key=child, gc=gc, ic=ic)
        new_leaf = self.previous_leaf(key=child, gc=next_gc, ic=next_ic)
        return word, cur_leaf, new_leaf

    def reset(self, key: int, word: int) -> tuple[int, _Reset]:
        fields = self.fields
        offset = next((i for i in range(self.num_ic) if fields.get(word, self._indicator(i))), None)
        if offset is not None and self.has_child(key * self.num_ic + offset):
            word, cur_leaf, new_leaf = self.step(key, word, offset)
            return word, _Reset(cur_leaf, (offset, new_leaf))
        return word, _Reset(self.random_child_leaf(), None)

    def moves(self, key: int, reset: _Reset) -> list[tuple[int, int]]:
        if reset.pending is None:
            return []
        offset, new_leaf = reset.pending
        return [(key // self.num_ic * self.num_ic + offset, new_leaf)]


class _DaPosMapLevel(CounterPosMapLevel[_DaCounters]):
    def access(self, cur_key: int, cur_index: int, new_leaf: int, reset: _Reset, leaves: list[int]) -> _DaStep:
        self._read_paths(leaves)

        data = self._require_in_stash(key=cur_key)
        word, next_cur_leaf, next_new_leaf = self._counters.step(cur_key, data.word, cur_index)
        word, next_reset = self._counters.reset(cur_key, word)
        data.word = word
        data.leaf = new_leaf
        self._remap(self._counters.moves(cur_key, reset))

        self._write_paths()
        return _DaStep(next_cur_leaf, next_new_leaf, next_reset)


class DAOram(CounterOram[DaOramConfig, _DaCounters, _DaPosMapLevel]):
    @override
    def _new_counters(self, *, child_num_data: int, child_level: int) -> _DaCounters:
        return _DaCounters(self._config, self._prf, child_num_data=child_num_data, child_level=child_level)

    @override
    def _new_level(self, config: DaOramConfig, counters: _DaCounters) -> _DaPosMapLevel:
        return _DaPosMapLevel(config, counters)

    def _get_leaf_from_pos_map(self, key: int) -> _DaStep:
        counters = self._counters
        (chip_key, chip_offset), *level_keys = pos_map_keys(key, ratio=counters.num_ic, count=len(self._levels) + 1)

        word, cur_leaf, new_leaf = counters.step(chip_key, self._on_chip_storage[chip_key], chip_offset)
        word, reset = counters.reset(chip_key, word)
        self._on_chip_storage[chip_key] = word

        for level, (cur_key, cur_index) in zip(self._levels, level_keys, strict=True):
            cur_leaf, new_leaf, reset = level.access(
                cur_key=cur_key,
                cur_index=cur_index,
                new_leaf=new_leaf,
                reset=reset,
                leaves=[cur_leaf, reset.cur_leaf],
            )

        return _DaStep(cur_leaf, new_leaf, reset)

    @override
    def _operate_on_key_without_eviction(self, key: int, value: bytes | Unset) -> bytes:
        cur_leaf, new_leaf, reset = self._get_leaf_from_pos_map(key=key)
        read_value = self._read_block(key, value, leaves=[cur_leaf, reset.cur_leaf], new_leaf=new_leaf)
        self._remap(self._counters.moves(key, reset))
        return read_value
