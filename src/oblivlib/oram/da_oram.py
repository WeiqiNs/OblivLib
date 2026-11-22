import secrets
from typing import Any, NamedTuple, Protocol, override

from oblivlib.dependency.config import DaOramConfig
from oblivlib.oram.counter_oram import GC, CounterMixin, CounterOram, CounterPosMapLevel
from oblivlib.oram.tree_base_oram import pos_map_keys


class ResetLeaf(NamedTuple):
    offset: int
    cur_leaf: int
    new_leaf: int | None


class DaStep(NamedTuple):
    cur_leaf: int
    new_leaf: int
    r_index: int
    r_cur_leaf: int
    r_new_leaf: int | None


class DaCounterMixin(CounterMixin[DaOramConfig], Protocol):
    def _indicator(self, offset: int) -> int:
        return 1 + self._num_ic + offset

    def _step(self, key: int, word: int, offset: int) -> tuple[int, int, int]:
        counters = self._counters
        gc = counters.get(word, GC)
        ic = counters.get(word, self._ic(offset))

        if counters.get(word, self._indicator(offset)):
            next_ic, next_gc = 0, gc
            gc -= 1
            word = counters.set(word, self._indicator(offset), 0)
        elif ic + 1 >= 1 << self._ic_length:
            next_ic, next_gc = 0, gc + 1
            word = counters.set(word, GC, next_gc)
            for other in range(self._num_ic):
                word = counters.set(word, self._indicator(other), int(other != offset))
        else:
            next_ic, next_gc = ic + 1, gc

        word = counters.set(word, self._ic(offset), next_ic)
        child = key * self._num_ic + offset
        cur_leaf = self._get_previous_leaf_from_prf(key=child, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=child, gc=next_gc, ic=next_ic)
        return word, cur_leaf, new_leaf

    def _reset(self, key: int, word: int) -> tuple[int, ResetLeaf]:
        counters = self._counters
        pending = (i for i in range(self._num_ic) if counters.get(word, self._indicator(i)))
        offset = next(pending, -1)
        if offset != -1 and key * self._num_ic + offset < self._last_oram_data:
            word, cur_leaf, new_leaf = self._step(key, word, offset)
            return word, ResetLeaf(offset, cur_leaf, new_leaf)
        return word, ResetLeaf(-1, secrets.randbelow(pow(2, self._last_oram_level - 1)), None)


class DaPosMapLevel(CounterPosMapLevel[DaOramConfig], DaCounterMixin):
    def access(
        self, cur_key: int, cur_index: int, new_leaf: int, r_key: int | None, r_new_leaf: int | None, leaves: list[int]
    ) -> DaStep:
        self._read_paths(leaves)

        data = self._require_in_stash(key=cur_key)
        word, next_cur_leaf, next_new_leaf = self._step(cur_key, data.value, cur_index)
        word, reset = self._reset(cur_key, word)
        data.value = word
        data.leaf = new_leaf
        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        self._write_paths()
        return DaStep(next_cur_leaf, next_new_leaf, *reset)


class DAOram(CounterOram[DaOramConfig, DaPosMapLevel], DaCounterMixin):
    @override
    def _new_level(self, config: DaOramConfig, *, last_oram_data: int, last_oram_level: int) -> DaPosMapLevel:
        return DaPosMapLevel(config, last_oram_data=last_oram_data, last_oram_level=last_oram_level)

    def _get_leaf_from_pos_map(self, key: int) -> DaStep:
        (chip_key, chip_offset), *level_keys = pos_map_keys(key, ratio=self._num_ic, count=len(self._levels) + 1)

        word, cur_leaf, new_leaf = self._step(chip_key, self._on_chip_storage[chip_key], chip_offset)
        word, (r_index, r_cur_leaf, r_new_leaf) = self._reset(chip_key, word)
        self._on_chip_storage[chip_key] = word

        for level, (cur_key, cur_index) in zip(self._levels, level_keys, strict=True):
            r_key = None if r_index == -1 else cur_key // self._num_ic * self._num_ic + r_index
            cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = level.access(
                cur_key=cur_key,
                cur_index=cur_index,
                new_leaf=new_leaf,
                r_key=r_key,
                r_new_leaf=r_new_leaf,
                leaves=[cur_leaf, r_cur_leaf],
            )

        return DaStep(cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf)

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = self._get_leaf_from_pos_map(key=key)

        r_key = None if r_index == -1 else key // self._num_ic * self._num_ic + r_index

        read_value = self._read_block(key, value, leaves=[cur_leaf, r_cur_leaf], new_leaf=new_leaf)

        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        return read_value
