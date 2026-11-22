import secrets
from typing import Any, NamedTuple, Protocol, override

from oblivlib.dependency.config import FreecursiveOramConfig, ResetMethod
from oblivlib.oram.counter_oram import GC, CounterMixin, CounterOram, CounterPosMapLevel
from oblivlib.oram.tree_base_oram import pos_map_keys

_CSPRNG = secrets.SystemRandom()


class ResetEntry(NamedTuple):
    key: int | None
    cur_leaf: int
    new_leaf: int | None


ResetChunk = list[ResetEntry]
ResetLeaves = list[ResetChunk]


def _split_at_key(chunks: ResetLeaves, key: int) -> tuple[ResetLeaves, ResetEntry, ResetChunk, ResetLeaves]:
    for index, chunk in enumerate(chunks):
        for entry in chunk:
            if entry.key == key:
                return chunks[:index], entry, chunk, chunks[index + 1 :]
    raise KeyError(f"Key {key} is in no reset chunk.")


class FreecursiveCounterMixin(CounterMixin[FreecursiveOramConfig], Protocol):
    def _read_paths(self, leaves: list[int]) -> None: ...

    def _write_paths(self) -> None: ...

    def _update_stash_leaf(self, key: int | None, new_leaf: int | None) -> None: ...

    def _remap_others(self, chunk: ResetChunk, key: int | None) -> None:
        for entry in chunk:
            if entry.key != key:
                self._update_stash_leaf(key=entry.key, new_leaf=entry.new_leaf)

    def _pass_reset_chunk(self, chunk: ResetChunk) -> None:
        self._read_paths([entry.cur_leaf for entry in chunk])
        self._remap_others(chunk, key=None)
        self._write_paths()

    @property
    def _reset_method(self) -> ResetMethod:
        return self._config.reset_method

    @property
    def _reset_prob(self) -> float:
        return 1 / self._num_ic if self._config.reset_prob is None else self._config.reset_prob

    def _advance(self, key: int, word: int, offset: int) -> tuple[int, ResetLeaves]:
        counters = self._counters
        gc = counters.get(word, GC)
        ic = counters.get(word, self._ic(offset))
        if self._reset_method == "prob":
            reset = _CSPRNG.random() <= self._reset_prob
        else:
            reset = ic + 1 >= 1 << self._ic_length
        if reset:
            return counters.set(0, GC, gc + 1), self._get_reset_leaves(key, word)
        return counters.set(word, self._ic(offset), ic + 1), self._single_entry(key, offset, gc, ic)

    def _get_reset_leaves(self, key: int, word: int) -> ResetLeaves:
        num_ic, counters = self._num_ic, self._counters
        gc = counters.get(word, GC)
        ic = [counters.get(word, self._ic(i)) for i in range(num_ic)]

        reset_leaves = [
            ResetEntry(
                key * num_ic + i,
                self._get_previous_leaf_from_prf(key=key * num_ic + i, gc=gc, ic=ic[i]),
                self._get_previous_leaf_from_prf(key=key * num_ic + i, gc=gc + 1, ic=0),
            )
            if key * num_ic + i < self._last_oram_data
            else ResetEntry(None, secrets.randbelow(pow(2, self._last_oram_level - 1)), None)
            for i in range(num_ic)
        ]
        return [reset_leaves[i : i + 2] for i in range(0, len(reset_leaves), 2)]

    def _single_entry(self, key: int, offset: int, gc: int, ic: int) -> ResetLeaves:
        child = key * self._num_ic + offset
        return [
            [
                ResetEntry(
                    child,
                    self._get_previous_leaf_from_prf(key=child, gc=gc, ic=ic),
                    self._get_previous_leaf_from_prf(key=child, gc=gc, ic=ic + 1),
                )
            ]
        ]


class FreecursivePosMapLevel(CounterPosMapLevel[FreecursiveOramConfig], FreecursiveCounterMixin):
    def access(self, cur_key: int, cur_index: int, chunks: ResetLeaves) -> ResetLeaves:
        before, entry, chunk, after = _split_at_key(chunks, cur_key)
        for other in before:
            self._pass_reset_chunk(other)

        assert entry.new_leaf is not None
        leaves = [each.cur_leaf for each in chunk]
        self._read_paths(leaves)
        data = self._require_in_stash(key=cur_key)
        data.value, next_chunks = self._advance(cur_key, data.value, cur_index)
        data.leaf = entry.new_leaf
        self._remap_others(chunk, key=cur_key)
        self._write_paths()

        for other in after:
            self._pass_reset_chunk(other)
        return next_chunks


class FreecursiveOram(CounterOram[FreecursiveOramConfig, FreecursivePosMapLevel], FreecursiveCounterMixin):
    def __init__(self, config: FreecursiveOramConfig):
        super().__init__(config)
        self._pending_reset_chunks: ResetLeaves = []

    @override
    def _new_level(
        self, config: FreecursiveOramConfig, *, last_oram_data: int, last_oram_level: int
    ) -> FreecursivePosMapLevel:
        return FreecursivePosMapLevel(config, last_oram_data=last_oram_data, last_oram_level=last_oram_level)

    def _get_leaf_from_pos_map(self, key: int) -> ResetLeaves:
        (chip_key, chip_offset), *level_keys = pos_map_keys(key, ratio=self._num_ic, count=len(self._levels) + 1)

        self._on_chip_storage[chip_key], chunks = self._advance(chip_key, self._on_chip_storage[chip_key], chip_offset)
        for level, (cur_key, cur_index) in zip(self._levels, level_keys, strict=True):
            chunks = level.access(cur_key=cur_key, cur_index=cur_index, chunks=chunks)
        return chunks

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        before, entry, chunk, after = _split_at_key(self._get_leaf_from_pos_map(key=key), key)
        for other in before:
            self._pass_reset_chunk(other)

        assert entry.new_leaf is not None
        read_value = self._read_block(key, value, leaves=[each.cur_leaf for each in chunk], new_leaf=entry.new_leaf)
        self._remap_others(chunk, key=key)
        self._pending_reset_chunks = after
        return read_value

    @override
    def _write_back(self) -> None:
        super()._write_back()
        for chunk in self._pending_reset_chunks:
            self._pass_reset_chunk(chunk)
        self._pending_reset_chunks = []
