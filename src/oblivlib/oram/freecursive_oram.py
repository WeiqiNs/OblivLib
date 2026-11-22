import secrets
from typing import NamedTuple, override

from oblivlib.dependency import Blake2Prf
from oblivlib.dependency.config import FreecursiveOramConfig
from oblivlib.dependency.types import Unset
from oblivlib.oram.counter_oram import GC, CounterOram, CounterPosMapLevel, Counters
from oblivlib.oram.tree_base_oram import pos_map_keys

_CSPRNG = secrets.SystemRandom()


class _Move(NamedTuple):
    key: int
    leaf: int


class _ResetEntry(NamedTuple):
    cur_leaf: int
    move: _Move | None


ResetChunk = list[_ResetEntry]
ResetLeaves = list[ResetChunk]


def _split_at_key(chunks: ResetLeaves, key: int) -> tuple[ResetLeaves, _Move, ResetChunk, ResetLeaves]:
    found = [
        (index, entry.move)
        for index, chunk in enumerate(chunks)
        for entry in chunk
        if entry.move is not None and entry.move.key == key
    ]
    assert found, f"key {key} is in no reset chunk"
    index, move = found[0]
    return chunks[:index], move, chunks[index], chunks[index + 1 :]


def _leaves(chunk: ResetChunk) -> list[int]:
    return [entry.cur_leaf for entry in chunk]


def _moves(chunk: ResetChunk, *, skip: int | None = None) -> list[_Move]:
    return [entry.move for entry in chunk if entry.move is not None and entry.move.key != skip]


class _FreecursiveCounters(Counters):
    def __init__(self, config: FreecursiveOramConfig, prf: Blake2Prf, *, num_data: int, leaf_range: int):
        super().__init__(config, prf, num_data=num_data, leaf_range=leaf_range)
        self._hard_reset: bool = config.reset_method == "hard"
        self._reset_prob: float = 1 / config.num_ic if config.reset_prob is None else config.reset_prob

    def advance(self, key: int, word: int, offset: int) -> tuple[int, ResetLeaves]:
        fields = self.fields
        gc = fields.get(word, GC)
        ic = fields.get(word, self.ic(offset))
        if self._hard_reset:
            reset = ic + 1 >= 1 << self.ic_length
        else:
            reset = _CSPRNG.random() <= self._reset_prob
        if reset:
            return fields.set(0, GC, gc + 1), self._reset_leaves(key, word)
        return fields.set(word, self.ic(offset), ic + 1), self._single_entry(key, offset, gc, ic)

    def _reset_leaves(self, key: int, word: int) -> ResetLeaves:
        num_ic, fields = self.num_ic, self.fields
        gc = fields.get(word, GC)
        ic = [fields.get(word, self.ic(i)) for i in range(num_ic)]

        reset_leaves = [
            _ResetEntry(
                self.leaf(key * num_ic + i, gc=gc, ic=ic[i]),
                _Move(key * num_ic + i, self.leaf(key * num_ic + i, gc=gc + 1, ic=0)),
            )
            if self.contains(key * num_ic + i)
            else _ResetEntry(self.random_leaf(), None)
            for i in range(num_ic)
        ]
        return [reset_leaves[i : i + 2] for i in range(0, len(reset_leaves), 2)]

    def _single_entry(self, key: int, offset: int, gc: int, ic: int) -> ResetLeaves:
        child = key * self.num_ic + offset
        return [
            [
                _ResetEntry(
                    self.leaf(child, gc=gc, ic=ic),
                    _Move(child, self.leaf(child, gc=gc, ic=ic + 1)),
                )
            ]
        ]


class _FreecursivePosMapLevel(CounterPosMapLevel[_FreecursiveCounters]):
    def access(self, cur_key: int, cur_index: int, chunks: ResetLeaves) -> ResetLeaves:
        before, move, chunk, after = _split_at_key(chunks, cur_key)
        for other in before:
            self._pass_paths(_leaves(other), _moves(other))

        self._read_paths(_leaves(chunk))
        data = self._require_in_stash(key=cur_key)
        data.word, next_chunks = self._counters.advance(cur_key, data.word, cur_index)
        data.leaf = move.leaf
        self._remap(_moves(chunk, skip=cur_key))
        self._write_paths()

        for other in after:
            self._pass_paths(_leaves(other), _moves(other))
        return next_chunks


class FreecursiveOram(CounterOram[FreecursiveOramConfig, _FreecursiveCounters, _FreecursivePosMapLevel]):
    def __init__(self, config: FreecursiveOramConfig):
        super().__init__(config)
        self._pending_reset_chunks: ResetLeaves = []

    @override
    def _new_counters(self, *, num_data: int, leaf_range: int) -> _FreecursiveCounters:
        return _FreecursiveCounters(self._config, self._prf, num_data=num_data, leaf_range=leaf_range)

    @override
    def _new_level(
        self, config: FreecursiveOramConfig, *, own: _FreecursiveCounters, child: _FreecursiveCounters
    ) -> _FreecursivePosMapLevel:
        return _FreecursivePosMapLevel(config, own=own, child=child)

    def _get_leaf_from_pos_map(self, key: int) -> ResetLeaves:
        counters = self._counters
        (chip_key, chip_offset), *level_keys = pos_map_keys(key, ratio=counters.num_ic, count=len(self._levels) + 1)

        self._on_chip_storage[chip_key], chunks = counters.advance(
            chip_key, self._on_chip_storage[chip_key], chip_offset
        )
        for level, (cur_key, cur_index) in zip(self._levels, level_keys, strict=True):
            chunks = level.access(cur_key=cur_key, cur_index=cur_index, chunks=chunks)
        return chunks

    @override
    def _operate_on_key_without_eviction(self, key: int, value: bytes | Unset) -> bytes:
        before, move, chunk, after = _split_at_key(self._get_leaf_from_pos_map(key=key), key)
        for other in before:
            self._pass_paths(_leaves(other), _moves(other))

        read_value = self._read_block(key, value, leaves=_leaves(chunk), new_leaf=move.leaf)
        self._remap(_moves(chunk, skip=key))
        self._pending_reset_chunks = after
        return read_value

    @override
    def _write_back(self) -> None:
        super()._write_back()
        for chunk in self._pending_reset_chunks:
            self._pass_paths(_leaves(chunk), _moves(chunk))
        self._pending_reset_chunks = []
