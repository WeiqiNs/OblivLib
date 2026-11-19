import secrets
from typing import Any, NamedTuple, Protocol, override

from oblivlib.dependency import Data
from oblivlib.dependency.config import FreecursiveOramConfig, ResetMethod
from oblivlib.dependency.helpers import binary_str_to_bytes, bytes_to_binary_str
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.oram.counter_oram import CounterMixin, CounterOram, CounterPosMapLevel
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


def _remap_others(storage: TreeStorageBase[FreecursiveOramConfig], chunk: ResetChunk, key: int | None) -> None:
    for entry in chunk:
        if entry.key != key:
            storage._update_stash_leaf(key=entry.key, new_leaf=entry.new_leaf)


def _pass_reset_chunk(storage: TreeStorageBase[FreecursiveOramConfig], chunk: ResetChunk) -> None:
    leaves = [entry.cur_leaf for entry in chunk]
    storage._read_paths(leaves)
    _remap_others(storage, chunk, key=None)
    storage._write_paths(leaves)


class FreecursiveCounterMixin(CounterMixin[FreecursiveOramConfig], Protocol):
    @property
    def _reset_method(self) -> ResetMethod:
        return self._config.reset_method

    @property
    def _reset_prob(self) -> float:
        return 1 / self._num_ic if self._config.reset_prob is None else self._config.reset_prob

    def _get_reset_leaves(self, key: int, data: str) -> ResetLeaves:
        num_ic, ic_length, gc_length = self._num_ic, self._ic_length, self._gc_length
        gc = int(data[:gc_length], 2)
        ic_str = data[gc_length:]
        ic = [int(ic_str[i * ic_length : (i + 1) * ic_length], 2) for i in range(num_ic)]

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
    def _update_data_prob_reset(self, key: int, data: Data, offset: int) -> ResetLeaves:
        bits = bytes_to_binary_str(data.value).zfill(self._count_length)

        gc = int(bits[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(bits[ic_start:ic_end], 2)

        if _CSPRNG.random() <= self._reset_prob:
            reset_leaves = self._get_reset_leaves(key=key, data=bits)
            data.value = binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            return reset_leaves

        if ic + 1 >= pow(2, self._ic_length):
            raise ValueError("Overflow happened under probabilistic resets.")
        data.value = binary_str_to_bytes(f"{bits[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{bits[ic_end:]}")
        return self._single_entry(key, offset, gc, ic)

    def _update_data_hard_reset(self, key: int, data: Data, offset: int) -> ResetLeaves:
        bits = bytes_to_binary_str(data.value).zfill(self._count_length)

        gc = int(bits[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(bits[ic_start:ic_end], 2)

        if ic + 1 >= pow(2, self._ic_length):
            reset_leaves = self._get_reset_leaves(key=key, data=bits)
            data.value = binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            return reset_leaves

        data.value = binary_str_to_bytes(f"{bits[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{bits[ic_end:]}")
        return self._single_entry(key, offset, gc, ic)

    def _access_pos_map_level(self, cur_key: int, cur_index: int, chunks: ResetLeaves) -> ResetLeaves:
        before, entry, chunk, after = _split_at_key(chunks, cur_key)
        for other in before:
            _pass_reset_chunk(self, other)

        assert entry.new_leaf is not None
        leaves = [each.cur_leaf for each in chunk]
        self._read_paths(leaves)
        data = self._require_in_stash(key=cur_key)
        next_chunks = (
            self._update_data_prob_reset(key=cur_key, data=data, offset=cur_index)
            if self._reset_method == "prob"
            else self._update_data_hard_reset(key=cur_key, data=data, offset=cur_index)
        )
        data.leaf = entry.new_leaf
        _remap_others(self, chunk, key=cur_key)
        self._write_paths(leaves)

        for other in after:
            _pass_reset_chunk(self, other)
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

    def _update_on_chip_data(self, key: int, offset: int) -> ResetLeaves:
        data = bytes_to_binary_str(self._on_chip_storage[key]).zfill(self._count_length)

        gc = int(data[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data[ic_start:ic_end], 2)

        if self._reset_method == "prob":
            if _CSPRNG.random() <= self._reset_prob:
                reset_leaves = self._get_reset_leaves(key=key, data=data)
                self._on_chip_storage[key] = binary_str_to_bytes(
                    f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
                )
                return reset_leaves

            if ic + 1 >= pow(2, self._ic_length):
                raise ValueError("Overflow happened under probabilistic resets.")
            self._on_chip_storage[key] = binary_str_to_bytes(
                f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data[ic_end:]}"
            )
            return self._single_entry(key, offset, gc, ic)

        if ic + 1 >= pow(2, self._ic_length):
            reset_leaves = self._get_reset_leaves(key=key, data=data)
            self._on_chip_storage[key] = binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            return reset_leaves

        self._on_chip_storage[key] = binary_str_to_bytes(
            f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data[ic_end:]}"
        )
        return self._single_entry(key, offset, gc, ic)

    def _get_leaf_from_pos_map(self, key: int) -> ResetLeaves:
        (chip_key, chip_offset), *level_keys = pos_map_keys(key, ratio=self._num_ic, count=len(self._levels) + 1)

        chunks = self._update_on_chip_data(key=chip_key, offset=chip_offset)
        for level, (cur_key, cur_index) in zip(self._levels, level_keys, strict=True):
            chunks = level._access_pos_map_level(cur_key=cur_key, cur_index=cur_index, chunks=chunks)
        return chunks

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        before, entry, chunk, after = _split_at_key(self._get_leaf_from_pos_map(key=key), key)
        for other in before:
            _pass_reset_chunk(self, other)

        assert entry.new_leaf is not None
        read_value = self._read_block(key, value, leaves=[each.cur_leaf for each in chunk], new_leaf=entry.new_leaf)
        _remap_others(self, chunk, key=key)
        self._pending_reset_chunks = after
        return read_value

    @override
    def _write_back(self) -> None:
        super()._write_back()
        for chunk in self._pending_reset_chunks:
            _pass_reset_chunk(self, chunk)
        self._pending_reset_chunks = []
