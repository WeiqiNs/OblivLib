import secrets
from typing import Any, NamedTuple, override

from oblivlib.dependency import Data
from oblivlib.dependency.config import DaOramConfig
from oblivlib.dependency.helpers import binary_str_to_bytes, bytes_to_binary_str
from oblivlib.oram.counter_oram import CounterOram, CounterPosMapLevel
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


class DaPosMapLevel(CounterPosMapLevel[DaOramConfig]):
    def _perform_reset(self, key: int, data: Data) -> ResetLeaf:
        data.value = bytes_to_binary_str(data.value).zfill(self._count_length)

        ic_indicators = data.value[-self._num_ic :]
        offset = ic_indicators.find("1")

        data.value = binary_str_to_bytes(data.value)

        if key * self._num_ic + offset >= self._last_oram_data:
            offset = -1

        if offset != -1:
            cur_leaf, new_leaf = self._update_data(key=key, data=data, offset=offset)
        else:
            cur_leaf, new_leaf = secrets.randbelow(pow(2, self._last_oram_level - 1)), None

        return ResetLeaf(offset, cur_leaf, new_leaf)

    def _update_data(self, key: int, data: Data, offset: int) -> tuple[int, int]:
        data.value = bytes_to_binary_str(data.value).zfill(self._count_length)

        gc = int(data.value[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data.value[ic_start:ic_end], 2)

        ic_ind_start = self._gc_length + self._num_ic * self._ic_length

        if data.value[ic_ind_start + offset] == "1":
            next_ic = 0
            next_gc = gc
            gc = gc - 1
            data.value = f"{data.value[: ic_ind_start + offset]}{'0'}{data.value[ic_ind_start + offset + 1 :]}"

        elif ic + 1 >= pow(2, self._ic_length):
            next_ic = 0
            next_gc = gc + 1
            data.value = (
                f"{bin(next_gc)[2:].zfill(self._gc_length)}"
                f"{data.value[self._gc_length : -self._num_ic]}"
                f"{'1' * offset + '0' + '1' * (self._num_ic - offset - 1)}"
            )

        else:
            next_ic = ic + 1
            next_gc = gc

        data.value = binary_str_to_bytes(
            f"{data.value[:ic_start]}{bin(next_ic)[2:].zfill(self._ic_length)}{data.value[ic_end:]}"
        )

        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=next_gc, ic=next_ic)

        return cur_leaf, new_leaf

    def _access_pos_map_level(
        self, cur_key: int, cur_index: int, new_leaf: int, r_key: int | None, r_new_leaf: int | None, leaves: list[int]
    ) -> DaStep:
        self._read_paths(leaves)

        data = self._require_in_stash(key=cur_key)
        next_cur_leaf, next_new_leaf = self._update_data(key=cur_key, data=data, offset=cur_index)
        reset = self._perform_reset(key=cur_key, data=data)
        data.leaf = new_leaf
        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        self._write_paths(leaves)
        return DaStep(next_cur_leaf, next_new_leaf, *reset)


class DAOram(CounterOram[DaOramConfig, DaPosMapLevel]):
    @override
    def _new_level(self, config: DaOramConfig, *, last_oram_data: int, last_oram_level: int) -> DaPosMapLevel:
        return DaPosMapLevel(config, last_oram_data=last_oram_data, last_oram_level=last_oram_level)

    def _perform_reset_on_chip(self, key: int) -> ResetLeaf:
        data = bytes_to_binary_str(self._on_chip_storage[key]).zfill(self._count_length)

        ic_indicators = data[-self._num_ic :]
        offset = ic_indicators.find("1")

        if key * self._num_ic + offset >= self._last_oram_data:
            offset = -1

        if offset != -1:
            cur_leaf, new_leaf = self._update_data_on_chip(key=key, offset=offset)
        else:
            cur_leaf, new_leaf = secrets.randbelow(pow(2, self._last_oram_level - 1)), None

        return ResetLeaf(offset, cur_leaf, new_leaf)

    def _update_data_on_chip(self, key: int, offset: int) -> tuple[int, int]:
        data = bytes_to_binary_str(self._on_chip_storage[key]).zfill(self._count_length)

        gc = int(data[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data[ic_start:ic_end], 2)

        ic_ind_start = self._gc_length + self._num_ic * self._ic_length

        if data[ic_ind_start + offset] == "1":
            next_ic = 0
            next_gc = gc
            gc = gc - 1
            data = f"{data[: ic_ind_start + offset]}{'0'}{data[ic_ind_start + offset + 1 :]}"

        elif ic + 1 >= pow(2, self._ic_length):
            next_ic = 0
            next_gc = gc + 1
            data = (
                f"{bin(next_gc)[2:].zfill(self._gc_length)}"
                f"{data[self._gc_length : -self._num_ic]}"
                f"{'1' * offset + '0' + '1' * (self._num_ic - offset - 1)}"
            )

        else:
            next_ic = ic + 1
            next_gc = gc

        self._on_chip_storage[key] = binary_str_to_bytes(
            f"{data[:ic_start]}{bin(next_ic)[2:].zfill(self._ic_length)}{data[ic_end:]}"
        )

        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=next_gc, ic=next_ic)

        return cur_leaf, new_leaf

    def _get_leaf_from_pos_map(self, key: int) -> DaStep:
        (chip_key, chip_offset), *level_keys = pos_map_keys(key, ratio=self._num_ic, count=len(self._levels) + 1)

        cur_leaf, new_leaf = self._update_data_on_chip(key=chip_key, offset=chip_offset)
        r_index, r_cur_leaf, r_new_leaf = self._perform_reset_on_chip(key=chip_key)

        for level, (cur_key, cur_index) in zip(self._levels, level_keys, strict=True):
            r_key = None if r_index == -1 else cur_key // self._num_ic * self._num_ic + r_index
            cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = level._access_pos_map_level(
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
