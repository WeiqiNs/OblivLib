import math
import secrets
from dataclasses import replace
from functools import cached_property
from typing import Any, NamedTuple, Protocol, override

from oblivlib.dependency import Blake2Prf, Data, InitData, PathRows
from oblivlib.dependency.config import DaOramConfig
from oblivlib.dependency.helpers import binary_str_to_bytes, bytes_to_binary_str
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.oram.tree_base_oram import TreeBaseOram, pos_map_keys, pos_map_level_count


class ResetLeaf(NamedTuple):
    offset: int
    cur_leaf: int
    new_leaf: int | None


class ProcessedData(NamedTuple):
    cur_leaf: int
    new_leaf: int
    r_index: int
    r_cur_leaf: int
    r_new_leaf: int | None


class DaCounterMixin(Protocol):
    _config: DaOramConfig
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

    @cached_property
    def _count_length(self) -> int:
        return self._gc_length + (self._ic_length + 1) * self._num_ic

    @property
    def _zero_counter(self) -> bytes:
        return binary_str_to_bytes("0" * self._count_length)

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


class DaPosMapLevel(TreeStorageBase[DaOramConfig], DaCounterMixin):
    def __init__(self, config: DaOramConfig, *, last_oram_data: int, last_oram_level: int):
        super().__init__(config)
        self._prf = Blake2Prf(key=config.prf_key)
        self._last_oram_data = last_oram_data
        self._last_oram_level = last_oram_level

    def _build_and_host(self) -> None:
        zero_counter = self._zero_counter
        self._host_tree(
            self._build_tree(
                Data(
                    key=key,
                    leaf=self._prf_leaf(key=key, gc=0, ic=0, level=self._level, num_data=self._num_data),
                    value=zero_counter,
                )
                for key in range(self._num_data)
            )
        )

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

    def _retrieve_pos_map_block_with_reset(
        self, key: int, offset: int, new_leaf: int, r_key: int | None, r_new_leaf: int | None, path: PathRows
    ) -> ProcessedData:
        self._absorb_path(path=path)

        data = self._require_in_stash(key=key)
        next_cur_leaf, next_new_leaf = self._update_data(key=key, data=data, offset=offset)
        r_index, r_next_cur_leaf, r_next_new_leaf = self._perform_reset(key=key, data=data)
        data.leaf = new_leaf

        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        return ProcessedData(next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf)

    def _access_pos_map_level(
        self, cur_key: int, cur_index: int, new_leaf: int, r_key: int | None, r_new_leaf: int | None, leaves: list[int]
    ) -> ProcessedData:
        self._client.add_read_path(label=self._name, leaves=leaves)
        result = self._client.execute()
        path = result.require(self._name)

        next_cur_leaf, next_new_leaf, r_index, r_cur_leaf, r_new_leaf = self._retrieve_pos_map_block_with_reset(
            key=cur_key, r_key=r_key, offset=cur_index, new_leaf=new_leaf, r_new_leaf=r_new_leaf, path=path
        )

        evicted_path = self._evict_stash(leaves=leaves)

        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return ProcessedData(next_cur_leaf, next_new_leaf, r_index, r_cur_leaf, r_new_leaf)


class DAOram(TreeBaseOram[DaOramConfig, DaPosMapLevel], DaCounterMixin):
    def __init__(self, config: DaOramConfig):
        if config.num_data <= config.on_chip_mem:
            raise ValueError(f"num_data ({config.num_data}) must be greater than on_chip_mem ({config.on_chip_mem}).")

        super().__init__(config)

        self._prf = Blake2Prf(key=config.prf_key)

        self._levels = self._make_levels()
        smallest = self._levels[0] if self._levels else self
        self._last_oram_data, self._last_oram_level = smallest._num_data, smallest._level
        self._on_chip_storage: list[bytes] = [self._zero_counter] * (-(-self._last_oram_data // self._num_ic) + 1)

    @property
    def _on_chip_mem(self) -> int:
        return self._config.on_chip_mem

    @cached_property
    def _pos_map_oram_dummy_size(self) -> int:
        return math.ceil(self._count_length / 8)

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._prf_leaf(key=key, gc=0, ic=0, level=self._level, num_data=self._num_data)

    def _make_levels(self) -> list[DaPosMapLevel]:
        levels = []
        last_oram_data, last_oram_level = self._num_data, self._level
        count = pos_map_level_count(num_data=self._num_data, on_chip=self._on_chip_mem, ratio=self._num_ic) - 1
        for i in range(count):
            level = DaPosMapLevel(
                replace(
                    self._config,
                    prf_key=self._prf.key,
                    num_data=-(-last_oram_data // self._num_ic),
                    name=f"{self._name}_pos_map_{count - i - 1}",
                    data_size=self._pos_map_oram_dummy_size,
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

    def _get_leaf_from_pos_map(self, key: int) -> ProcessedData:
        (chip_key, chip_offset), *level_keys = pos_map_keys(key, ratio=self._num_ic, count=len(self._levels) + 1)

        cur_leaf, new_leaf = self._update_data_on_chip(key=chip_key, offset=chip_offset)

        r_index, r_cur_leaf, r_new_leaf = self._perform_reset_on_chip(key=chip_key)

        for level, (cur_key, cur_index) in zip(self._levels, level_keys, strict=True):
            r_key = None if r_index == -1 else cur_key // self._num_ic * self._num_ic + r_index

            leaves = [cur_leaf, r_cur_leaf]

            next_cur_leaf, next_new_leaf, r_index, r_cur_leaf, r_new_leaf = level._access_pos_map_level(
                cur_key=cur_key,
                cur_index=cur_index,
                new_leaf=new_leaf,
                r_key=r_key,
                r_new_leaf=r_new_leaf,
                leaves=leaves,
            )

            cur_leaf, new_leaf = next_cur_leaf, next_new_leaf

        return ProcessedData(cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf)

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = self._get_leaf_from_pos_map(key=key)

        r_key = None if r_index == -1 else key // self._num_ic * self._num_ic + r_index

        read_value = self._read_block(key, value, leaves=[cur_leaf, r_cur_leaf], new_leaf=new_leaf)

        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        return read_value
