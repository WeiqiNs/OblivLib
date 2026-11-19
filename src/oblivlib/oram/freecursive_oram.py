import math
import secrets
from dataclasses import replace
from functools import cached_property
from typing import Any, NamedTuple, Protocol, override

from oblivlib.dependency import Blake2Prf, Data, InitData, PathRows
from oblivlib.dependency.config import FreecursiveOramConfig, ResetMethod
from oblivlib.dependency.helpers import binary_str_to_bytes, bytes_to_binary_str
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.oram.tree_base_oram import TreeBaseOram, pos_map_keys, pos_map_level_count

_CSPRNG = secrets.SystemRandom()


class ResetEntry(NamedTuple):
    key: int | None
    cur_leaf: int
    new_leaf: int | None


ResetChunk = list[ResetEntry]
ResetLeaves = list[ResetChunk]


class ProcessedData(NamedTuple):
    cur_leaf: int | None
    new_leaf: int | None
    reset_leaves: ResetLeaves | None


class UnpackedReset(NamedTuple):
    ck_a: int | None
    nl_a: int | None
    ck_b: int | None
    nl_b: int | None
    leaves: list[int]


def _unpack_reset_leaves(cur_reset_leaves: ResetChunk) -> UnpackedReset:
    ck_a, cl_a, nl_a = cur_reset_leaves[0]
    if len(cur_reset_leaves) == 2:
        ck_b, cl_b, nl_b = cur_reset_leaves[1]
        return UnpackedReset(ck_a, nl_a, ck_b, nl_b, [cl_a, cl_b])
    return UnpackedReset(ck_a, nl_a, None, None, [cl_a])


def _pass_reset_chunk(storage: TreeStorageBase[FreecursiveOramConfig], chunk: ResetChunk) -> None:
    ck_a, nl_a, ck_b, nl_b, leaves = _unpack_reset_leaves(chunk)

    storage._client.add_read_path(label=storage._name, leaves=leaves)
    storage._absorb_path(storage._client.execute().require(storage._name))
    storage._update_stash_leaf(key=ck_a, new_leaf=nl_a)
    storage._update_stash_leaf(key=ck_b, new_leaf=nl_b)

    storage._client.add_write_path(label=storage._name, data=storage._evict_stash(leaves=leaves))
    storage._client.execute()


class FreecursiveCounterMixin(Protocol):
    _config: FreecursiveOramConfig
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
    def _reset_method(self) -> ResetMethod:
        return self._config.reset_method

    @property
    def _reset_prob(self) -> float:
        return 1 / self._num_ic if self._config.reset_prob is None else self._config.reset_prob

    @cached_property
    def _count_length(self) -> int:
        return self._num_ic * self._ic_length + self._gc_length

    @property
    def _zero_counter(self) -> bytes:
        return binary_str_to_bytes("0" * self._count_length)

    def _prf_leaf(self, *, key: int, gc: int, ic: int, level: int) -> int:
        return self._prf.digest_mod_n(
            binary_str_to_bytes(
                bin(key)[2:].zfill(level - 1) + bin(gc)[2:].zfill(self._gc_length) + bin(ic)[2:].zfill(self._ic_length)
            ),
            pow(2, level - 1),
        )

    def _get_previous_leaf_from_prf(self, key: int, gc: int, ic: int) -> int:
        return self._prf_leaf(key=key, gc=gc, ic=ic, level=self._last_oram_level)

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


class FreecursivePosMapLevel(TreeStorageBase[FreecursiveOramConfig], FreecursiveCounterMixin):
    def __init__(self, config: FreecursiveOramConfig, *, last_oram_data: int, last_oram_level: int):
        super().__init__(config)
        self._prf = Blake2Prf(key=config.prf_key)
        self._last_oram_data = last_oram_data
        self._last_oram_level = last_oram_level

    def _build_and_host(self) -> None:
        zero_counter = self._zero_counter
        self._host_tree(
            self._build_tree(
                Data(key=key, leaf=self._prf_leaf(key=key, gc=0, ic=0, level=self._level), value=zero_counter)
                for key in range(self._num_data)
            )
        )

    def _update_data_prob_reset(self, key: int, data: Data, offset: int) -> ProcessedData:
        data.value = bytes_to_binary_str(data.value).zfill(self._count_length)

        gc = int(data.value[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data.value[ic_start:ic_end], 2)

        if _CSPRNG.random() <= self._reset_prob:
            reset_leaves = self._get_reset_leaves(key=key, data=data.value)
            data.value = binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            return ProcessedData(None, None, reset_leaves)

        if ic + 1 >= pow(2, self._ic_length):
            raise ValueError("Overflow happened under probabilistic resets.")
        data.value = binary_str_to_bytes(
            f"{data.value[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data.value[ic_end:]}"
        )
        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
        return ProcessedData(cur_leaf, new_leaf, None)

    def _update_data_hard_reset(self, key: int, data: Data, offset: int) -> ProcessedData:
        data.value = bytes_to_binary_str(data.value).zfill(self._count_length)

        gc = int(data.value[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data.value[ic_start:ic_end], 2)

        if ic + 1 >= pow(2, self._ic_length):
            reset_leaves = self._get_reset_leaves(key=key, data=data.value)
            data.value = binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            return ProcessedData(None, None, reset_leaves)

        data.value = binary_str_to_bytes(
            f"{data.value[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data.value[ic_end:]}"
        )
        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
        return ProcessedData(cur_leaf, new_leaf, None)

    def _retrieve_pos_map_block(self, key: int, offset: int, new_leaf: int, path: PathRows) -> ProcessedData:
        self._absorb_path(path=path)
        data = self._require_in_stash(key=key)
        processed = (
            self._update_data_prob_reset(key=key, data=data, offset=offset)
            if self._reset_method == "prob"
            else self._update_data_hard_reset(key=key, data=data, offset=offset)
        )
        data.leaf = new_leaf
        return processed

    def _access_pos_map_level(
        self, cur_key: int, cur_index: int, cur_leaf: int | None, new_leaf: int | None, reset_leaves: ResetLeaves | None
    ) -> ProcessedData:
        next_cur_leaf, next_new_leaf = None, None

        if reset_leaves is not None:
            for chunk in reset_leaves:
                ck_a, nl_a, ck_b, nl_b, leaves = _unpack_reset_leaves(chunk)
                if cur_key not in (ck_a, ck_b):
                    _pass_reset_chunk(self, chunk)
                    continue

                chunk_new_leaf, partner, partner_leaf = (nl_a, ck_b, nl_b) if ck_a == cur_key else (nl_b, ck_a, nl_a)
                assert chunk_new_leaf is not None
                self._client.add_read_path(label=self._name, leaves=leaves)
                path = self._client.execute().require(self._name)

                next_cur_leaf, next_new_leaf, reset_leaves = self._retrieve_pos_map_block(
                    key=cur_key, offset=cur_index, new_leaf=chunk_new_leaf, path=path
                )
                self._update_stash_leaf(key=partner, new_leaf=partner_leaf)

                self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=leaves))
                self._client.execute()

        else:
            assert cur_leaf is not None and new_leaf is not None
            self._client.add_read_path(label=self._name, leaves=[cur_leaf])
            result = self._client.execute()
            path = result.require(self._name)

            next_cur_leaf, next_new_leaf, reset_leaves = self._retrieve_pos_map_block(
                key=cur_key, offset=cur_index, new_leaf=new_leaf, path=path
            )

            evicted_path = self._evict_stash(leaves=[cur_leaf])

            self._client.add_write_path(label=self._name, data=evicted_path)
            self._client.execute()

        return ProcessedData(next_cur_leaf, next_new_leaf, reset_leaves)


class FreecursiveOram(TreeBaseOram[FreecursiveOramConfig, FreecursivePosMapLevel], FreecursiveCounterMixin):
    def __init__(self, config: FreecursiveOramConfig):
        if config.num_data <= config.on_chip_size:
            raise ValueError(f"num_data ({config.num_data}) must be greater than on_chip_size ({config.on_chip_size}).")

        super().__init__(config)

        self._pending_reset_chunks: ResetLeaves = []

        self._prf = Blake2Prf(key=config.prf_key)

        self._levels = self._make_levels()
        smallest = self._levels[0] if self._levels else self
        self._last_oram_data, self._last_oram_level = smallest._num_data, smallest._level
        self._on_chip_storage: list[bytes] = [self._zero_counter] * (-(-self._last_oram_data // self._num_ic) + 1)

    @property
    def _on_chip_size(self) -> int:
        return self._config.on_chip_size

    @cached_property
    def _pos_map_oram_dummy_size(self) -> int:
        return math.ceil(self._count_length / 8)

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._prf_leaf(key=key, gc=0, ic=0, level=self._level)

    def _make_levels(self) -> list[FreecursivePosMapLevel]:
        levels = []
        last_oram_data, last_oram_level = self._num_data, self._level
        count = pos_map_level_count(num_data=self._num_data, on_chip=self._on_chip_size, ratio=self._num_ic) - 1
        for i in range(count):
            level = FreecursivePosMapLevel(
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

    def _update_on_chip_data(self, key: int, offset: int) -> ProcessedData:
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
                return ProcessedData(None, None, reset_leaves)

            if ic + 1 >= pow(2, self._ic_length):
                raise ValueError("Overflow happened under probabilistic resets.")
            self._on_chip_storage[key] = binary_str_to_bytes(
                f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data[ic_end:]}"
            )
            cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
            new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
            return ProcessedData(cur_leaf, new_leaf, None)

        if ic + 1 >= pow(2, self._ic_length):
            reset_leaves = self._get_reset_leaves(key=key, data=data)
            self._on_chip_storage[key] = binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            return ProcessedData(None, None, reset_leaves)

        self._on_chip_storage[key] = binary_str_to_bytes(
            f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data[ic_end:]}"
        )
        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
        return ProcessedData(cur_leaf, new_leaf, None)

    def _get_leaf_from_pos_map(self, key: int) -> ProcessedData:
        (chip_key, chip_offset), *level_keys = pos_map_keys(key, ratio=self._num_ic, count=len(self._levels) + 1)

        cur_leaf, new_leaf, reset_leaves = self._update_on_chip_data(key=chip_key, offset=chip_offset)

        for level, (cur_key, cur_index) in zip(self._levels, level_keys, strict=True):
            cur_leaf, new_leaf, reset_leaves = level._access_pos_map_level(
                cur_key=cur_key,
                cur_index=cur_index,
                cur_leaf=cur_leaf,
                new_leaf=new_leaf,
                reset_leaves=reset_leaves,
            )

        return ProcessedData(cur_leaf, new_leaf, reset_leaves)

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        cur_leaf, next_leaf, reset_leaves = self._get_leaf_from_pos_map(key=key)

        if not reset_leaves:
            assert cur_leaf is not None and next_leaf is not None
            return self._read_block(key, value, leaves=[cur_leaf], new_leaf=next_leaf)

        for index, chunk in enumerate(reset_leaves):
            ck_a, nl_a, ck_b, nl_b, leaves = _unpack_reset_leaves(chunk)
            if key not in (ck_a, ck_b):
                _pass_reset_chunk(self, chunk)
                continue

            new_leaf, partner, partner_leaf = (nl_a, ck_b, nl_b) if ck_a == key else (nl_b, ck_a, nl_a)
            assert new_leaf is not None
            read_value = self._read_block(key, value, leaves=leaves, new_leaf=new_leaf)
            self._update_stash_leaf(key=partner, new_leaf=partner_leaf)
            self._pending_reset_chunks = reset_leaves[index + 1 :]
            return read_value

    @override
    def _write_back(self) -> None:
        super()._write_back()
        for chunk in self._pending_reset_chunks:
            _pass_reset_chunk(self, chunk)
        self._pending_reset_chunks = []
