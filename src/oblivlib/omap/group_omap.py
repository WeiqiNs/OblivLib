import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, override

from oblivlib.dependency import Blake2Prf, ContractError, hash_data_to_map
from oblivlib.dependency.balls_into_bins import max_bin_load
from oblivlib.dependency.config import GroupOmapConfig
from oblivlib.dependency.contract import require_omap_init, require_omap_key, require_value
from oblivlib.dependency.layout import Bytes, Col, Field, FixedBytes, Pair, Record, UInt
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.omap.base_omap import BaseOmap, KvList
from oblivlib.oram.tree_base_oram import Oram

_UNPLACED = 0


@dataclass
class _GroupBlock:
    key: bytes
    leaf: int
    value: bytes


class _BucketStore(TreeStorageBase[GroupOmapConfig, _GroupBlock]):
    @property
    @override
    def _block_layout(self) -> Field[_GroupBlock]:
        return Record(
            _GroupBlock,
            [
                Col(lambda block: block.key, Bytes(self._config.key_size)),
                Col(lambda block: block.leaf, self._leaf_field),
                Col(lambda block: block.value, Bytes(self.data_size)),
            ],
        )

    def host(self, blocks: list[_GroupBlock]) -> None:
        self._host_tree(self._build_tree(blocks))

    def open_bucket(self, leaves: list[int], width: int, in_bucket: Callable[[bytes], bool]) -> list[_GroupBlock]:
        paths = set(leaves)
        while len(paths) < width:
            paths.add(self._get_new_leaf())
        self._read_paths(sorted(paths))
        return [data for data in self._stash if in_bucket(data.key)]

    def close_bucket(self, blocks: list[_GroupBlock], kept: list[_GroupBlock]) -> None:
        old = {id(data) for data in blocks}
        dropped = old - {id(data) for data in kept}
        self._stash[:] = [data for data in self._stash if id(data) not in dropped]
        self._stash.extend(data for data in kept if id(data) not in old)
        self._check_stash()
        self._write_paths()

    def add_block(self, block: _GroupBlock) -> None:
        self._read_paths([self._get_new_leaf()])
        self._stash.append(block)
        self._check_stash()
        self._write_paths()


class GroupOmap(BaseOmap):
    SEED_SIZE: Final = 32

    def __init__(self, config: GroupOmapConfig, upper_oram: Oram):
        if upper_oram.num_data != config.num_data:
            raise ValueError(
                f"GroupOmap upper ORAM num_data={upper_oram.num_data} differs from num_data={config.num_data}; the "
                + "upper ORAM needs one slot per bucket."
            )
        required = self.upper_oram_data_size(config)
        if upper_oram.data_size < required:
            raise ValueError(
                f"GroupOmap upper ORAM data_size={upper_oram.data_size} is too small for bucket "
                + f"metadata; it must be at least {required}. Size it with GroupOmap.upper_oram_data_size(config)."
            )

        self._config: GroupOmapConfig = config

        self._upper_oram: Oram = upper_oram

        self._bucket_prf: Blake2Prf = Blake2Prf()
        self._leaf_prf: Blake2Prf = Blake2Prf()

        self._upper_bound: int = max_bin_load(config.num_data)
        self._metadata: Pair[bytes, int] = self._metadata_layout(config.num_data)
        self._size: int = 0

        self._lower_oram: _BucketStore = _BucketStore(
            replace(config, name=f"{config.name}_lower", stash_scale=config.stash_scale * self._upper_bound)
        )

    @staticmethod
    def _metadata_layout(num_data: int) -> Pair[bytes, int]:
        return Pair(FixedBytes(GroupOmap.SEED_SIZE), UInt.covering(max_bin_load(num_data)))

    @staticmethod
    def upper_oram_data_size(config: GroupOmapConfig) -> int:
        return GroupOmap._metadata_layout(config.num_data).size

    @property
    def _name(self) -> str:
        return self._config.name

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._name!r}"

    @property
    def _num_data(self) -> int:
        return self._config.num_data

    @property
    def _key_size(self) -> int:
        return self._config.key_size

    @property
    def _data_size(self) -> int:
        return self._config.data_size

    @property
    def _width(self) -> int:
        return min(self._upper_bound, self._lower_oram.leaf_range)

    def _slot_leaf(self, seed: bytes, slot: int) -> int:
        return self._leaf_prf.digest_mod_n(message=seed + slot.to_bytes(4, "big"), mod=self._lower_oram.leaf_range)

    def _hash_key_to_bucket(self, key: bytes) -> int:
        return self._bucket_prf.digest_mod_n(message=key, mod=self._num_data)

    @override
    def init_server_storage(self, data: KvList | None = None) -> None:
        data = data or []
        require_omap_init(self._identity, data, key_size=self._key_size, data_size=self._data_size)
        if len(data) > self._num_data:
            raise ContractError(f"{self._identity}: {len(data)} initial keys exceed num_data={self._num_data}.")
        data_map = hash_data_to_map(prf=self._bucket_prf, data=data, map_size=self._num_data)

        upper_data: dict[int, bytes] = {}
        lower_blocks: list[_GroupBlock] = []
        for bucket_id in range(self._num_data):
            items = data_map[bucket_id]
            assert len(items) <= self._upper_bound
            seed = os.urandom(self.SEED_SIZE)
            upper_data[bucket_id] = self._metadata.encode((seed, len(items)))
            lower_blocks.extend(
                _GroupBlock(key=key, leaf=self._slot_leaf(seed, slot), value=value)
                for slot, (key, value) in enumerate(items, start=1)
            )

        self._upper_oram.init_server_storage(upper_data)
        self._lower_oram.host(lower_blocks)
        self._size = len(data)

    def _access_bucket[T](
        self, bucket_id: int, change: Callable[[list[_GroupBlock]], tuple[list[_GroupBlock], T]]
    ) -> T:
        lower = self._lower_oram

        def rehome(metadata: bytes) -> tuple[bytes, T]:
            seed, count = self._metadata.decode(metadata)
            leaves = [self._slot_leaf(seed, slot) for slot in range(1, count + 1)]
            blocks = lower.open_bucket(leaves, self._width, lambda key: self._hash_key_to_bucket(key) == bucket_id)
            kept, result = change(blocks)
            assert len(kept) <= self._upper_bound
            new_seed = os.urandom(self.SEED_SIZE)
            for slot, data in enumerate(kept, start=1):
                data.leaf = self._slot_leaf(new_seed, slot)
            lower.close_bucket(blocks, kept)
            return self._metadata.encode((new_seed, len(kept))), result

        return self._upper_oram.update(bucket_id, rehome)

    @override
    def search(self, key: bytes, value: bytes | None = None) -> bytes | None:
        require_omap_key(self._identity, key, self._key_size)
        if value is not None:
            require_value(self._identity, value, self._data_size)

        def find(blocks: list[_GroupBlock]) -> tuple[list[_GroupBlock], bytes | None]:
            for data in blocks:
                if data.key == key:
                    found = data.value
                    if value is not None:
                        data.value = value
                    return blocks, found
            return blocks, None

        return self._access_bucket(self._hash_key_to_bucket(key), find)

    @override
    def insert(self, key: bytes, value: bytes) -> None:
        require_omap_key(self._identity, key, self._key_size)
        require_value(self._identity, value, self._data_size)
        if self._size >= self._num_data:
            raise ContractError(f"{self._identity}: the map already holds num_data={self._num_data} keys.")
        bucket_id = self._hash_key_to_bucket(key)
        if self._config.distinguishable:
            self._insert_into_next_slot(bucket_id, key, value)
        else:
            self._access_bucket(
                bucket_id, lambda blocks: ([*blocks, _GroupBlock(key=key, leaf=_UNPLACED, value=value)], None)
            )
        self._size += 1

    def _insert_into_next_slot(self, bucket_id: int, key: bytes, value: bytes) -> None:
        lower = self._lower_oram

        def append(metadata: bytes) -> tuple[bytes, None]:
            seed, count = self._metadata.decode(metadata)
            assert count < self._upper_bound
            lower.add_block(_GroupBlock(key=key, leaf=self._slot_leaf(seed, count + 1), value=value))
            return self._metadata.encode((seed, count + 1)), None

        self._upper_oram.update(bucket_id, append)

    def delete(self, key: bytes) -> bytes | None:
        require_omap_key(self._identity, key, self._key_size)

        def remove(blocks: list[_GroupBlock]) -> tuple[list[_GroupBlock], bytes | None]:
            for data in blocks:
                if data.key == key:
                    return [each for each in blocks if each is not data], data.value
            return blocks, None

        deleted = self._access_bucket(self._hash_key_to_bucket(key), remove)
        if deleted is not None:
            self._size -= 1
        return deleted

    def search_group(self, bucket_id: int) -> list[tuple[bytes, bytes]]:
        return self._access_bucket(bucket_id, lambda blocks: (blocks, [(data.key, data.value) for data in blocks]))
