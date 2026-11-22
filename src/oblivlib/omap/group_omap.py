import os
from collections.abc import Callable
from dataclasses import replace
from typing import Final, override

from oblivlib.dependency import Blake2Prf, hash_data_to_map
from oblivlib.dependency.balls_into_bins import max_bin_load
from oblivlib.dependency.config import GroupOmapConfig
from oblivlib.dependency.contract import require_omap_init, require_omap_key, require_omap_pair, require_room
from oblivlib.dependency.keyed_store import KeyedStore
from oblivlib.dependency.layout import FixedBytes, Pair, UInt
from oblivlib.dependency.types import KeyedBlock
from oblivlib.omap.base_omap import BaseOmap, KvList
from oblivlib.oram.tree_base_oram import Oram, require_oram_fits

_UNPLACED = 0


class _BucketStore(KeyedStore[GroupOmapConfig]):
    def open_bucket(self, leaves: list[int], width: int, in_bucket: Callable[[bytes], bool]) -> list[KeyedBlock]:
        self.read_padded(leaves, width)
        return [data for data in self._stash if in_bucket(data.key)]

    def close_bucket(self, blocks: list[KeyedBlock], kept: list[KeyedBlock]) -> None:
        old = {id(data) for data in blocks}
        dropped = old - {id(data) for data in kept}
        self._stash[:] = [data for data in self._stash if id(data) not in dropped]
        self._stash.extend(data for data in kept if id(data) not in old)
        self._check_stash()
        self.write_back()

    def add_block(self, block: KeyedBlock) -> None:
        self.read_padded([], 1)
        self._stash.append(block)
        self._check_stash()
        self.write_back()


class GroupOmap(BaseOmap):
    SEED_SIZE: Final = 32

    def __init__(self, config: GroupOmapConfig, upper_oram: Oram):
        require_oram_fits(
            type(self).__name__,
            upper_oram,
            num_data=config.num_data,
            data_size=self.upper_oram_data_size(config),
            unit="bucket",
            sizing="GroupOmap.upper_oram_data_size(config)",
        )
        self._config: GroupOmapConfig = config

        self._upper_oram: Oram = upper_oram

        self._bucket_prf: Blake2Prf = Blake2Prf()
        self._leaf_prf: Blake2Prf = Blake2Prf()

        self._upper_bound: int = max_bin_load(config.num_data)
        self._metadata: Pair[bytes, int] = self._metadata_layout(config.num_data)
        self._size: int = 0

        self._buckets: _BucketStore = _BucketStore(
            replace(config, name=f"{config.name}_lower", stash_scale=config.stash_scale * self._upper_bound),
            key_size=config.key_size,
        )

    @staticmethod
    def _metadata_layout(num_data: int) -> Pair[bytes, int]:
        return Pair(FixedBytes(GroupOmap.SEED_SIZE), UInt.covering(max_bin_load(num_data)))

    @staticmethod
    def upper_oram_data_size(config: GroupOmapConfig) -> int:
        return GroupOmap._metadata_layout(config.num_data).size

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._config.name!r}"

    @property
    def _width(self) -> int:
        return min(self._upper_bound, self._buckets.leaf_range)

    def _slot_leaf(self, seed: bytes, slot: int) -> int:
        return self._leaf_prf.digest_mod_n(message=seed + slot.to_bytes(4, "big"), mod=self._buckets.leaf_range)

    def _hash_key_to_bucket(self, key: bytes) -> int:
        return self._bucket_prf.digest_mod_n(message=key, mod=self._config.num_data)

    def _require_pair(self, key: bytes, value: bytes | None, *, value_required: bool) -> None:
        require_omap_pair(
            self._identity,
            key,
            value,
            key_size=self._config.key_size,
            data_size=self._config.data_size,
            value_required=value_required,
        )

    @override
    def init_server_storage(self, data: KvList | None = None) -> None:
        data = data or []
        require_omap_init(self._identity, data, key_size=self._config.key_size, data_size=self._config.data_size)
        require_room(self._identity, held=0, fresh=len(data), capacity=self._config.num_data)
        data_map = hash_data_to_map(prf=self._bucket_prf, data=data, map_size=self._config.num_data)

        upper_data: dict[int, bytes] = {}
        lower_blocks: list[KeyedBlock] = []
        for bucket_id in range(self._config.num_data):
            items = data_map[bucket_id]
            assert len(items) <= self._upper_bound
            seed = os.urandom(self.SEED_SIZE)
            upper_data[bucket_id] = self._metadata.encode((seed, len(items)))
            lower_blocks.extend(
                KeyedBlock(key=key, leaf=self._slot_leaf(seed, slot), value=value)
                for slot, (key, value) in enumerate(items, start=1)
            )

        self._upper_oram.init_server_storage(upper_data)
        self._buckets.host(lower_blocks)
        self._size = len(data)

    def _access_bucket[T](self, bucket_id: int, change: Callable[[list[KeyedBlock]], tuple[list[KeyedBlock], T]]) -> T:
        lower = self._buckets

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
        self._require_pair(key, value, value_required=False)

        def find(blocks: list[KeyedBlock]) -> tuple[list[KeyedBlock], bytes | None]:
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
        self._require_pair(key, value, value_required=True)
        require_room(self._identity, held=self._size, fresh=1, capacity=self._config.num_data)
        bucket_id = self._hash_key_to_bucket(key)
        if self._config.distinguishable:
            self._insert_into_next_slot(bucket_id, key, value)
        else:
            self._access_bucket(
                bucket_id, lambda blocks: ([*blocks, KeyedBlock(key=key, leaf=_UNPLACED, value=value)], None)
            )
        self._size += 1

    def _insert_into_next_slot(self, bucket_id: int, key: bytes, value: bytes) -> None:
        lower = self._buckets

        def append(metadata: bytes) -> tuple[bytes, None]:
            seed, count = self._metadata.decode(metadata)
            assert count < self._upper_bound
            lower.add_block(KeyedBlock(key=key, leaf=self._slot_leaf(seed, count + 1), value=value))
            return self._metadata.encode((seed, count + 1)), None

        self._upper_oram.update(bucket_id, append)

    @override
    def delete(self, key: bytes) -> bytes | None:
        require_omap_key(self._identity, key, self._config.key_size)

        def remove(blocks: list[KeyedBlock]) -> tuple[list[KeyedBlock], bytes | None]:
            for data in blocks:
                if data.key == key:
                    return [each for each in blocks if each is not data], data.value
            return blocks, None

        deleted = self._access_bucket(self._hash_key_to_bucket(key), remove)
        if deleted is not None:
            self._size -= 1
        return deleted

    def search_bucket(self, bucket_id: int) -> list[tuple[bytes, bytes]]:
        return self._access_bucket(bucket_id, lambda blocks: (blocks, [(data.key, data.value) for data in blocks]))
