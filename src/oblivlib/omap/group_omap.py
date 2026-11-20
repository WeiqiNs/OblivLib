import os
from typing import Any, override

import msgpack

from oblivlib.dependency import UNSET, Blake2Prf, Data, hash_data_to_map
from oblivlib.dependency.config import GroupOmapConfig, MulPathOramConfig
from oblivlib.dependency.contract import require_omap_init, require_omap_key, require_value
from oblivlib.dependency.load_bound import max_bucket_load
from oblivlib.omap.base_omap import BaseOmap
from oblivlib.oram import MulPathOram, TreeBaseOram

_KEY_SLOT_OVERHEAD = 4


def _encode_metadata(seed: bytes, keys: list[bytes]) -> bytes:
    return msgpack.packb([seed, keys])


def _decode_metadata(value: bytes) -> tuple[bytes, list[bytes]]:
    seed, keys = msgpack.unpackb(value)
    return seed, keys


class GroupOmap(BaseOmap):
    SEED_SIZE = 32

    def __init__(self, config: GroupOmapConfig, upper_oram: TreeBaseOram[Any, Any]):
        self._config: GroupOmapConfig = config

        self._upper_oram = upper_oram

        self._bucket_prf = Blake2Prf()
        self._leaf_prf = Blake2Prf()

        self._upper_bound = self._bucket_upper_bound(config.num_data)

        required = self.upper_oram_data_size(config.num_data, config.key_size)
        if self._upper_oram._data_size < required:
            raise ValueError(
                f"GroupOmap upper ORAM data_size={self._upper_oram._data_size} is too small for bucket "
                + f"metadata; it must be at least {required}. Size it with "
                + "GroupOmap.upper_oram_data_size(num_data, key_size)."
            )

        self._lower_oram = MulPathOram(
            MulPathOramConfig(
                num_data=config.num_data,
                data_size=config.data_size + config.key_size + _KEY_SLOT_OVERHEAD,
                client=config.client,
                name=f"{config.name}_lower",
                bucket_size=config.bucket_size,
                stash_scale=config.stash_scale,
                encryptor=config.encryptor,
                stash_scale_multiplier=self._upper_bound,
            )
        )

    @staticmethod
    def _bucket_upper_bound(num_data: int) -> int:
        return max_bucket_load(num_data)

    @staticmethod
    def upper_oram_data_size(num_data: int, key_size: int) -> int:
        upper_bound = GroupOmap._bucket_upper_bound(num_data)
        return len(_encode_metadata(os.urandom(GroupOmap.SEED_SIZE), [os.urandom(key_size)] * upper_bound))

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

    def _compute_path(self, seed: bytes, key: bytes) -> int:
        return self._leaf_prf.digest_mod_n(message=seed + key, mod=self._lower_oram._leaf_range)

    def _hash_key_to_bucket(self, key: bytes) -> int:
        return self._bucket_prf.digest_mod_n(message=key, mod=self._num_data)

    @override
    def init_server_storage(self, data: list[tuple[bytes, bytes]] | None = None) -> None:
        require_omap_init(self._identity, data or [], key_size=self._key_size, data_size=self._data_size)
        data_map = hash_data_to_map(prf=self._bucket_prf, data=data or [], map_size=self._num_data)

        upper_data: dict[int, bytes] = {}
        lower_blocks: list[Data] = []
        for bucket_id in range(self._num_data):
            bucket_items = data_map.get(bucket_id, [])
            if len(bucket_items) > self._upper_bound:
                raise MemoryError(
                    f"Bucket {bucket_id} has {len(bucket_items)} items, exceeds upper bound {self._upper_bound}"
                )

            seed = os.urandom(self.SEED_SIZE)
            upper_data[bucket_id] = _encode_metadata(seed, [key for key, _ in bucket_items])
            lower_blocks.extend(
                Data(key=key, leaf=self._compute_path(seed, key), value=value) for key, value in bucket_items
            )

        self._upper_oram.init_server_storage(upper_data)
        self._lower_oram._host_tree(self._lower_oram._build_tree(lower_blocks))

    def _read_bucket(self, seed: bytes, new_seed: bytes, keys: list[bytes]) -> dict[Any, Any]:
        return self._lower_oram._operate_on_keys_without_eviction(
            key_value_map=dict.fromkeys(keys, UNSET),
            key_path_map={key: self._compute_path(seed, key) for key in keys},
            new_path_map={key: self._compute_path(new_seed, key) for key in keys},
        )

    def _access_bucket(
        self, bucket_id: int, update: tuple[bytes, bytes] | None = None
    ) -> tuple[list[bytes], dict[Any, Any]]:
        metadata = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        seed, bucket_keys = _decode_metadata(metadata)
        try:
            new_seed = os.urandom(self.SEED_SIZE)
            values = self._read_bucket(seed, new_seed, bucket_keys)
            metadata = _encode_metadata(new_seed, bucket_keys)
            if bucket_keys:
                updates = dict([update]) if update is not None and update[0] in values else None
                self._lower_oram._eviction_for_mul_keys(updates=updates)
        finally:
            self._upper_oram.eviction_with_update_stash(key=bucket_id, value=metadata)
        return bucket_keys, values

    @override
    def search(self, key: bytes, value: bytes | None = None) -> bytes | None:
        require_omap_key(self._identity, key, self._key_size)
        if value is not None:
            require_value(self._identity, value, self._data_size)

        _, values = self._access_bucket(self._hash_key_to_bucket(key), None if value is None else (key, value))
        return values.get(key)

    @override
    def insert(self, key: bytes, value: bytes) -> None:
        require_omap_key(self._identity, key, self._key_size)
        require_value(self._identity, value, self._data_size)

        bucket_id = self._hash_key_to_bucket(key)
        metadata = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        seed, bucket_keys = _decode_metadata(metadata)
        try:
            if len(bucket_keys) >= self._upper_bound:
                raise MemoryError(f"Bucket {bucket_id} is full ({len(bucket_keys)} items), cannot insert")
            block = Data(key=key, leaf=self._compute_path(seed, key), value=value)
            try:
                self._lower_oram._insert_block(block)
            except BaseException:
                if any(data is block for data in self._lower_oram._stash):
                    metadata = _encode_metadata(seed, [*bucket_keys, key])
                raise
            metadata = _encode_metadata(seed, [*bucket_keys, key])
        finally:
            self._upper_oram.eviction_with_update_stash(key=bucket_id, value=metadata)

    def search_group(self, bucket_id: int) -> list[tuple[bytes, bytes]]:
        bucket_keys, values = self._access_bucket(bucket_id)
        return [(key, values[key]) for key in bucket_keys]
