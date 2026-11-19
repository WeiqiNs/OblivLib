from dataclasses import replace
from typing import Any

from oblivlib.dependency import UNSET, Data, DataMap, PosMap
from oblivlib.dependency.config import MulPathOramConfig
from oblivlib.dependency.contract import require_oram_key, require_value
from oblivlib.oram.path_oram import PathOram


class MulPathOram(PathOram[MulPathOramConfig]):
    def __init__(self, config: MulPathOramConfig):
        super().__init__(replace(config, stash_scale=config.stash_scale * config.stash_scale_multiplier))

    def _check_batch(self, key_value_map: dict[int, Any]) -> None:
        for key, value in key_value_map.items():
            self._require_access(key, value)

    def operate_on_keys(
        self,
        key_value_map: DataMap,
        key_path_map: PosMap | None = None,
        new_path_map: PosMap | None = None,
    ) -> DataMap:
        self._check_batch(key_value_map)
        return self._operate_on_keys(key_value_map, key_path_map, new_path_map)

    def _operate_on_keys(
        self,
        key_value_map: dict[Any, Any],
        key_path_map: dict[Any, int] | None = None,
        new_path_map: dict[Any, int] | None = None,
    ) -> dict[Any, Any]:
        if not key_value_map:
            return {}
        read_values = self._operate_on_keys_without_eviction(key_value_map, key_path_map, new_path_map)
        self._write_back()
        return read_values

    def operate_on_keys_without_eviction(
        self,
        key_value_map: DataMap,
        key_path_map: PosMap | None = None,
        new_path_map: PosMap | None = None,
    ) -> DataMap:
        self._check_batch(key_value_map)
        return self._operate_on_keys_without_eviction(key_value_map, key_path_map, new_path_map)

    def _operate_on_keys_without_eviction(
        self,
        key_value_map: dict[Any, Any],
        key_path_map: dict[Any, int] | None = None,
        new_path_map: dict[Any, int] | None = None,
    ) -> dict[Any, Any]:
        if not key_value_map:
            return {}
        self._require_no_pending()

        old_leaves: list[int] = []
        key_leaf_map: dict[Any, int] = {}
        for key in key_value_map:
            old_leaves.append(key_path_map[key] if key_path_map is not None else self._pos_map[key])
            new_leaf = new_path_map[key] if new_path_map is not None and key in new_path_map else self._get_new_leaf()
            if key_path_map is None:
                self._pos_map[key] = new_leaf
            key_leaf_map[key] = new_leaf

        self._read_paths(old_leaves)
        read_values: dict[Any, Any] = {}
        for key, value in key_value_map.items():
            data = self._require_in_stash(key=key)
            read_values[key] = data.value
            if value is not UNSET:
                data.value = value
            data.leaf = key_leaf_map[key]
        self._pending_leaves = old_leaves
        return read_values

    def eviction_for_mul_keys(self, updates: DataMap | None = None) -> None:
        for key, value in (updates or {}).items():
            require_oram_key(self._identity, key, self._num_data)
            require_value(self._identity, value, self._data_size)
        self._eviction_for_mul_keys(updates)

    def _eviction_for_mul_keys(self, updates: dict[Any, Any] | None = None) -> None:
        if updates:
            for key, value in updates.items():
                for data in self._stash:
                    if data.key == key:
                        data.value = value
                        break

        self._write_back()

    def _insert_block(self, block: Data) -> None:
        self._require_no_pending()
        leaf = self._get_new_leaf()
        self._read_paths([leaf])
        self._stash.append(block)
        self._check_stash()
        self._write_paths([leaf])
