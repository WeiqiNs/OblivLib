from collections.abc import Mapping

from oblivlib.dependency.config import MulPathOramConfig
from oblivlib.dependency.types import UNSET, DataMap, Unset
from oblivlib.oram.path_oram import PathOram


class MulPathOram(PathOram[MulPathOramConfig]):
    def operate_on_keys(self, key_value_map: Mapping[int, bytes | Unset]) -> DataMap:
        for key, value in key_value_map.items():
            self._require_access(key, value)
        if not key_value_map:
            return {}
        return self._access(lambda: self._read_many(key_value_map))

    def _read_many(self, key_value_map: Mapping[int, bytes | Unset]) -> DataMap:
        old_leaves = [self._pos_map[key] for key in key_value_map]
        new_leaves = {key: self._get_new_leaf() for key in key_value_map}
        self._pos_map.update(new_leaves)
        self._read_paths(old_leaves)
        read_values: DataMap = {}
        for key, value in key_value_map.items():
            data = self._require_in_stash(key=key)
            read_values[key] = data.value
            if value is not UNSET:
                data.value = value
            data.leaf = new_leaves[key]
        return read_values
