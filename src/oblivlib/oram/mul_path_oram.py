from dataclasses import replace

from oblivlib.dependency import UNSET, DataMap
from oblivlib.dependency.config import MulPathOramConfig
from oblivlib.oram.path_oram import PathOram


class MulPathOram(PathOram[MulPathOramConfig]):
    def __init__(self, config: MulPathOramConfig):
        super().__init__(replace(config, stash_scale=config.stash_scale * config.stash_scale_multiplier))

    def operate_on_keys(self, key_value_map: DataMap) -> DataMap:
        for key, value in key_value_map.items():
            self._require_access(key, value)
        if not key_value_map:
            return {}
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
        self._write_back()
        return read_values
