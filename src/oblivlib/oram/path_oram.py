from typing import override

from oblivlib.dependency import OramConfig
from oblivlib.dependency.types import PosMap, Unset
from oblivlib.oram.tree_base_oram import TreeBaseOram


class PathOram[ConfigT: OramConfig](TreeBaseOram[ConfigT]):
    def __init__(self, config: ConfigT):
        super().__init__(config)

        self._pos_map: PosMap = {i: self._get_new_leaf() for i in range(self.num_data)}

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._pos_map[key]

    @override
    def _operate_on_key_without_eviction(self, key: int, value: bytes | Unset) -> bytes:
        leaf = self._pos_map[key]
        new_leaf = self._get_new_leaf()
        self._pos_map[key] = new_leaf
        return self._read_block(key, value, leaves=[leaf], new_leaf=new_leaf)
