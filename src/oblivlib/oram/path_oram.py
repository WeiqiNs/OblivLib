from typing import Any, Never, override

from oblivlib.dependency import InitData, OramConfig, PosMap
from oblivlib.oram.tree_base_oram import TreeBaseOram


class PathOram[ConfigT: OramConfig](TreeBaseOram[ConfigT, Never]):
    def __init__(self, config: ConfigT):
        super().__init__(config)

        self._pos_map: PosMap = {i: self._get_new_leaf() for i in range(self._num_data)}

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._pos_map[key]

    @override
    def init_server_storage(self, data: InitData | None = None) -> None:
        self._host_tree(self._build_tree(self._initial_blocks(data)))

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        leaf = self._pos_map[key]
        new_leaf = self._get_new_leaf()
        self._pos_map[key] = new_leaf
        return self._read_block(key, value, leaves=[leaf], new_leaf=new_leaf)
