from typing import override

from oblivlib.dependency import Blake2Prf
from oblivlib.dependency.config import StaticOramConfig
from oblivlib.dependency.types import Unset
from oblivlib.oram.tree_base_oram import TreeBaseOram


class StaticOram(TreeBaseOram[StaticOramConfig]):
    def __init__(self, config: StaticOramConfig):
        super().__init__(config)
        self._prf: Blake2Prf = Blake2Prf(key=config.prf_key)

    def _get_path_number(self, key: int) -> int:
        return self._prf.digest_mod_n(str(key).encode(), self.leaf_range)

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._get_path_number(key)

    @override
    def _operate_on_key_without_eviction(self, key: int, value: bytes | Unset) -> bytes:
        leaf = self._get_path_number(key)
        return self._read_block(key, value, leaves=[leaf], new_leaf=leaf)
