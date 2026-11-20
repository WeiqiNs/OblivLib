from typing import Any, override

from oblivlib.dependency import Blake2Prf, PosMap, PseudoRandomFunction
from oblivlib.dependency.config import StaticOramConfig
from oblivlib.oram.path_oram import PathOram


class StaticOram(PathOram[StaticOramConfig]):
    def __init__(self, config: StaticOramConfig, prf: PseudoRandomFunction | None = None):
        super().__init__(config)
        self._prf = prf if prf else Blake2Prf()

    @override
    def _new_pos_map(self) -> PosMap:
        return {}

    def _get_path_number(self, key: int) -> int:
        return self._prf.digest_mod_n(str(key).encode(), self._leaf_range)

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._get_path_number(key)

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        leaf = self._get_path_number(key)
        return self._read_block(key, value, leaves=[leaf], new_leaf=leaf)
