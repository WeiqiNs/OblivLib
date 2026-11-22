from collections.abc import Callable
from typing import override

from oblivlib.dependency import Blake2Prf, hash_data_to_leaf, hash_data_to_map
from oblivlib.dependency.balls_into_bins import max_bin_load
from oblivlib.dependency.config import TreeOmapConfig
from oblivlib.dependency.contract import require_omap_init, require_omap_key, require_omap_pair
from oblivlib.dependency.layout import Optional
from oblivlib.dependency.types import Block
from oblivlib.omap.base_omap import BaseOmap, KvList
from oblivlib.omap.ost_base_omap import OstBaseOmap, Root
from oblivlib.oram.tree_base_oram import Oram, require_oram_fits


class OramOstOmap[OmapConfigT: TreeOmapConfig, KeyT, NodeT: Block[object]](BaseOmap):
    def __init__(self, *, ost: type[OstBaseOmap[OmapConfigT, KeyT, NodeT]], ost_config: OmapConfigT, oram: Oram):
        if ost_config.cached or ost_config.batch_size > 1:
            raise ValueError(
                "OramOstOmap runs single-key ops on one tree at a time; its tree map must use cached=False and "
                + "batch_size=1."
            )
        num_data = ost_config.num_data
        self._root_layout: Optional[Root[KeyT]] = ost.root_layout(ost_config)
        require_oram_fits(
            type(self).__name__,
            oram,
            num_data=num_data,
            data_size=self._root_layout.size,
            unit="tree root",
            sizing="OramOstOmap.oram_data_size(ost, ost_config)",
        )

        self._ost: OstBaseOmap[OmapConfigT, KeyT, NodeT] = ost(
            ost_config, height_keys=min(num_data, max_bin_load(num_data))
        )
        self._oram: Oram = oram
        self._prf: Blake2Prf = Blake2Prf()

    @staticmethod
    def oram_data_size[C: TreeOmapConfig, K, N: Block[object]](ost: type[OstBaseOmap[C, K, N]], ost_config: C) -> int:
        return ost.root_layout(ost_config).size

    def _encode_root(self, root: Root[KeyT] | None) -> bytes:
        return self._root_layout.encode(root)

    def _decode_root(self, value: bytes) -> Root[KeyT] | None:
        return self._root_layout.decode(value)

    @property
    def _num_data(self) -> int:
        return self._ost.num_data

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._ost.name!r}"

    def _require_pair(self, key: bytes, value: bytes | None, *, value_required: bool) -> None:
        require_omap_pair(
            self._identity,
            key,
            value,
            key_size=self._ost.key_size,
            data_size=self._ost.data_size,
            value_required=value_required,
        )

    @override
    def init_server_storage(self, data: KvList | None = None) -> None:
        data = data or []
        require_omap_init(self._identity, data, key_size=self._ost.key_size, data_size=self._ost.data_size)
        data_map = hash_data_to_map(prf=self._prf, data=data, map_size=self._num_data)
        roots = self._ost.init_mul_tree_server_storage([data_map[key] for key in range(self._num_data)])
        self._oram.init_server_storage((key, self._encode_root(root)) for key, root in enumerate(roots))

    def _on_slot[T](self, key: bytes, op: Callable[[], T]) -> T:
        def run(slot: bytes) -> tuple[bytes, T]:
            root, result = self._ost.on_root(self._decode_root(slot), op)
            return self._encode_root(root), result

        return self._oram.update(hash_data_to_leaf(prf=self._prf, data=key, map_size=self._num_data), run)

    @override
    def search(self, key: bytes, value: bytes | None = None) -> bytes | None:
        self._require_pair(key, value, value_required=False)
        return self._on_slot(key, lambda: self._ost.search(key, value))

    @override
    def insert(self, key: bytes, value: bytes) -> None:
        self._require_pair(key, value, value_required=True)
        self._ost.require_room(1)
        self._on_slot(key, lambda: self._ost.insert(key, value))

    @override
    def delete(self, key: bytes) -> bytes | None:
        require_omap_key(self._identity, key, self._ost.key_size)
        return self._on_slot(key, lambda: self._ost.delete(key))
