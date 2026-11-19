from typing import Any, override

import msgpack

from oblivlib.dependency import Blake2Prf, hash_data_to_leaf, hash_data_to_map
from oblivlib.dependency.config import OramOstOmapConfig
from oblivlib.dependency.contract import require_omap_init, require_omap_key, require_value
from oblivlib.omap.base_omap import BaseOmap
from oblivlib.omap.ost_base_omap import ROOT, OstBaseOmap
from oblivlib.oram.tree_base_oram import TreeBaseOram


class OramOstOmap(BaseOmap):
    def __init__(self, config: OramOstOmapConfig, ost: OstBaseOmap[Any, Any], oram: TreeBaseOram[Any, Any]):
        required = self.oram_data_size(ost._key_size)
        if oram._data_size < required:
            raise ValueError(
                f"OramOstOmap ORAM data_size={oram._data_size} is too small for an ODS root pointer; it must be at "
                + f"least {required}. Size it with OramOstOmap.oram_data_size(key_size)."
            )

        self._config: OramOstOmapConfig = config
        self._ost: OstBaseOmap[Any, Any] = ost
        self._oram: TreeBaseOram[Any, Any] = oram

        self._ost.update_mul_tree_height(num_tree=self._num_data)

        self._prf: Blake2Prf = Blake2Prf()

    @staticmethod
    def oram_data_size(key_size: int) -> int:
        return len(OramOstOmap._encode_root((bytes(key_size), 2**64 - 1)))

    @staticmethod
    def _encode_root(root: ROOT | None) -> bytes:
        return b"" if root is None else msgpack.packb(list(root))

    @staticmethod
    def _decode_root(value: bytes) -> ROOT | None:
        if value == b"":
            return None
        key, leaf = msgpack.unpackb(value)
        return key, leaf

    @property
    def _num_data(self) -> int:
        return self._config.num_data

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._ost._name!r}"

    @override
    def init_server_storage(self, data: list[tuple[bytes, bytes]] | None = None) -> None:
        require_omap_init(self._identity, data or [], key_size=self._ost._key_size, data_size=self._ost._data_size)
        data_map = hash_data_to_map(prf=self._prf, data=data or [], map_size=self._num_data)
        data_list = [data_map[key] for key in range(self._num_data)]

        roots = self._ost.init_mul_tree_server_storage(data_list=data_list)
        self._oram.init_server_storage((key, self._encode_root(root)) for key, root in enumerate(roots))

    @override
    def search(self, key: bytes, value: bytes | None = None) -> bytes | None:
        require_omap_key(self._identity, key, self._ost._key_size)
        if value is not None:
            require_value(self._identity, value, self._ost._data_size)
        oram_key = hash_data_to_leaf(prf=self._prf, data=key, map_size=self._num_data)
        self._ost.root = self._decode_root(self._oram.operate_on_key_without_eviction(key=oram_key))
        old_value = self._ost._search(key, value)
        self._oram.eviction_with_update_stash(key=oram_key, value=self._encode_root(self._ost.root))
        return old_value

    @override
    def insert(self, key: bytes, value: bytes) -> None:
        require_omap_key(self._identity, key, self._ost._key_size)
        require_value(self._identity, value, self._ost._data_size)
        oram_key = hash_data_to_leaf(prf=self._prf, data=key, map_size=self._num_data)
        self._ost.root = self._decode_root(self._oram.operate_on_key_without_eviction(key=oram_key))
        self._ost._insert(key, value)
        self._oram.eviction_with_update_stash(key=oram_key, value=self._encode_root(self._ost.root))
