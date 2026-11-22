from collections.abc import Callable
from typing import Any, override

from oblivlib.dependency import Blake2Prf, ContractError, hash_data_to_leaf, hash_data_to_map
from oblivlib.dependency.config import OramOstOmapConfig
from oblivlib.dependency.contract import require_omap_init, require_omap_key, require_value
from oblivlib.omap.base_omap import BaseOmap
from oblivlib.omap.ost_base_omap import ROOT, OstBaseOmap
from oblivlib.oram.tree_base_oram import TreeBaseOram


class OramOstOmap(BaseOmap):
    def __init__(self, config: OramOstOmapConfig, ost: OstBaseOmap[Any, Any], oram: TreeBaseOram[Any, Any]):
        required = self.oram_data_size(ost)
        if oram._data_size < required:
            raise ValueError(
                f"OramOstOmap ORAM data_size={oram._data_size} is too small for an ODS root pointer; it must be at "
                + f"least {required}. Size it with OramOstOmap.oram_data_size(ost)."
            )

        self._config: OramOstOmapConfig = config
        self._ost: OstBaseOmap[Any, Any] = ost
        self._oram: TreeBaseOram[Any, Any] = oram

        self._ost.update_mul_tree_height(num_tree=self._num_data)

        self._prf: Blake2Prf = Blake2Prf()

    @staticmethod
    def oram_data_size(ost: OstBaseOmap[Any, Any]) -> int:
        return ost.root_layout.size

    def _encode_root(self, root: ROOT | None) -> bytes:
        return self._ost.root_layout.encode(root)

    def _decode_root(self, value: bytes) -> ROOT | None:
        return None if value == b"" else self._ost.root_layout.decode(value)

    @property
    def _num_data(self) -> int:
        return self._config.num_data

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._ost._name!r}"

    @override
    def init_server_storage(self, data: list[tuple[bytes, bytes]] | None = None) -> None:
        require_omap_init(self._identity, data or [], key_size=self._ost._key_size, data_size=self._ost._data_size)
        if len(data or []) > self._num_data:
            raise ContractError(f"{self._identity}: {len(data or [])} initial keys exceed num_data={self._num_data}.")
        data_map = hash_data_to_map(prf=self._prf, data=data or [], map_size=self._num_data)
        data_list = [data_map[key] for key in range(self._num_data)]

        roots = self._ost.init_mul_tree_server_storage(data_list=data_list)
        self._oram.init_server_storage((key, self._encode_root(root)) for key, root in enumerate(roots))

    def _on_slot[T](self, key: bytes, op: Callable[[], T]) -> T:
        def run(slot: bytes) -> tuple[bytes, T]:
            self._ost.root = self._decode_root(slot)
            result = op()
            return self._encode_root(self._ost.root), result

        return self._oram.update(hash_data_to_leaf(prf=self._prf, data=key, map_size=self._num_data), run)

    @override
    def search(self, key: bytes, value: bytes | None = None) -> bytes | None:
        require_omap_key(self._identity, key, self._ost._key_size)
        if value is not None:
            require_value(self._identity, value, self._ost._data_size)
        return self._on_slot(key, lambda: self._ost._search(key, value))

    @override
    def insert(self, key: bytes, value: bytes) -> None:
        require_omap_key(self._identity, key, self._ost._key_size)
        require_value(self._identity, value, self._ost._data_size)
        if self._ost._size >= self._num_data:
            raise ContractError(f"{self._identity}: the map already holds num_data={self._num_data} keys.")
        self._ost._require_room(1)
        self._on_slot(key, lambda: self._ost._insert(key, value))

    def delete(self, key: bytes) -> bytes | None:
        require_omap_key(self._identity, key, self._ost._key_size)
        return self._on_slot(key, lambda: self._ost._delete(key))
