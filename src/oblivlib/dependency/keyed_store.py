from collections.abc import Iterable
from typing import override

from oblivlib.dependency.config import OramConfig
from oblivlib.dependency.layout import Bytes, Col, Field, Record
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.dependency.types import KeyedBlock


class KeyedStore[ConfigT: OramConfig](TreeStorageBase[ConfigT, KeyedBlock]):
    def __init__(self, config: ConfigT, *, key_size: int):
        super().__init__(config)
        self._key_size: int = key_size

    @property
    @override
    def _block_layout(self) -> Field[KeyedBlock]:
        return Record(
            KeyedBlock,
            [
                Col(lambda block: block.key, Bytes(self._key_size)),
                Col(lambda block: block.leaf, self._leaf_field),
                Col(lambda block: block.value, Bytes(self.data_size)),
            ],
        )

    def host(self, blocks: list[KeyedBlock]) -> None:
        self._host_tree(self._build_tree(blocks))

    def read_padded(self, leaves: Iterable[int], width: int) -> None:
        self._read_paths(self._padded_leaves(leaves, width))

    def write_back(self) -> None:
        self._write_paths()
