"""Recursive Path ORAM.

The large position map is compressed into a chain of smaller position-map trees (each storing
``compression_ratio`` leaves per block); only a small top-level map is kept on the client.
Call ``init_server_storage`` once, then use ``operate_on_key``.
"""

import secrets
from dataclasses import replace
from functools import cached_property
from typing import Any, override

from oblivlib.dependency import Data, InitData, PathRows, PosMap
from oblivlib.dependency.codec import packed_size
from oblivlib.dependency.config import RecursiveOramConfig
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.oram.tree_base_oram import TreeBaseOram, pos_map_keys, pos_map_level_count


class RecursivePosMapLevel(TreeStorageBase[RecursiveOramConfig]):
    def _build_and_host(self, child_leaves: list[int]) -> list[int]:
        """Build and host this level over ``child_leaves`` (padding a short final block with random
        leaves); returns this level's own leaves, one per block."""
        leaves = [self._get_new_leaf() for _ in range(self._num_data)]
        ratio = self._config.compression_ratio
        tree = self._build_tree(
            [
                Data(
                    key=key,
                    leaf=leaf,
                    value=[
                        child_leaves[i] if i < len(child_leaves) else secrets.randbelow(self._num_data)
                        for i in range(key * ratio, (key + 1) * ratio)
                    ],
                )
                for key, leaf in enumerate(leaves)
            ]
        )
        self._host_tree(tree)
        return leaves

    def _retrieve_pos_map_block(self, key: int, offset: int, new_leaf: int, value: int, path: PathRows) -> int:
        """Pull the path into the stash; read the leaf at offset for key, overwrite it with value, remap to new_leaf."""
        self._absorb_path(path=path)
        data = self._require_in_stash(key=key)
        read_value = data.value[offset]
        data.value[offset] = value
        data.leaf = new_leaf
        return read_value

    def _access_pos_map_level(
        self, cur_key: int, cur_index: int, cur_leaf: int, new_cur_leaf: int, new_next_leaf: int
    ) -> int:
        """Drive this position-map level's own server I/O and return the leaf read at the offset.

        Reads its path on the shared client (under its own name), pulls the block into the stash
        while writing ``new_next_leaf`` at ``cur_index`` and remapping the block to ``new_cur_leaf``,
        then evicts and writes the path back.
        """
        self._client.add_read_path(label=self._name, leaves=[cur_leaf])
        result = self._client.execute()
        path_data = result.require(self._name)

        next_leaf = self._retrieve_pos_map_block(
            key=cur_key, offset=cur_index, path=path_data, new_leaf=new_cur_leaf, value=new_next_leaf
        )

        evicted_path = self._evict_stash(leaves=[cur_leaf])

        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return next_leaf


class RecursivePathOram(TreeBaseOram[RecursiveOramConfig, RecursivePosMapLevel]):
    def __init__(self, config: RecursiveOramConfig):
        if config.client is None:
            raise ValueError("Client is required for main ORAM.")
        if config.num_data <= config.on_chip_mem:
            raise ValueError(f"num_data ({config.num_data}) must be greater than on_chip_mem ({config.on_chip_mem}).")

        super().__init__(config)

        self._pos_map: PosMap = {i: self._get_new_leaf() for i in range(self._num_data)}
        self._levels = self._make_levels()

    @property
    def _on_chip_mem(self) -> int:
        return self._config.on_chip_mem

    @property
    def _compression_ratio(self) -> int:
        return self._config.compression_ratio

    @cached_property
    def _pos_map_oram_dummy_size(self) -> int:
        """Byte size bounding a position-map block's value, a list of ``compression_ratio`` leaves."""
        return packed_size([self._leaf_range - 1] * self._compression_ratio)

    def _make_levels(self) -> list[RecursivePosMapLevel]:
        """The position-map levels, smallest first; each block holds ``compression_ratio`` child leaves."""
        levels = []
        size = self._num_data
        count = pos_map_level_count(num_data=self._num_data, on_chip=self._on_chip_mem, ratio=self._compression_ratio)
        for i in range(count):
            size = -(-size // self._compression_ratio)
            levels.append(
                RecursivePosMapLevel(
                    replace(
                        self._config,
                        num_data=size,
                        name=f"{self._name}_pos_map_{count - i - 1}",
                        data_size=self._pos_map_oram_dummy_size,
                    )
                )
            )
        levels.reverse()
        return levels

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._pos_map[key]

    @override
    def init_server_storage(self, data: InitData | None = None) -> None:
        self._host_tree(self._build_tree(self._initial_blocks(data)))
        leaves = list(self._pos_map.values())
        for level in reversed(self._levels):
            leaves = level._build_and_host(leaves)
        self._pos_map = dict(enumerate(leaves))

    def _get_leaf_from_pos_map(self, key: int) -> tuple[int, int]:
        """Walk the position-map chain to find key's current leaf and assign its new leaf."""
        keys = pos_map_keys(key, ratio=self._compression_ratio, count=len(self._levels))
        top_key = keys[0][0]
        cur_leaf, new_cur_leaf = self._pos_map[top_key], self._levels[0]._get_new_leaf()
        self._pos_map[top_key] = new_cur_leaf

        children: list[TreeStorageBase[RecursiveOramConfig]] = [*self._levels[1:], self]
        for level, child, (cur_key, cur_index) in zip(self._levels, children, keys, strict=True):
            new_next_leaf = child._get_new_leaf()
            cur_leaf = level._access_pos_map_level(
                cur_key=cur_key,
                cur_index=cur_index,
                cur_leaf=cur_leaf,
                new_cur_leaf=new_cur_leaf,
                new_next_leaf=new_next_leaf,
            )
            new_cur_leaf = new_next_leaf

        return cur_leaf, new_cur_leaf

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        leaf, new_leaf = self._get_leaf_from_pos_map(key=key)
        return self._read_block(key, value, leaves=[leaf], new_leaf=new_leaf)
