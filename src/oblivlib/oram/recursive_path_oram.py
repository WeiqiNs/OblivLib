import secrets
from dataclasses import replace
from functools import cached_property
from typing import Any, override

from oblivlib.dependency import Data, InitData, PosMap
from oblivlib.dependency.config import RecursiveOramConfig
from oblivlib.dependency.layout import Array, Field, UInt
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.oram.tree_base_oram import TreeBaseOram, pos_map_keys, pos_map_level_count


class RecursivePosMapLevel(TreeStorageBase[RecursiveOramConfig]):
    def __init__(self, config: RecursiveOramConfig, *, child_leaf_range: int):
        super().__init__(config)
        self._child_leaf_range = child_leaf_range

    @cached_property
    @override
    def _value_field(self) -> Field[Any]:
        return Array(UInt.covering(self._child_leaf_range - 1), self._config.compression_ratio)

    def _build_and_host(self, child_leaves: list[int]) -> list[int]:
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

    def _access_pos_map_level(
        self, cur_key: int, cur_index: int, cur_leaf: int, new_cur_leaf: int, new_next_leaf: int
    ) -> int:
        self._read_paths([cur_leaf])
        data = self._require_in_stash(key=cur_key)
        next_leaf = data.value[cur_index]
        data.value[cur_index] = new_next_leaf
        data.leaf = new_cur_leaf
        self._write_paths()
        return next_leaf


class RecursivePathOram(TreeBaseOram[RecursiveOramConfig, RecursivePosMapLevel]):
    def __init__(self, config: RecursiveOramConfig):
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

    def _make_levels(self) -> list[RecursivePosMapLevel]:
        levels: list[RecursivePosMapLevel] = []
        size, child_leaf_range = self._num_data, self._leaf_range
        count = pos_map_level_count(num_data=self._num_data, on_chip=self._on_chip_mem, ratio=self._compression_ratio)
        for i in range(count):
            size = -(-size // self._compression_ratio)
            level = RecursivePosMapLevel(
                replace(self._config, num_data=size, name=f"{self._name}_pos_map_{count - i - 1}"),
                child_leaf_range=child_leaf_range,
            )
            levels.append(level)
            child_leaf_range = level._leaf_range
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
