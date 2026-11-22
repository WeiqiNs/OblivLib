import secrets
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import override

from oblivlib.dependency import PosMap
from oblivlib.dependency.config import RecursiveOramConfig
from oblivlib.dependency.layout import Array, Col, Field, Record, UInt
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.dependency.types import Unset
from oblivlib.oram.tree_base_oram import PosMapLevel, TreeBaseOram, pos_map_keys, pos_map_level_count


@dataclass
class _PosMapBlock:
    key: int
    leaf: int
    leaves: list[int]


class _RecursivePosMapLevel(TreeStorageBase[RecursiveOramConfig, _PosMapBlock]):
    def __init__(self, config: RecursiveOramConfig, *, child_leaf_range: int):
        super().__init__(config)
        self._child_leaf_range: int = child_leaf_range

    @property
    @override
    def _block_layout(self) -> Field[_PosMapBlock]:
        leaves = Array(UInt.covering(self._child_leaf_range - 1), self._config.compression_ratio)
        return Record(
            _PosMapBlock,
            [
                Col(lambda block: block.key, UInt.covering(self.num_data - 1)),
                Col(lambda block: block.leaf, self._leaf_field),
                Col(lambda block: block.leaves, leaves),
            ],
        )

    def build_and_host(self, child_leaves: list[int]) -> list[int]:
        leaves = [self._get_new_leaf() for _ in range(self.num_data)]
        ratio = self._config.compression_ratio
        tree = self._build_tree(
            [
                _PosMapBlock(
                    key=key,
                    leaf=leaf,
                    leaves=[
                        child_leaves[i] if i < len(child_leaves) else secrets.randbelow(self.num_data)
                        for i in range(key * ratio, (key + 1) * ratio)
                    ],
                )
                for key, leaf in enumerate(leaves)
            ]
        )
        self._host_tree(tree)
        return leaves

    def access(self, cur_key: int, cur_index: int, cur_leaf: int, new_cur_leaf: int, new_next_leaf: int) -> int:
        self._read_paths([cur_leaf])
        data = self._require_in_stash(key=cur_key)
        next_leaf = data.leaves[cur_index]
        data.leaves[cur_index] = new_next_leaf
        data.leaf = new_cur_leaf
        self._write_paths()
        return next_leaf


class RecursivePathOram(TreeBaseOram[RecursiveOramConfig]):
    def __init__(self, config: RecursiveOramConfig):
        if config.num_data <= config.on_chip_mem:
            raise ValueError(f"num_data ({config.num_data}) must be greater than on_chip_mem ({config.on_chip_mem}).")
        super().__init__(config)

        self._pos_map: PosMap = {i: self._get_new_leaf() for i in range(self.num_data)}
        self._levels: list[_RecursivePosMapLevel] = self._make_levels()

    @override
    def _pos_map_levels(self) -> Sequence[PosMapLevel]:
        return self._levels

    @property
    def _compression_ratio(self) -> int:
        return self._config.compression_ratio

    def _make_levels(self) -> list[_RecursivePosMapLevel]:
        levels: list[_RecursivePosMapLevel] = []
        size, child_leaf_range = self.num_data, self.leaf_range
        count = pos_map_level_count(
            num_data=self.num_data, on_chip=self._config.on_chip_mem, ratio=self._compression_ratio
        )
        for i in range(count):
            size = -(-size // self._compression_ratio)
            level = _RecursivePosMapLevel(
                replace(self._config, num_data=size, name=f"{self.name}_pos_map_{count - i - 1}"),
                child_leaf_range=child_leaf_range,
            )
            levels.append(level)
            child_leaf_range = level.leaf_range
        levels.reverse()
        return levels

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._pos_map[key]

    @override
    def _host_levels(self) -> None:
        leaves = list(self._pos_map.values())
        for level in reversed(self._levels):
            leaves = level.build_and_host(leaves)
        self._pos_map = dict(enumerate(leaves))

    def _get_leaf_from_pos_map(self, key: int) -> tuple[int, int]:
        keys = pos_map_keys(key, ratio=self._compression_ratio, count=len(self._levels))
        top_key = keys[0][0]
        cur_leaf, new_cur_leaf = self._pos_map[top_key], secrets.randbelow(self._levels[0].leaf_range)
        self._pos_map[top_key] = new_cur_leaf

        child_leaf_ranges = [*(level.leaf_range for level in self._levels[1:]), self.leaf_range]
        for level, child_leaf_range, (cur_key, cur_index) in zip(self._levels, child_leaf_ranges, keys, strict=True):
            new_next_leaf = secrets.randbelow(child_leaf_range)
            cur_leaf = level.access(
                cur_key=cur_key,
                cur_index=cur_index,
                cur_leaf=cur_leaf,
                new_cur_leaf=new_cur_leaf,
                new_next_leaf=new_next_leaf,
            )
            new_cur_leaf = new_next_leaf

        return cur_leaf, new_cur_leaf

    @override
    def _operate_on_key_without_eviction(self, key: int, value: bytes | Unset) -> bytes:
        leaf, new_leaf = self._get_leaf_from_pos_map(key=key)
        return self._read_block(key, value, leaves=[leaf], new_leaf=new_leaf)
