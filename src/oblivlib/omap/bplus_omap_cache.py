from typing import Any, override

from oblivlib.dependency.config import BPlusOmapCachedConfig
from oblivlib.omap.bplus_omap import BPlusOmap


class BPlusOmapCached(BPlusOmap):
    def __init__(self, config: BPlusOmapCachedConfig):
        super().__init__(config)

    @override
    def _move_node_to_local(self, key: Any, leaf: int | None, parent_key: Any = None) -> None:
        stash_idx = self._find_in_stash(key)
        if stash_idx >= 0:
            node = self._stash.pop(stash_idx)
            self._local.add(node=node, parent_key=parent_key)
        else:
            super()._move_node_to_local(key=key, leaf=leaf, parent_key=parent_key)

    @override
    def _fetch_nodes(self, nodes: list[tuple[Any, int]]) -> None:
        keys_to_read: list[Any] = []
        leaves_to_read: list[int] = []

        for node_key, leaf in nodes:
            stash_idx = self._find_in_stash(node_key)
            if stash_idx >= 0:
                self._local.add(node=self._stash.pop(stash_idx))
            else:
                keys_to_read.append(node_key)
                leaves_to_read.append(leaf)

        self._op_rounds += 1
        keys_to_find = set(keys_to_read)
        for bucket in self._open_paths(self._round_leaves(leaves_to_read)).values():
            for data in bucket:
                if data.key in keys_to_find:
                    self._local.add(node=data)
                    keys_to_find.discard(data.key)
                else:
                    self._stash.append(data)

        self._check_stash()
        self._write_paths()

    @property
    @override
    def _round_width(self) -> int:
        return 2

    @override
    def _op_round_bounds(self) -> dict[str, int]:
        h = self._max_height
        return {"search": h, "insert": h, "delete": h}
