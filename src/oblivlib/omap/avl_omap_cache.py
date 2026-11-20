from typing import Any, override

from oblivlib.dependency.config import AvlOmapCachedConfig
from oblivlib.omap.avl_omap import AVLOmap


class AVLOmapCached(AVLOmap):
    def __init__(self, config: AvlOmapCachedConfig):
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
    def _search(self, key: Any, value: Any = None) -> Any:
        self._op_rounds = 0
        budget = self._op_budget("search")
        if self._short_circuit_read(key=key, num_round=budget):
            return None

        self._flush_local_to_stash()
        current_key = self._descend_to_key(key=key)
        node = self._local.require(current_key)

        search_value = node.value.value if node.key == key else None
        if value is not None and node.key == key:
            node.value.value = value

        self._local.update_all_leaves(self._get_new_leaf)
        root_node = self._local.require_root()
        self.root = (root_node.key, root_node.leaf)

        self._pad_to(budget)
        return search_value

    @override
    def _insert(self, key: Any, value: Any = None) -> None:
        self._flush_local_to_stash()
        super()._insert(key, value)

    @override
    def _delete(self, key: Any) -> Any:
        self._flush_local_to_stash()
        return super()._delete(key)
