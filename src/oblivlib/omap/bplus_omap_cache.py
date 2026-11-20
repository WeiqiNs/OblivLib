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

        if not leaves_to_read:
            return

        self._op_rounds += 1
        keys_to_find = set(keys_to_read)
        for bucket in self._open_paths(leaves_to_read).values():
            for data in bucket:
                if data.key in keys_to_find:
                    self._local.add(node=data)
                    keys_to_find.discard(data.key)
                else:
                    self._stash.append(data)

        self._check_stash()
        self._write_paths()

    @override
    def _search(self, key: Any, value: Any = None) -> Any:
        self._flush_local_to_stash()
        return super()._search(key=key, value=value)

    @override
    def _insert(self, key: Any, value: Any = None) -> None:
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return

        if self.root is None:
            data_block = self._get_bplus_data(keys=[key], values=[value])
            self._stash.append(data_block)
            assert data_block.leaf is not None
            self.root = (data_block.key, data_block.leaf)
            self._perform_dummy_operation(num_round=self._max_height)
            return

        self._flush_local_to_stash()
        self._find_leaf_to_local(key=key)

        num_retrieved_nodes = len(self._local)
        self._insert_into_loaded_leaf(key=key, value=value)

        self._flush_local_to_stash()
        self._perform_dummy_operation(num_round=max(0, self._max_height - num_retrieved_nodes))

    @override
    def _delete(self, key: Any) -> Any:
        self._op_rounds = 0
        budget = self._max_height
        if self._short_circuit_read(key=key, num_round=budget):
            return None
        self._flush_local_to_stash()
        return self._delete_with_siblings(key, budget)
