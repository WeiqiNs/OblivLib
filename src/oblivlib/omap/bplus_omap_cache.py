from typing import Any, override

from oblivlib.dependency.config import BPlusOmapCachedConfig
from oblivlib.omap.bplus_omap import BPlusNode, BPlusOmap


class BPlusOmapCached(BPlusOmap):
    def __init__(self, config: BPlusOmapCachedConfig):
        super().__init__(config)

    @override
    def _move_node_to_local(
        self, key: Any, leaf: int | None, parent_key: Any = None, child_index: int | None = None
    ) -> None:
        stash_idx = self._find_in_stash(key)
        if stash_idx >= 0:
            node = self._stash.pop(stash_idx)
            self._local.add(node=node, parent_key=parent_key, child_index=child_index)
        else:
            super()._move_node_to_local(key=key, leaf=leaf, parent_key=parent_key, child_index=child_index)

    def _batch_move_nodes_to_local(self, nodes_to_fetch: list[tuple[Any, int]]) -> None:
        keys_to_read: list[Any] = []
        leaves_to_read: list[int] = []

        for node_key, leaf in nodes_to_fetch:
            stash_idx = self._find_in_stash(node_key)
            if stash_idx >= 0:
                self._local.add(node=self._stash.pop(stash_idx))
            else:
                keys_to_read.append(node_key)
                leaves_to_read.append(leaf)

        if not leaves_to_read:
            return

        self._op_rounds += 1
        to_index = len(self._stash)
        self._client.add_read_path(label=self._name, leaves=leaves_to_read)
        result = self._client.execute()
        path = self._cipher.open_path(result.require(self._name))

        keys_to_find = set(keys_to_read)
        for bucket in path.values():
            for data in bucket:
                if data.key in keys_to_find:
                    self._local.add(node=data)
                    keys_to_find.discard(data.key)
                elif data.key is not None:
                    self._stash.append(data)

        self._check_stash()

        for node_key in list(keys_to_find):
            stash_idx = self._find_in_stash(node_key)
            if 0 <= stash_idx < to_index:
                self._local.add(node=self._stash.pop(stash_idx))
                keys_to_find.discard(node_key)

        if keys_to_find:
            raise KeyError(f"The search key(s) {keys_to_find} are not found.")

        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=leaves_to_read))
        self._client.execute()

    def _find_path_with_siblings_cached(
        self, key: Any
    ) -> tuple[dict[int, BPlusNode], list[int], dict[int, BPlusNode], dict[int, int]]:
        self._flush_local_to_stash()

        path_nodes: dict[int, BPlusNode] = {}
        child_indices: list[int] = []
        siblings: dict[int, BPlusNode] = {}
        sibling_indices: dict[int, int] = {}
        level = 0

        root = self.root
        assert root is not None

        self._move_node_to_local(key=root[0], leaf=root[1], parent_key=None, child_index=None)
        node = self._local.pop(self._local.root_key)
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        path_nodes[level] = node

        while not self._local.is_leaf_node(node):
            new_leaf = self._get_new_leaf()

            child_index = len(node.value.keys)
            for index, each_key in enumerate(node.value.keys):
                if key < each_key:
                    child_index = index
                    break
                elif key == each_key:
                    child_index = index + 1
                    break
            child_indices.append(child_index)
            child_key, child_leaf = node.value.values[child_index]

            sibling_index: int | None = None
            if child_index > 0:
                sibling_index = child_index - 1
            elif child_index < len(node.value.values) - 1:
                sibling_index = child_index + 1

            nodes_to_fetch: list[tuple[Any, int]] = [(child_key, child_leaf)]
            sib_key = sib_new_leaf = None
            if sibling_index is not None:
                sib_key, sib_leaf = node.value.values[sibling_index]
                sib_new_leaf = self._get_new_leaf()
                nodes_to_fetch.append((sib_key, sib_leaf))

            self._batch_move_nodes_to_local(nodes_to_fetch)

            child_node = self._local.pop(child_key)
            node.value.values[child_index] = (child_key, new_leaf)
            child_node.leaf = new_leaf

            if sibling_index is not None:
                assert sib_new_leaf is not None
                sibling_node = self._local.pop(sib_key)
                sibling_node.leaf = sib_new_leaf
                node.value.values[sibling_index] = (sib_key, sib_new_leaf)
                siblings[level + 1] = sibling_node
                sibling_indices[level + 1] = sibling_index

            level += 1
            path_nodes[level] = child_node
            node = child_node

        return path_nodes, child_indices, siblings, sibling_indices

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
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved_nodes)

    @override
    def _delete(self, key: Any) -> Any:
        self._op_rounds = 0
        budget = self._max_height

        if self._short_circuit_read(key=key, num_round=budget):
            return None

        path_nodes, child_indices, siblings, sibling_indices = self._find_path_with_siblings_cached(key=key)

        deleted_value = self._apply_delete(
            key=key,
            path_nodes=path_nodes,
            child_indices=child_indices,
            siblings=siblings,
            sibling_indices=sibling_indices,
        )
        for node in path_nodes.values():
            self._stash.append(node)
        for sib in siblings.values():
            self._stash.append(sib)

        self._pad_to(budget)

        return deleted_value
