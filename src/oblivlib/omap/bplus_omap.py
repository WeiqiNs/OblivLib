from bisect import bisect_right
from collections.abc import Callable
from functools import cached_property
from typing import Any, Protocol, cast, override

from oblivlib.dependency import BPlusData, BPlusTree, BPlusTreeNode, Data, KVPair
from oblivlib.dependency.config import BPlusOmapConfig
from oblivlib.dependency.layout import Bytes, Field, Record, Seq, UInt, Variant
from oblivlib.dependency.load_bound import max_bucket_load
from oblivlib.omap.ost_base_omap import ROOT, LocalNodesBase, OstBaseOmap

_BLOCK_ID_BITS = 64


def bplus_max_levels(num_keys: int, order: int) -> int:
    fanout = -(-order // 2)
    levels, split_keys, subtree_keys = 1, order, fanout - 1
    while split_keys <= num_keys:
        levels += 1
        split_keys += (order - 1) * subtree_keys
        subtree_keys *= fanout
    return levels


class BPlusValue(Protocol):
    keys: list[Any]
    values: list[Any]


class BPlusNode(Protocol):
    key: Any
    leaf: int
    value: BPlusValue


class LocalNodes(LocalNodesBase[BPlusNode]):
    @staticmethod
    def is_leaf_node(node: BPlusNode) -> bool:
        return LocalNodes.is_leaf_node_value(node.value)

    @staticmethod
    def is_leaf_node_value(value: BPlusValue) -> bool:
        return len(value.keys) == len(value.values)

    def require_leaf(self) -> BPlusNode:
        node = cast(BPlusNode | None, self.nodes.get(self.path[-1])) if self.path else None
        if node is None:
            raise KeyError("The local has no leaf node.")
        return node


class BPlusOmap(OstBaseOmap[BPlusOmapConfig, LocalNodes]):
    def __init__(self, config: BPlusOmapConfig):
        super().__init__(config)

        self._mid: int = config.order // 2
        self._block_id: int = 0

        self._max_height: int = bplus_max_levels(self._num_data, config.order)

    @override
    def _new_local(self) -> LocalNodes:
        return LocalNodes()

    @override
    def update_mul_tree_height(self, num_tree: int) -> None:
        tree_size = max_bucket_load(num_tree)
        self._max_height = bplus_max_levels(tree_size, self._order)

    @property
    def _order(self) -> int:
        return self._config.order

    @cached_property
    @override
    def _key_field(self) -> Field[Any]:
        return UInt(_BLOCK_ID_BITS)

    @cached_property
    @override
    def _value_field(self) -> Field[Any]:
        keys = Seq(Bytes(self._key_size), self._order - 1)
        leaf_node = Record((keys, Seq(Bytes(self._data_size), self._order - 1)), BPlusData)
        child = Record((self._key_field, self._leaf_field))
        internal_node = Record((keys, Seq(child, self._order)), BPlusData)
        return Variant((leaf_node, internal_node), lambda node: 0 if LocalNodes.is_leaf_node_value(node) else 1)

    def _get_bplus_data(self, keys: Any = None, values: Any = None) -> Data:
        data_block = Data(key=self._block_id, leaf=self._get_new_leaf(), value=BPlusData(keys=keys, values=values))
        self._block_id += 1
        return data_block

    @override
    def _build_ods_blocks(self, data: list[KVPair]) -> tuple[list[Data], ROOT]:
        root = BPlusTreeNode()
        bplus_tree = BPlusTree(order=self._order, leaf_range=self._leaf_range)
        for kv_pair in data:
            root = bplus_tree.insert(root=root, kv_pair=kv_pair)

        blocks = bplus_tree.get_data_list(root=root, block_id=self._block_id)
        self._block_id += len(blocks)
        assert root.id is not None and root.leaf is not None
        return blocks, (root.id, root.leaf)

    def _descend(self, key: Any, hold: Callable[[BPlusNode], bool] | None = None) -> BPlusNode:
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")
        root = self.root
        assert root is not None

        self._fetch_node(key=root[0], leaf=root[1])
        node = self._local.require_root()
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)

        while not self._local.is_leaf_node(node):
            child_index = bisect_right(node.value.keys, key)
            child_key, child_leaf = node.value.values[child_index]
            new_leaf = self._get_new_leaf()
            node.value.values[child_index] = (child_key, new_leaf)
            if hold is None:
                self._return_local([node.key])
            self._fetch_node(key=child_key, leaf=child_leaf, parent_key=node.key)
            child = self._local.require(child_key)
            child.leaf = new_leaf
            if hold is not None and not hold(child):
                self._return_local(key for key in self._local.nodes if key != child_key)
            node = child
        return node

    def _split_node(self, node: BPlusNode) -> tuple[int, int]:
        right_node = self._get_bplus_data()

        if self._local.is_leaf_node(node):
            right_node.value.keys = node.value.keys[self._mid :]
            right_node.value.values = node.value.values[self._mid :]
            node.value.keys = node.value.keys[: self._mid]
            node.value.values = node.value.values[: self._mid]
        else:
            right_node.value.keys = node.value.keys[self._mid + 1 :]
            right_node.value.values = node.value.values[self._mid + 1 :]
            node.value.keys = node.value.keys[: self._mid]
            node.value.values = node.value.values[: self._mid + 1]

        self._return_to_stash([right_node])
        return right_node.key, right_node.require_leaf()

    def _insert_in_parent(self, child_node: BPlusNode, parent_node: BPlusNode) -> None:
        insert_key = child_node.value.keys[self._mid]
        right_node = self._split_node(node=child_node)

        index = bisect_right(parent_node.value.keys, insert_key)
        parent_node.value.keys.insert(index, insert_key)
        parent_node.value.values.insert(index + 1, right_node)

    def _create_parent(self, child_node: BPlusNode) -> None:
        insert_key = child_node.value.keys[self._mid]
        right_node = self._split_node(node=child_node)

        values = [(child_node.key, child_node.leaf), right_node]
        parent_node = self._get_bplus_data(keys=[insert_key], values=values)
        self._return_to_stash([parent_node])

        assert parent_node.leaf is not None
        self.root = (parent_node.key, parent_node.leaf)

    def _insert_into_loaded_leaf(self, key: Any, value: Any) -> None:
        leaf = self._local.require_leaf()

        index = bisect_right(leaf.value.keys, key)
        leaf.value.keys.insert(index, key)
        leaf.value.values.insert(index, value)

        self._perform_insertion()

    def _perform_insertion(self):
        path = self._local.path
        index = len(path) - 1

        while index >= 0:
            node_key = path[index]
            node = self._local.require(node_key)

            if len(node.value.keys) >= self._order:
                if index > 0:
                    parent_key = path[index - 1]
                    parent_node = self._local.require(parent_key)
                    self._insert_in_parent(child_node=node, parent_node=parent_node)
                    index -= 1
                else:
                    self._create_parent(child_node=node)
                    break
            else:
                break

    def _find_path_with_siblings(
        self, key: Any
    ) -> tuple[dict[int, BPlusNode], list[int], dict[int, BPlusNode], dict[int, int]]:
        path_nodes: dict[int, BPlusNode] = {}
        child_indices: list[int] = []
        siblings: dict[int, BPlusNode] = {}
        sibling_indices: dict[int, int] = {}
        min_keys = (self._order - 1) // 2
        level = 0

        root = self.root
        assert root is not None

        self._fetch_node(key=root[0], leaf=root[1])
        node = self._local.pop(root[0])
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        path_nodes[level] = node

        while not self._local.is_leaf_node(node):
            child_index = bisect_right(node.value.keys, key)
            child_indices.append(child_index)
            sibling_index = child_index - 1 if child_index > 0 else child_index + 1

            child_node = self._fetch_child(node, child_index)
            sibling_node = self._fetch_child(node, sibling_index)
            siblings[level + 1] = sibling_node
            sibling_indices[level + 1] = sibling_index
            level += 1
            if len(child_node.value.keys) > min_keys:
                self._return_to_stash([*path_nodes.values(), *siblings.values()])
                path_nodes.clear()
                siblings.clear()
            path_nodes[level] = child_node
            node = child_node

        return path_nodes, child_indices, siblings, sibling_indices

    def _fetch_child(self, parent: BPlusNode, index: int) -> BPlusNode:
        child_key, child_leaf = parent.value.values[index]
        new_leaf = self._get_new_leaf()
        self._fetch_node(key=child_key, leaf=child_leaf)
        child = self._local.pop(child_key)
        child.leaf = new_leaf
        parent.value.values[index] = (child_key, new_leaf)
        return child

    def _apply_delete(
        self,
        key: Any,
        path_nodes: dict[int, BPlusNode],
        child_indices: list[int],
        siblings: dict[int, BPlusNode],
        sibling_indices: dict[int, int],
    ) -> Any:
        min_keys = (self._order - 1) // 2

        leaf_level = len(child_indices)
        leaf = path_nodes[leaf_level]

        key_index = None
        deleted_value = None
        for i, k in enumerate(leaf.value.keys):
            if k == key:
                key_index = i
                deleted_value = leaf.value.values[i]
                break

        if key_index is not None:
            leaf.value.keys.pop(key_index)
            leaf.value.values.pop(key_index)

            if not child_indices:
                if len(leaf.value.keys) == 0:
                    self.root = None
                    del path_nodes[leaf_level]
            else:
                node_level = leaf_level
                node = leaf
                for level in range(len(child_indices) - 1, -1, -1):
                    if len(node.value.keys) >= min_keys:
                        break
                    parent = path_nodes[level]
                    child_index = child_indices[level]
                    child_level = level + 1
                    sibling = siblings[child_level]
                    sib_index = sibling_indices[child_level]
                    is_left_sibling = sib_index < child_index
                    is_leaf = len(node.value.keys) == len(node.value.values)

                    if len(sibling.value.keys) > min_keys:
                        if is_left_sibling:
                            if is_leaf:
                                node.value.keys.insert(0, sibling.value.keys.pop())
                                node.value.values.insert(0, sibling.value.values.pop())
                                parent.value.keys[child_index - 1] = node.value.keys[0]
                            else:
                                node.value.keys.insert(0, parent.value.keys[child_index - 1])
                                node.value.values.insert(0, sibling.value.values.pop())
                                parent.value.keys[child_index - 1] = sibling.value.keys.pop()
                        else:
                            if is_leaf:
                                node.value.keys.append(sibling.value.keys.pop(0))
                                node.value.values.append(sibling.value.values.pop(0))
                                parent.value.keys[child_index] = sibling.value.keys[0]
                            else:
                                node.value.keys.append(parent.value.keys[child_index])
                                node.value.values.append(sibling.value.values.pop(0))
                                parent.value.keys[child_index] = sibling.value.keys.pop(0)
                        break

                    if is_left_sibling:
                        if is_leaf:
                            sibling.value.keys.extend(node.value.keys)
                            sibling.value.values.extend(node.value.values)
                        else:
                            sibling.value.keys.append(parent.value.keys[child_index - 1])
                            sibling.value.keys.extend(node.value.keys)
                            sibling.value.values.extend(node.value.values)
                        parent.value.keys.pop(child_index - 1)
                        parent.value.values.pop(child_index)
                        del path_nodes[node_level]
                    else:
                        if is_leaf:
                            node.value.keys.extend(sibling.value.keys)
                            node.value.values.extend(sibling.value.values)
                        else:
                            node.value.keys.append(parent.value.keys[child_index])
                            node.value.keys.extend(sibling.value.keys)
                            node.value.values.extend(sibling.value.values)
                        parent.value.keys.pop(child_index)
                        parent.value.values.pop(child_index + 1)
                        del siblings[child_level]

                    node_level = level
                    node = parent

                root_node = path_nodes.get(0)
                if root_node is not None and len(root_node.value.keys) == 0:
                    child_key, child_leaf = root_node.value.values[0]
                    self.root = (child_key, child_leaf)
                    del path_nodes[0]

        return deleted_value

    @override
    def _search(self, key: Any, value: Any = None) -> Any:
        self._begin_op("search")
        if self._short_circuit_read(key=key):
            return None

        leaf = self._descend(key)
        search_value = None

        for index, each_key in enumerate(leaf.value.keys):
            if key == each_key:
                search_value = leaf.value.values[index]
                if value is not None:
                    leaf.value.values[index] = value
                break

        self._flush_local_to_stash()
        self._pad_to()
        return search_value

    @override
    def _insert(self, key: Any, value: Any = None) -> None:
        self._begin_op("insert")
        if key is None:
            self._pad_to()
            return

        if self.root is None:
            data_block = self._get_bplus_data(keys=[key], values=[value])
            self._return_to_stash([data_block])
            assert data_block.leaf is not None
            self.root = (data_block.key, data_block.leaf)
            self._pad_to()
            return

        self._descend(key, hold=lambda node: len(node.value.keys) >= self._order - 1)
        self._insert_into_loaded_leaf(key=key, value=value)

        self._flush_local_to_stash()
        self._pad_to()

    @override
    def _delete(self, key: Any) -> Any:
        self._begin_op("delete")

        if self._short_circuit_read(key=key):
            return None
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        path_nodes, child_indices, siblings, sibling_indices = self._find_path_with_siblings(key=key)
        deleted_value = self._apply_delete(
            key=key,
            path_nodes=path_nodes,
            child_indices=child_indices,
            siblings=siblings,
            sibling_indices=sibling_indices,
        )

        self._return_to_stash([*path_nodes.values(), *siblings.values()])
        self._pad_to()
        return deleted_value
