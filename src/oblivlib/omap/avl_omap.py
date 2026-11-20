from collections.abc import Callable
from functools import cached_property
from typing import Any, Protocol, cast, override

from oblivlib.dependency import AVLData, AVLTree, AVLTreeNode, Data, KVPair
from oblivlib.dependency.config import AvlOmapConfig
from oblivlib.dependency.layout import Bytes, Field, Optional, Record, UInt
from oblivlib.dependency.load_bound import max_bucket_load
from oblivlib.omap.ost_base_omap import ROOT, LocalNodesBase, OstBaseOmap

_HEIGHT_BITS = 8


def avl_delete_fetches(height: int) -> int:
    return max(path + 2 * min(height - path, path - 1) for path in range(1, height + 1))


def avl_max_height(num_nodes: int) -> int:
    height, fewest, previous = 1, 1, 0
    while True:
        fewest, previous = fewest + previous + 1, fewest
        if fewest > num_nodes:
            return height
        height += 1


class AVLNode(Protocol):
    key: Any
    leaf: int
    value: AVLData


class LocalNodes(LocalNodesBase[AVLNode]):
    def reparent(self, child_key: Any, new_parent_key: Any) -> None:
        self.parent_of[child_key] = new_parent_key
        if new_parent_key is None:
            self.root_key = child_key

    def swap_in_path(self, key_a: Any, key_b: Any) -> None:
        if key_a in self.path and key_b in self.path:
            idx_a = self.path.index(key_a)
            idx_b = self.path.index(key_b)
            self.path[idx_a], self.path[idx_b] = self.path[idx_b], self.path[idx_a]

    def update_all_leaves(self, get_new_leaf: Callable[[], int]) -> None:
        for key in self.nodes:
            self.refresh_leaf(key, get_new_leaf)

    def refresh_leaf(self, key: Any, get_new_leaf: Callable[[], int]) -> None:
        node = self.nodes[key]
        node.leaf = get_new_leaf()
        parent = self.get_parent(key)
        if parent:
            if parent.value.l_key == key:
                parent.value.l_leaf = node.leaf
            elif parent.value.r_key == key:
                parent.value.r_leaf = node.leaf

    def update_child_in_parent(
        self, parent_key: Any, old_child_key: Any, new_key: Any, new_leaf: Any, new_height: int
    ) -> None:
        parent = cast(AVLNode | None, self.nodes.get(parent_key))
        if parent is None:
            return

        if parent.value.l_key == old_child_key:
            parent.value.l_key = new_key
            parent.value.l_leaf = new_leaf
            parent.value.l_height = new_height
        elif parent.value.r_key == old_child_key:
            parent.value.r_key = new_key
            parent.value.r_leaf = new_leaf
            parent.value.r_height = new_height

    def replace_node_key(self, old_key: Any, new_key: Any) -> None:
        self.nodes[new_key] = self.nodes.pop(old_key)
        self.parent_of[new_key] = self.parent_of.pop(old_key)
        if old_key in self.path:
            self.path[self.path.index(old_key)] = new_key
        if self.root_key == old_key:
            self.root_key = new_key
        for key, parent_key in self.parent_of.items():
            if parent_key == old_key:
                self.parent_of[key] = new_key


def _balance(node: AVLNode) -> int:
    l_height = node.value.l_height if node.value.l_key is not None else 0
    r_height = node.value.r_height if node.value.r_key is not None else 0
    return l_height - r_height


class AVLOmap(OstBaseOmap[AvlOmapConfig, LocalNodes]):
    def __init__(self, config: AvlOmapConfig):
        super().__init__(config)

        self._max_height: int = avl_max_height(self._num_data)

    @override
    def _new_local(self) -> LocalNodes:
        return LocalNodes()

    @override
    def update_mul_tree_height(self, num_tree: int) -> None:
        tree_size = max_bucket_load(num_tree)
        self._max_height = avl_max_height(tree_size)

    @cached_property
    @override
    def _key_field(self) -> Field[Any]:
        return Bytes(self._key_size)

    @cached_property
    @override
    def _value_field(self) -> Field[Any]:
        child_key, child_leaf, height = Optional(self._key_field), Optional(self._leaf_field), UInt(_HEIGHT_BITS)
        return Record((Bytes(self._data_size), child_key, child_leaf, height, child_key, child_leaf, height), AVLData)

    @override
    def _build_ods_blocks(self, data: list[KVPair]) -> tuple[list[Data], ROOT]:
        root: AVLTreeNode | None = None
        avl_tree = AVLTree(leaf_range=self._leaf_range)
        for kv_pair in data:
            root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

        assert root is not None
        blocks = avl_tree.get_data_list(root=root)
        assert root.leaf is not None
        return blocks, (root.key, root.leaf)

    def _update_height(self) -> None:
        for node_key in reversed(self._local.path):
            node = self._local.require(node_key)

            l_height = node.value.l_height if node.value.l_key is not None else 0
            r_height = node.value.r_height if node.value.r_key is not None else 0
            new_height = 1 + max(l_height, r_height)

            parent = self._local.get_parent(node_key)
            if parent:
                if parent.value.l_key == node_key:
                    parent.value.l_height = new_height
                else:
                    parent.value.r_height = new_height

    def _rotate_node(self, node_key: Any, rotate_left: bool) -> tuple[Any, int, int]:
        node = self._local.require(node_key)
        pivot_key = node.value.r_key if rotate_left else node.value.l_key
        pivot = self._local.require(pivot_key)

        if rotate_left:
            node.value.r_key = pivot.value.l_key
            node.value.r_leaf = pivot.value.l_leaf
            node.value.r_height = pivot.value.l_height
            pivot.value.l_key = node.key
            pivot.value.l_leaf = node.leaf
            pivot.value.l_height = 1 + max(node.value.l_height, node.value.r_height)
        else:
            node.value.l_key = pivot.value.r_key
            node.value.l_leaf = pivot.value.r_leaf
            node.value.l_height = pivot.value.r_height
            pivot.value.r_key = node.key
            pivot.value.r_leaf = node.leaf
            pivot.value.r_height = 1 + max(node.value.l_height, node.value.r_height)

        grandparent_key = self._local.get_parent_key(node_key)
        self._local.reparent(pivot_key, grandparent_key)
        self._local.reparent(node_key, pivot_key)
        self._local.swap_in_path(node_key, pivot_key)

        new_height = 1 + max(pivot.value.l_height, pivot.value.r_height)
        return pivot.key, pivot.leaf, new_height

    def _fetch_for_rebalance(self, key: Any, leaf: int | None, parent_key: Any) -> None:
        self._fetch_node(key=key, leaf=leaf, parent_key=parent_key)
        self._local.refresh_leaf(key, self._get_new_leaf)

    def _balance_node(self, node_key: Any, is_delete: bool = False) -> tuple[Any, int, int]:
        node = self._local.require(node_key)
        balance = node.value.l_height - node.value.r_height

        if balance > 1:
            child_key = node.value.l_key
            if is_delete and not self._local.get(child_key):
                self._fetch_for_rebalance(key=child_key, leaf=node.value.l_leaf, parent_key=node_key)

            child_node = self._local.require(child_key)
            if child_node.value.l_height - child_node.value.r_height < 0:
                grandchild_key = child_node.value.r_key
                if is_delete and not self._local.get(grandchild_key):
                    self._fetch_for_rebalance(key=grandchild_key, leaf=child_node.value.r_leaf, parent_key=child_key)

                key, leaf, height = self._rotate_node(child_key, rotate_left=True)
                node.value.l_key, node.value.l_leaf, node.value.l_height = key, leaf, height

            return self._rotate_node(node_key, rotate_left=False)

        if balance < -1:
            child_key = node.value.r_key
            if is_delete and not self._local.get(child_key):
                self._fetch_for_rebalance(key=child_key, leaf=node.value.r_leaf, parent_key=node_key)

            child_node = self._local.require(child_key)
            if child_node.value.l_height - child_node.value.r_height > 0:
                grandchild_key = child_node.value.l_key
                if is_delete and not self._local.get(grandchild_key):
                    self._fetch_for_rebalance(key=grandchild_key, leaf=child_node.value.l_leaf, parent_key=child_key)

                key, leaf, height = self._rotate_node(child_key, rotate_left=False)
                node.value.r_key, node.value.r_leaf, node.value.r_height = key, leaf, height

            return self._rotate_node(node_key, rotate_left=True)

        return node.key, node.leaf, 1 + max(node.value.l_height, node.value.r_height)

    def _balance_local(self, is_delete: bool = False) -> None:
        idx = len(self._local.path) - 1

        while idx >= 0:
            node_key = self._local.path[idx]
            original_key = self._local.require(node_key).key
            key, leaf, height = self._balance_node(node_key=node_key, is_delete=is_delete)

            new_node = self._local.get(key)
            parent = self._local.get_parent(key) if new_node else None

            if parent:
                if parent.value.r_key == original_key or parent.value.r_key == key:
                    parent.value.r_key = key
                    parent.value.r_leaf = leaf
                    parent.value.r_height = height
                elif parent.value.l_key == original_key or parent.value.l_key == key:
                    parent.value.l_key = key
                    parent.value.l_leaf = leaf
                    parent.value.l_height = height

            if is_delete and idx > 0:
                ancestors = set(self._local.path[:idx])
                self._return_local([node for node in self._local.nodes if node not in ancestors])
            idx -= 1

        root_node = self._local.get_root()
        if root_node:
            self.root = (root_node.key, root_node.leaf)

    def _descend(self, key: Any, keep_from: Callable[[list[AVLNode]], int] | None = None) -> AVLNode:
        root = self.root
        assert root is not None
        new_leaf = self._get_new_leaf()
        self._fetch_node(key=root[0], leaf=root[1])
        node = self._local.require(root[0])
        node.leaf = new_leaf
        self.root = (node.key, new_leaf)

        while node.key != key:
            go_right = node.key < key
            child_key = node.value.r_key if go_right else node.value.l_key
            if child_key is None:
                break
            child_leaf = node.value.r_leaf if go_right else node.value.l_leaf
            new_leaf = self._get_new_leaf()
            if go_right:
                node.value.r_leaf = new_leaf
            else:
                node.value.l_leaf = new_leaf
            if keep_from is None:
                self._return_local([node.key])
            self._fetch_node(key=child_key, leaf=child_leaf, parent_key=node.key)
            node = self._local.require(child_key)
            node.leaf = new_leaf
            if keep_from is not None:
                path = [self._local.require(path_key) for path_key in self._local.path]
                self._return_local(self._local.path[: keep_from(path)])
        return node

    @staticmethod
    def _keep_from_rotation_parent(path: list[AVLNode]) -> int:
        unbalanced = [i for i, node in enumerate(path) if _balance(node) != 0]
        return max(unbalanced[-1] - 1, 0) if unbalanced else 0

    def _delete_at_node(self, node_key: Any) -> tuple[Any, bool]:
        node = self._local.require(node_key)
        deleted_value = node.value.value
        parent_key = self._local.get_parent_key(node_key)

        if node.value.l_key is None and node.value.r_key is None:
            if len(self._local) == 1:
                self.root = None
                self._local.clear()
                return deleted_value, True

            self._local.update_child_in_parent(parent_key, node_key, None, None, 0)
            self._local.remove(node_key)

        elif node.value.l_key is None or node.value.r_key is None:
            has_left = node.value.l_key is not None
            child_key = node.value.l_key if has_left else node.value.r_key
            child_leaf = node.value.l_leaf if has_left else node.value.r_leaf
            child_height = node.value.l_height if has_left else node.value.r_height

            if len(self._local) == 1:
                assert child_leaf is not None
                self.root = (child_key, child_leaf)
                self._local.clear()
                return deleted_value, True

            self._local.update_child_in_parent(parent_key, node_key, child_key, child_leaf, child_height)
            self._local.remove(node_key)

        else:
            use_predecessor = node.value.l_height > node.value.r_height
            original_key = node.key

            traverse_key = node.value.l_key if use_predecessor else node.value.r_key
            traverse_leaf = node.value.l_leaf if use_predecessor else node.value.r_leaf
            self._fetch_node(key=traverse_key, leaf=traverse_leaf, parent_key=node_key)
            current = self._local.require(traverse_key)
            parent_of_replacement_key = node_key

            next_key = current.value.r_key if use_predecessor else current.value.l_key
            while next_key is not None:
                next_leaf = current.value.r_leaf if use_predecessor else current.value.l_leaf
                self._fetch_node(key=next_key, leaf=next_leaf, parent_key=traverse_key)
                parent_of_replacement_key = traverse_key
                traverse_key = next_key
                current = self._local.require(traverse_key)
                next_key = current.value.r_key if use_predecessor else current.value.l_key

            replacement_node = current
            node.key = replacement_node.key
            node.value.value = replacement_node.value.value

            child_key = replacement_node.value.l_key if use_predecessor else replacement_node.value.r_key
            child_leaf = replacement_node.value.l_leaf if use_predecessor else replacement_node.value.r_leaf
            child_height = (
                (replacement_node.value.l_height if use_predecessor else replacement_node.value.r_height)
                if child_key is not None
                else 0
            )
            self._local.update_child_in_parent(
                parent_of_replacement_key, replacement_node.key, child_key, child_leaf, child_height
            )
            self._local.remove(replacement_node.key)

            self._local.update_child_in_parent(parent_key, original_key, node.key, node.leaf, 0)
            self._local.replace_node_key(original_key, node.key)

        return deleted_value, False

    @override
    def _search(self, key: Any, value: Any = None) -> Any:
        self._begin_op("search")
        if self._short_circuit_read(key=key):
            return None

        node = self._descend(key)
        search_value = node.value.value if node.key == key else None
        if value is not None and node.key == key:
            node.value.value = value

        self._flush_local_to_stash()
        self._finish_op()
        return search_value

    @override
    def _insert(self, key: Any, value: Any = None) -> None:
        self._begin_op("insert")
        if key is None:
            self._finish_op()
            return
        data_block = Data(key=key, leaf=self._get_new_leaf(), value=AVLData(value=value))

        if self.root is None:
            self._return_to_stash([data_block])
            assert data_block.leaf is not None
            self.root = (data_block.key, data_block.leaf)
            self._finish_op()
            return

        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        parent = self._descend(key, keep_from=self._keep_from_rotation_parent)
        if parent.key < key:
            parent.value.r_key, parent.value.r_leaf = data_block.key, data_block.leaf
        else:
            parent.value.l_key, parent.value.l_leaf = data_block.key, data_block.leaf
        self._local.add(node=data_block, parent_key=parent.key)
        self._update_height()
        self._balance_local()

        self._flush_local_to_stash()
        self._finish_op()

    @override
    def _delete(self, key: Any) -> Any:
        self._begin_op("delete")
        if self._short_circuit_read(key=key):
            return None

        node = self._descend(key, keep_from=lambda _: 0)
        if node.key != key:
            self._flush_local_to_stash()
            self._finish_op()
            return None

        deleted_value, done = self._delete_at_node(node_key=node.key)
        if done:
            self._finish_op()
            return deleted_value

        self._update_height()
        self._local.update_all_leaves(self._get_new_leaf)
        self._balance_local(is_delete=True)

        self._flush_local_to_stash()
        self._finish_op()
        return deleted_value
