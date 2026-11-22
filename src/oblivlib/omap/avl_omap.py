from collections.abc import Callable, Sequence
from typing import override

from oblivlib.dependency import AVLNode, AVLTree, AVLTreeNode, KVPair
from oblivlib.dependency.config import AvlOmapConfig
from oblivlib.dependency.layout import Bytes, Field, Optional, Record, UInt
from oblivlib.omap.ost_base_omap import OstBaseOmap, Root

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


def _child(key: bytes | None, leaf: int | None) -> tuple[bytes, int]:
    assert key is not None and leaf is not None
    return key, leaf


class AVLOmap(OstBaseOmap[AvlOmapConfig, bytes, AVLNode]):
    @property
    @override
    def _fanout(self) -> int:
        return 2

    @override
    def _height_bound(self, num_nodes: int) -> int:
        return avl_max_height(num_nodes)

    @property
    @override
    def _key_field(self) -> Field[bytes]:
        return Bytes(self._key_size)

    @property
    @override
    def _block_layout(self) -> Field[AVLNode]:
        child_key, child_leaf, height = Optional(self._key_field), Optional(self._leaf_field), UInt(_HEIGHT_BITS)
        return Record(
            AVLNode,
            (
                self._key_field,
                self._leaf_field,
                Bytes(self._data_size),
                child_key,
                child_leaf,
                height,
                child_key,
                child_leaf,
                height,
            ),
        )

    @override
    def _build_ods_blocks(self, data: list[KVPair]) -> tuple[list[AVLNode], Root[bytes]]:
        root: AVLTreeNode | None = None
        avl_tree = AVLTree(leaf_range=self._leaf_range)
        for kv_pair in data:
            root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

        assert root is not None
        blocks = avl_tree.get_data_list(root=root)
        assert root.leaf is not None
        return blocks, (root.key, root.leaf)

    def _path_to(self, key: bytes) -> list[bytes]:
        path: list[bytes] = []
        parent_key: bytes | None = None
        node_key = self._local.root_key
        while node_key is not None:
            node = self._local.require(node_key)
            self._local.parent_of[node_key] = parent_key
            path.append(node_key)
            parent_key, node_key = node_key, node.r_key if node.key < key else node.l_key
        self._local.path = path
        return path

    def _reparent(self, child_key: bytes, new_parent_key: bytes | None) -> None:
        self._local.parent_of[child_key] = new_parent_key
        if new_parent_key is None:
            self._local.root_key = child_key

    def _swap_in_path(self, key_a: bytes, key_b: bytes) -> None:
        path = self._local.path
        if key_a in path and key_b in path:
            idx_a = path.index(key_a)
            idx_b = path.index(key_b)
            path[idx_a], path[idx_b] = path[idx_b], path[idx_a]

    def _refresh_leaf(self, key: bytes) -> None:
        node = self._local.nodes[key]
        node.leaf = self._get_new_leaf()
        parent = self._local.get_parent(key)
        if parent is not None:
            if parent.l_key == key:
                parent.l_leaf = node.leaf
            elif parent.r_key == key:
                parent.r_leaf = node.leaf

    def _update_child_in_parent(
        self,
        parent_key: bytes | None,
        old_child_key: bytes,
        *,
        new_key: bytes | None,
        new_leaf: int | None,
        new_height: int,
    ) -> None:
        parent = None if parent_key is None else self._local.nodes.get(parent_key)
        if parent is None:
            return

        if parent.l_key == old_child_key:
            parent.l_key = new_key
            parent.l_leaf = new_leaf
            parent.l_height = new_height
        elif parent.r_key == old_child_key:
            parent.r_key = new_key
            parent.r_leaf = new_leaf
            parent.r_height = new_height

    def _replace_node_key(self, old_key: bytes, new_key: bytes) -> None:
        local = self._local
        local.nodes[new_key] = local.nodes.pop(old_key)
        local.parent_of[new_key] = local.parent_of.pop(old_key)
        if old_key in local.path:
            local.path[local.path.index(old_key)] = new_key
        if local.root_key == old_key:
            local.root_key = new_key
        for key, parent_key in local.parent_of.items():
            if parent_key == old_key:
                local.parent_of[key] = new_key

    def _update_height(self) -> None:
        for node_key in reversed(self._local.path):
            node = self._local.require(node_key)

            l_height = node.l_height if node.l_key is not None else 0
            r_height = node.r_height if node.r_key is not None else 0
            new_height = 1 + max(l_height, r_height)

            parent = self._local.get_parent(node_key)
            if parent is not None:
                if parent.l_key == node_key:
                    parent.l_height = new_height
                else:
                    parent.r_height = new_height

    def _rotate_node(self, node_key: bytes, rotate_left: bool) -> tuple[bytes, int, int]:
        node = self._local.require(node_key)
        pivot_key = node.r_key if rotate_left else node.l_key
        assert pivot_key is not None
        pivot = self._local.require(pivot_key)

        if rotate_left:
            node.r_key = pivot.l_key
            node.r_leaf = pivot.l_leaf
            node.r_height = pivot.l_height
            pivot.l_key = node.key
            pivot.l_leaf = node.leaf
            pivot.l_height = 1 + max(node.l_height, node.r_height)
        else:
            node.l_key = pivot.r_key
            node.l_leaf = pivot.r_leaf
            node.l_height = pivot.r_height
            pivot.r_key = node.key
            pivot.r_leaf = node.leaf
            pivot.r_height = 1 + max(node.l_height, node.r_height)

        grandparent_key = self._local.get_parent_key(node_key)
        self._reparent(pivot_key, grandparent_key)
        self._reparent(node_key, pivot_key)
        self._swap_in_path(node_key, pivot_key)

        new_height = 1 + max(pivot.l_height, pivot.r_height)
        return pivot.key, pivot.leaf, new_height

    def _fetch_for_rebalance(self, key: bytes, leaf: int, parent_key: bytes) -> None:
        self._fetch_node(key=key, leaf=leaf, parent_key=parent_key)
        self._refresh_leaf(key)

    def _balance_node(self, node_key: bytes) -> tuple[bytes, int, int]:
        node = self._local.require(node_key)
        balance = node.l_height - node.r_height

        if balance > 1:
            child_key, child_leaf = _child(node.l_key, node.l_leaf)
            if self._local.get(child_key) is None:
                self._fetch_for_rebalance(key=child_key, leaf=child_leaf, parent_key=node_key)

            child_node = self._local.require(child_key)
            if child_node.l_height - child_node.r_height < 0:
                grandchild_key, grandchild_leaf = _child(child_node.r_key, child_node.r_leaf)
                if self._local.get(grandchild_key) is None:
                    self._fetch_for_rebalance(key=grandchild_key, leaf=grandchild_leaf, parent_key=child_key)

                key, leaf, height = self._rotate_node(child_key, rotate_left=True)
                node.l_key, node.l_leaf, node.l_height = key, leaf, height

            return self._rotate_node(node_key, rotate_left=False)

        if balance < -1:
            child_key, child_leaf = _child(node.r_key, node.r_leaf)
            if self._local.get(child_key) is None:
                self._fetch_for_rebalance(key=child_key, leaf=child_leaf, parent_key=node_key)

            child_node = self._local.require(child_key)
            if child_node.l_height - child_node.r_height > 0:
                grandchild_key, grandchild_leaf = _child(child_node.l_key, child_node.l_leaf)
                if self._local.get(grandchild_key) is None:
                    self._fetch_for_rebalance(key=grandchild_key, leaf=grandchild_leaf, parent_key=child_key)

                key, leaf, height = self._rotate_node(child_key, rotate_left=False)
                node.r_key, node.r_leaf, node.r_height = key, leaf, height

            return self._rotate_node(node_key, rotate_left=True)

        return node.key, node.leaf, 1 + max(node.l_height, node.r_height)

    def _balance_local(self, is_delete: bool = False) -> None:
        idx = len(self._local.path) - 1

        while idx >= 0:
            node_key = self._local.path[idx]
            key, leaf, height = self._balance_node(node_key=node_key)

            parent = self._local.get_parent(key)
            if parent is not None:
                if parent.r_key == node_key or parent.r_key == key:
                    parent.r_key = key
                    parent.r_leaf = leaf
                    parent.r_height = height
                elif parent.l_key == node_key or parent.l_key == key:
                    parent.l_key = key
                    parent.l_leaf = leaf
                    parent.l_height = height

            if is_delete and idx > 0:
                ancestors = set(self._local.path[:idx])
                self._return_local([node for node in self._local.nodes if node not in ancestors])
            idx -= 1

        root_node = self._local.get_root()
        if root_node is not None:
            self.root: Root[bytes] | None = (root_node.key, root_node.leaf)

    @override
    def _descend(
        self, keys: Sequence[bytes], *, hold: bool, on_stop: Callable[[int, AVLNode], None] | None = None
    ) -> None:
        root_key = self._fetch_root().key
        at = dict.fromkeys(range(len(keys)), root_key)
        level = [root_key]
        while at:
            requests: dict[bytes, tuple[int, bytes | None, int]] = {}
            for lane, node_key in list(at.items()):
                node = self._local.require(node_key)
                go_right = node.key < keys[lane]
                child_key = None if node.key == keys[lane] else node.r_key if go_right else node.l_key
                if child_key is None:
                    if on_stop is not None:
                        on_stop(lane, node)
                    del at[lane]
                    continue
                if child_key not in requests:
                    new_leaf = self._get_new_leaf()
                    if go_right:
                        _, child_leaf = _child(child_key, node.r_leaf)
                        requests[child_key] = (child_leaf, node.key, new_leaf)
                        node.r_leaf = new_leaf
                    else:
                        _, child_leaf = _child(child_key, node.l_leaf)
                        requests[child_key] = (child_leaf, node.key, new_leaf)
                        node.l_leaf = new_leaf
                at[lane] = child_key
            if not requests:
                break
            if not hold:
                self._return_local(level)
            self._fetch_rehomed(requests)
            level = list(requests)

    def _delete_at_node(self, node_key: bytes) -> tuple[bytes, bool]:
        node = self._local.require(node_key)
        deleted_value = node.value
        parent_key = self._local.get_parent_key(node_key)

        if node.l_key is None and node.r_key is None:
            if len(self._local) == 1:
                self.root = None
                self._local.clear()
                return deleted_value, True

            self._update_child_in_parent(parent_key, node_key, new_key=None, new_leaf=None, new_height=0)
            self._local.remove(node_key)

        elif node.l_key is None or node.r_key is None:
            has_left = node.l_key is not None
            child_key = node.l_key if has_left else node.r_key
            child_leaf = node.l_leaf if has_left else node.r_leaf
            child_height = node.l_height if has_left else node.r_height

            if len(self._local) == 1:
                self.root = _child(child_key, child_leaf)
                self._local.clear()
                return deleted_value, True

            self._update_child_in_parent(
                parent_key, node_key, new_key=child_key, new_leaf=child_leaf, new_height=child_height
            )
            self._local.remove(node_key)

        else:
            use_predecessor = node.l_height > node.r_height
            original_key = node.key

            traverse_key, traverse_leaf = (
                _child(node.l_key, node.l_leaf) if use_predecessor else _child(node.r_key, node.r_leaf)
            )
            self._fetch_node(key=traverse_key, leaf=traverse_leaf, parent_key=node_key)
            current = self._local.require(traverse_key)
            parent_of_replacement_key = node_key

            next_key = current.r_key if use_predecessor else current.l_key
            while next_key is not None:
                _, next_leaf = _child(next_key, current.r_leaf if use_predecessor else current.l_leaf)
                self._fetch_node(key=next_key, leaf=next_leaf, parent_key=traverse_key)
                parent_of_replacement_key = traverse_key
                traverse_key = next_key
                current = self._local.require(traverse_key)
                next_key = current.r_key if use_predecessor else current.l_key

            replacement_node = current
            node.key = replacement_node.key
            node.value = replacement_node.value

            child_key = replacement_node.l_key if use_predecessor else replacement_node.r_key
            child_leaf = replacement_node.l_leaf if use_predecessor else replacement_node.r_leaf
            child_height = (
                (replacement_node.l_height if use_predecessor else replacement_node.r_height)
                if child_key is not None
                else 0
            )
            self._update_child_in_parent(
                parent_of_replacement_key,
                replacement_node.key,
                new_key=child_key,
                new_leaf=child_leaf,
                new_height=child_height,
            )
            self._local.remove(replacement_node.key)

            self._update_child_in_parent(parent_key, original_key, new_key=node.key, new_leaf=node.leaf, new_height=0)
            self._replace_node_key(original_key, node.key)

        return deleted_value, False

    @override
    def _swap_value(self, node: AVLNode, key: bytes, value: bytes | None) -> bytes | None:
        if node.key != key:
            return None
        found = node.value
        if value is not None:
            node.value = value
        return found

    @override
    def _insert_many(self, pairs: Sequence[tuple[bytes, bytes]]) -> None:
        self._begin_op("insert")
        blocks = [AVLNode(key=key, leaf=self._get_new_leaf(), value=value) for key, value in pairs]
        if not blocks:
            self._finish_op()
            return
        if self.root is None:
            first, *blocks = blocks
            self._local.add(first.key, first)
            self.root = (first.key, first.leaf)
        else:
            self._descend([block.key for block in blocks], hold=True)

        for block in blocks:
            parent = self._local.require(self._path_to(block.key)[-1])
            if parent.key < block.key:
                parent.r_key, parent.r_leaf = block.key, block.leaf
            else:
                parent.l_key, parent.l_leaf = block.key, block.leaf
            self._local.add(block.key, block, parent.key)
            self._update_height()
            self._balance_local()

        self._flush_local_to_stash()
        self._finish_op()

    @override
    def _delete_key(self, key: bytes | None) -> bytes | None:
        self._begin_op("delete")
        if key is None or self.root is None:
            self._finish_op()
            return None

        stops: list[AVLNode] = []
        self._descend([key], hold=True, on_stop=lambda _, node: stops.append(node))
        node = stops[0]
        if node.key != key:
            self._flush_local_to_stash()
            self._finish_op()
            return None

        deleted_value, done = self._delete_at_node(node_key=node.key)
        if done:
            self._finish_op()
            return deleted_value

        self._update_height()
        for node_key in self._local.nodes:
            self._refresh_leaf(node_key)
        self._balance_local(is_delete=True)

        self._flush_local_to_stash()
        self._finish_op()
        return deleted_value
