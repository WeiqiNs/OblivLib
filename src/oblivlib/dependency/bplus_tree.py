from __future__ import annotations

import secrets
from bisect import bisect_right
from dataclasses import dataclass

from oblivlib.dependency.types import KVPair


@dataclass
class BPlusLeaf:
    key: int
    leaf: int
    keys: list[bytes]
    values: list[bytes]


@dataclass
class BPlusInternal:
    key: int
    leaf: int
    keys: list[bytes]
    children: list[tuple[int, int]]


BPlusNode = BPlusLeaf | BPlusInternal


class BPlusTreeNode:
    def __init__(self):
        self.id: int | None = None
        self.leaf: int | None = None
        self.keys: list[bytes] = []
        self.values: list[bytes] = []
        self.children: list[BPlusTreeNode] = []
        self.is_leaf: bool = True

    def add_kv_pair(self, kv_pair: KVPair):
        index = bisect_right(self.keys, kv_pair.key)
        self.keys.insert(index, kv_pair.key)
        self.values.insert(index, kv_pair.value)


class BPlusTree:
    def __init__(self, order: int, leaf_range: int):
        self._order: int = order
        self._mid: int = order // 2
        self._leaf_range: int = leaf_range

    def _get_new_leaf(self) -> int:
        return secrets.randbelow(self._leaf_range)

    @staticmethod
    def _child_index(node: BPlusTreeNode, key: bytes) -> int:
        for index, each_key in enumerate(node.keys):
            if key == each_key:
                return index + 1
            if key < each_key:
                return index
        return len(node.keys)

    @staticmethod
    def _find_leaf_path(key: bytes, root: BPlusTreeNode) -> list[BPlusTreeNode]:
        result = [root]
        cur_node = root
        while not cur_node.is_leaf:
            cur_node = cur_node.children[BPlusTree._child_index(node=cur_node, key=key)]
            result.append(cur_node)
        return result

    def _split_node(self, node: BPlusTreeNode) -> BPlusTreeNode:
        right_node = BPlusTreeNode()

        if node.is_leaf:
            right_node.keys = node.keys[self._mid :]
            right_node.values = node.values[self._mid :]
            node.keys = node.keys[: self._mid]
            node.values = node.values[: self._mid]

        else:
            right_node.is_leaf = False
            right_node.keys = node.keys[self._mid + 1 :]
            right_node.children = node.children[self._mid + 1 :]
            node.keys = node.keys[: self._mid]
            node.children = node.children[: self._mid + 1]

        return right_node

    def _insert_in_parent(self, child_node: BPlusTreeNode, parent_node: BPlusTreeNode) -> None:
        insert_key = child_node.keys[self._mid]
        right_node = self._split_node(node=child_node)

        index = bisect_right(parent_node.keys, insert_key)
        parent_node.keys.insert(index, insert_key)
        parent_node.children.insert(index + 1, right_node)

    def _create_parent(self, child_node: BPlusTreeNode) -> BPlusTreeNode:
        insert_key = child_node.keys[self._mid]
        right_node = self._split_node(node=child_node)

        parent_node = BPlusTreeNode()
        parent_node.is_leaf = False
        parent_node.keys.append(insert_key)
        parent_node.children = [child_node, right_node]

        return parent_node

    def insert(self, root: BPlusTreeNode, kv_pair: KVPair) -> BPlusTreeNode:
        leaves = self._find_leaf_path(root=root, key=kv_pair.key)
        leaves[-1].add_kv_pair(kv_pair=kv_pair)

        index = len(leaves) - 1
        while True:
            if len(leaves[index].keys) >= self._order:
                if index > 0:
                    self._insert_in_parent(child_node=leaves[index], parent_node=leaves[index - 1])
                    index -= 1
                else:
                    return self._create_parent(child_node=leaves[index])
            else:
                break

        return root

    def get_data_list(self, root: BPlusTreeNode, block_id: int = 0) -> list[BPlusNode]:
        root.id = block_id
        root.leaf = self._get_new_leaf()
        block_id += 1

        stack = [root]
        result: list[BPlusNode] = []

        while stack:
            node = stack.pop()
            assert node.id is not None and node.leaf is not None

            if not node.is_leaf:
                children: list[tuple[int, int]] = []
                for child in node.children:
                    child.id = block_id
                    child.leaf = self._get_new_leaf()
                    children.append((child.id, child.leaf))
                    block_id += 1

                result.append(BPlusInternal(key=node.id, leaf=node.leaf, keys=node.keys, children=children))
                stack.extend(node.children)

            else:
                result.append(BPlusLeaf(key=node.id, leaf=node.leaf, keys=node.keys, values=node.values))

        return result
