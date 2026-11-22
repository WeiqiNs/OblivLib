from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from oblivlib.dependency.types import Data, FieldTuple, KVPair


@dataclass
class AVLData(FieldTuple):
    value: Any = None
    r_key: Any = None
    r_leaf: int | None = None
    r_height: int = 0
    l_key: Any = None
    l_leaf: int | None = None
    l_height: int = 0


class AVLTreeNode:
    def __init__(self, kv_pair: KVPair):
        self.key: Any = kv_pair.key
        self.leaf: int | None = None
        self.value: Any = kv_pair.value
        self.height: int = 1
        self.left_node: AVLTreeNode | None = None
        self.right_node: AVLTreeNode | None = None


class AVLTree:
    def __init__(self, leaf_range: int):
        self._leaf_range = leaf_range

    def _get_new_leaf(self) -> int:
        return secrets.randbelow(self._leaf_range)

    @staticmethod
    def _get_height(node: AVLTreeNode | None) -> int:
        return node.height if node is not None else 0

    @staticmethod
    def _get_balance(node: AVLTreeNode | None) -> int:
        return AVLTree._get_height(node.left_node) - AVLTree._get_height(node.right_node) if node is not None else 0

    def _update_height(self, node: AVLTreeNode) -> None:
        node.height = 1 + max(self._get_height(node.left_node), self._get_height(node.right_node))

    def _rotate_left(self, in_node: AVLTreeNode) -> AVLTreeNode:
        assert in_node.right_node is not None
        p_node = in_node.right_node
        tmp_node = p_node.left_node

        p_node.left_node = in_node
        in_node.right_node = tmp_node

        self._update_height(in_node)
        self._update_height(p_node)

        return p_node

    def _rotate_right(self, in_node: AVLTreeNode) -> AVLTreeNode:
        assert in_node.left_node is not None
        p_node = in_node.left_node
        tmp_node = p_node.right_node

        p_node.right_node = in_node
        in_node.left_node = tmp_node

        self._update_height(in_node)
        self._update_height(p_node)

        return p_node

    def _balance(self, node: AVLTreeNode) -> AVLTreeNode:
        self._update_height(node)
        balance = self._get_balance(node)

        if balance > 1:
            assert node.left_node is not None
            if self._get_balance(node.left_node) < 0:
                node.left_node = self._rotate_left(node.left_node)
            return self._rotate_right(node)

        if balance < -1:
            assert node.right_node is not None
            if self._get_balance(node.right_node) > 0:
                node.right_node = self._rotate_right(node.right_node)
            return self._rotate_left(node)

        return node

    def recursive_insert(self, root: AVLTreeNode | None, kv_pair: KVPair) -> AVLTreeNode:
        if root is None:
            return AVLTreeNode(kv_pair)
        elif kv_pair.key < root.key:
            root.left_node = self.recursive_insert(root=root.left_node, kv_pair=kv_pair)
        else:
            root.right_node = self.recursive_insert(root=root.right_node, kv_pair=kv_pair)

        return self._balance(root)

    def get_data_list(self, root: AVLTreeNode) -> list[Data]:
        root.leaf = self._get_new_leaf()
        stack = [root]

        result = []

        while stack:
            node = stack.pop()
            avl_data = AVLData(value=node.value)

            if node.left_node:
                node.left_node.leaf = self._get_new_leaf()
                avl_data.l_key = node.left_node.key
                avl_data.l_leaf = node.left_node.leaf
                avl_data.l_height = node.left_node.height
                stack.append(node.left_node)

            if node.right_node:
                node.right_node.leaf = self._get_new_leaf()
                avl_data.r_key = node.right_node.key
                avl_data.r_leaf = node.right_node.leaf
                avl_data.r_height = node.right_node.height
                stack.append(node.right_node)

            result.append(Data(key=node.key, leaf=node.leaf, value=avl_data))

        return result
