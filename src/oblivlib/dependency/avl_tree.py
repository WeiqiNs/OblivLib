from __future__ import annotations

import secrets
from dataclasses import dataclass

from oblivlib.dependency.types import KVPair


@dataclass
class AVLNode:
    key: bytes
    leaf: int
    value: bytes
    r_key: bytes | None = None
    r_leaf: int | None = None
    r_height: int = 0
    l_key: bytes | None = None
    l_leaf: int | None = None
    l_height: int = 0


class AVLTreeNode:
    def __init__(self, kv_pair: KVPair):
        self.key: bytes = kv_pair.key
        self.leaf: int | None = None
        self.value: bytes = kv_pair.value
        self.height: int = 1
        self.left_node: AVLTreeNode | None = None
        self.right_node: AVLTreeNode | None = None


class AVLTree:
    def __init__(self, leaf_range: int):
        self._leaf_range: int = leaf_range

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

    def get_data_list(self, root: AVLTreeNode) -> list[AVLNode]:
        root.leaf = self._get_new_leaf()
        stack = [root]

        result: list[AVLNode] = []

        while stack:
            node = stack.pop()
            assert node.leaf is not None
            block = AVLNode(key=node.key, leaf=node.leaf, value=node.value)

            if node.left_node:
                node.left_node.leaf = self._get_new_leaf()
                block.l_key = node.left_node.key
                block.l_leaf = node.left_node.leaf
                block.l_height = node.left_node.height
                stack.append(node.left_node)

            if node.right_node:
                node.right_node.leaf = self._get_new_leaf()
                block.r_key = node.right_node.key
                block.r_leaf = node.right_node.leaf
                block.r_height = node.right_node.height
                stack.append(node.right_node)

            result.append(block)

        return result
