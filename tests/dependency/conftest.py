"""Fixtures shared by the dependency tests.

The AVL and B+ trees expose the same insert/search/delete contract, so a small adapter lets one
parametrized behavioral suite (`test_search_tree_common.py`) cover both. Scheme-specific structural
assertions stay in `test_avl_tree.py` / `test_bplus_tree.py`.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, override

import pytest

from oblivlib.dependency import AVLTree, BPlusTree, BPlusTreeNode, KVPair


class _SearchTreeAdapter(ABC):
    """Uniform stateful wrapper over a search tree, holding the tree and its current root."""

    missing_raises: ClassVar[bool]

    @abstractmethod
    def insert(self, key, value) -> None: ...

    @abstractmethod
    def search(self, key): ...

    @abstractmethod
    def delete(self, key) -> None: ...


class _AVLAdapter(_SearchTreeAdapter):
    missing_raises = False

    def __init__(self, leaf_range: int):
        self._tree = AVLTree(leaf_range=leaf_range)
        self._root = None

    @override
    def insert(self, key, value) -> None:
        self._root = self._tree.insert(root=self._root, kv_pair=KVPair(key=key, value=value))

    @override
    def search(self, key):
        return self._tree.search(key=key, root=self._root)

    @override
    def delete(self, key) -> None:
        self._root = self._tree.delete(root=self._root, key=key)


class _BPlusAdapter(_SearchTreeAdapter):
    missing_raises = True

    def __init__(self, leaf_range: int, order: int = 4):
        self._tree = BPlusTree(order=order, leaf_range=leaf_range)
        self._root = BPlusTreeNode()

    @override
    def insert(self, key, value) -> None:
        self._root = self._tree.insert(root=self._root, kv_pair=KVPair(key=key, value=value))

    @override
    def search(self, key):
        return self._tree.search(key=key, root=self._root)

    @override
    def delete(self, key) -> None:
        self._root = self._tree.delete(root=self._root, key=key) or BPlusTreeNode()


@pytest.fixture(params=["avl", "bplus"])
def make_search_tree(request):
    """Return a factory `make_search_tree(leaf_range=...)` producing an AVL or B+ tree adapter."""
    builders = {"avl": _AVLAdapter, "bplus": _BPlusAdapter}
    return builders[request.param]
