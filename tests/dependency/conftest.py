from abc import ABC, abstractmethod
from typing import ClassVar, override

import pytest

from oblivlib.dependency import AVLTree, BPlusTree, BPlusTreeNode, KVPair


class _SearchTreeAdapter(ABC):
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
    builders = {"avl": _AVLAdapter, "bplus": _BPlusAdapter}
    return builders[request.param]
