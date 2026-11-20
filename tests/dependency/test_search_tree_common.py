import random

import pytest


def _assert_absent(tree, key) -> None:
    if tree.missing_raises:
        with pytest.raises(KeyError):
            tree.search(key)
    else:
        assert tree.search(key) is None


class TestSearchTreeBehavior:
    def test_random_insertion_order_round_trip(self, make_search_tree):
        tree = make_search_tree(leaf_range=200000)
        keys = random.Random(1234).sample(range(100000), 500)
        for k in keys:
            tree.insert(k, k + 1)
        for k in keys:
            assert tree.search(k) == k + 1

    def test_missing_key(self, make_search_tree):
        tree = make_search_tree(leaf_range=1000)
        for k in range(50):
            tree.insert(k, k)
        _assert_absent(tree, 999)
        _assert_absent(tree, -1)

    def test_delete_root_with_one_child(self, make_search_tree):
        tree = make_search_tree(leaf_range=1000)
        tree.insert(1, 10)
        tree.insert(2, 20)
        tree.delete(1)
        _assert_absent(tree, 1)
        assert tree.search(2) == 20

    def test_delete_stress_random(self, make_search_tree):
        tree = make_search_tree(leaf_range=200000)
        keys = list(range(200))
        random.Random(99).shuffle(keys)
        for k in keys:
            tree.insert(k, k)
        to_delete, remaining = keys[:100], set(keys[100:])
        for k in to_delete:
            tree.delete(k)
        for k in remaining:
            assert tree.search(k) == k
        for k in to_delete:
            _assert_absent(tree, k)
