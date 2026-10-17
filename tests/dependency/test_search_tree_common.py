"""Behavioral tests shared by AVLTree and BPlusTree, parametrized via the `make_search_tree` factory.

These assert the insert/search/delete contract holds regardless of internal structure; the
structure-specific assertions live in `test_avl_tree.py` and `test_bplus_tree.py`.
"""

import random

import pytest


def _assert_absent(tree, key) -> None:
    """Assert a missing key follows the tree's contract: KeyError for B+ tree, None for AVL."""
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


class TestSearchTreeBatched:
    """The batched (multi-key) descent that the oblivious port will mirror: one level-synchronized
    descent serves the whole key set rather than one descent per key. The per-tree files cross-check it
    against sequential single-key ops; this pins the empty and absent-key edges both trees share."""

    def test_multi_ops_handle_empty_inputs(self, make_search_tree):
        tree = make_search_tree(leaf_range=1000)
        assert tree.multi_search([1, 2, 3]) == {1: None, 2: None, 3: None}
        tree.multi_insert([])
        assert tree.multi_search([]) == {}

        tree.multi_insert([(5, 50), (6, 60)])
        assert tree.multi_search([5, 6, 7]) == {5: 50, 6: 60, 7: None}
