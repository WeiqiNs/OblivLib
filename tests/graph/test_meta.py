"""Tests for the delayed-duplication store: RL write paths, newest-wins merging, and the bucket bound."""

import pytest

from oblivlib.dependency import Data
from oblivlib.dependency.config import OramConfig
from oblivlib.graph.meta import DupStore, meta_bucket_size, overflow_probability


def _store(client, num_data=16, bucket_size=4):
    store = DupStore(OramConfig(num_data=num_data, data_size=4, client=client, bucket_size=bucket_size), key_size=4)
    store._host_tree(store._build_tree([]))
    return store


class TestDupStore:
    def test_rl_leaves_visit_every_leaf_once_per_cycle(self, client):
        store = _store(client, num_data=16)
        cycle = store.rl_leaves(store._leaf_range)
        assert sorted(cycle) == list(range(store._leaf_range))
        assert cycle[:4] == [0, 8, 4, 12]
        assert store.rl_leaves(2) == cycle[:2]

    def test_add_keeps_the_newest_copy_of_each_key(self, client):
        store = _store(client)
        store.add([Data(key=b"a", leaf=1, value=b"old"), Data(key=b"b", leaf=2, value=b"b")])
        store.add([Data(key=b"a", leaf=1, value=b"new"), Data(key=b"a", leaf=1, value=b"newer-ignored")])
        assert [(block.key, block.value) for block in store._stash] == [(b"a", b"new"), (b"b", b"b")]

    def _two_copies_on_path_3(self, client):
        store = _store(client)
        store._read_paths([3])
        store.add([Data(key=b"x", leaf=3, value=b"old")])
        store._write_paths()
        store._read_paths([11])
        store.add([Data(key=b"x", leaf=3, value=b"new")])
        store._write_paths()
        assert store._stash == []
        return store

    def test_a_read_keeps_the_shallowest_copy(self, client):
        store = self._two_copies_on_path_3(client)
        store._read_paths([3])
        assert [block.value for block in store._stash if block.key == b"x"] == [b"new"]

    def test_a_read_keeps_a_stash_copy_over_any_tree_copy(self, client):
        store = self._two_copies_on_path_3(client)
        store.add([Data(key=b"x", leaf=3, value=b"last")])
        store._read_paths([3])
        assert [block.value for block in store._stash if block.key == b"x"] == [b"last"]

    def test_take_returns_only_blocks_aimed_at_the_read_leaf(self, client):
        store = _store(client)
        store.add([Data(key=b"v1", leaf=3, value=b"a"), Data(key=b"v2", leaf=5, value=b"b")])
        assert [block.key for block in store.take(lambda key: key.startswith(b"v"), 3)] == [b"v1"]
        assert [block.key for block in store._stash] == [b"v2"]


class TestBucketBound:
    def test_more_insertions_need_a_larger_bucket(self):
        small = meta_bucket_size(insertions=2**10, group=4, level=11)
        large = meta_bucket_size(insertions=2**20, group=4, level=11)
        assert small <= large
        assert overflow_probability(insertions=2**10, group=4, bucket_size=small, level=11) <= 2**-128

    @pytest.mark.parametrize("group", [1, 3, 10])
    def test_the_bound_is_the_smallest_safe_bucket(self, group):
        size = meta_bucket_size(insertions=4096, group=group, level=13)
        assert overflow_probability(insertions=4096, group=group, bucket_size=size, level=13) <= 2**-128
        assert overflow_probability(insertions=4096, group=group, bucket_size=size - 1, level=13) > 2**-128
