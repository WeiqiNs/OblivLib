import random

import pytest

from oblivlib.dependency import AvlOmapConfig
from oblivlib.omap import AVLOmap
from oblivlib.omap.avl_omap import avl_delete_fetches, avl_max_height


def _key(i: int) -> bytes:
    return f"{i:06d}".encode()


@pytest.mark.parametrize(
    ("num_nodes", "height"), [(1, 1), (2, 2), (3, 2), (4, 3), (6, 3), (7, 4), (53, 7), (54, 8), (87, 8), (88, 9)]
)
def test_height_bound_is_the_tallest_avl_tree_with_that_many_nodes(num_nodes, height):
    assert avl_max_height(num_nodes) == height


@pytest.mark.parametrize(("height", "fetches"), [(1, 1), (2, 2), (3, 4), (4, 5), (5, 7), (8, 11), (11, 16)])
def test_delete_fetch_bound_is_the_longest_path_plus_two_fetches_per_rotation(height, fetches):
    assert avl_delete_fetches(height) == fetches


def _fibonacci(height, *, mirror=False):
    if height <= 0:
        return None
    taller, shorter = _fibonacci(height - 1, mirror=mirror), _fibonacci(height - 2, mirror=mirror)
    return (shorter, taller) if mirror else (taller, shorter)


def _rotation_chain(height):
    if height <= 0:
        return None
    if height == 1:
        return (None, None)
    return (_fibonacci(height - 1, mirror=True), _rotation_chain(height - 2))


def _balanced_spine(height):
    if height == 1:
        return (None, None)
    return (_fibonacci(height - 1), _balanced_spine(height - 1))


def _worst_insert_tree(height):
    return (_fibonacci(height - 2), _balanced_spine(height - 1))


def _worst_delete_tree(height):
    if height % 2:
        return _rotation_chain(height)
    return (_fibonacci(height - 2), _rotation_chain(height - 1))


def _keys_by_path(tree):
    keys = {}

    def walk(node, path):
        if node is not None:
            walk(node[0], path + "L")
            keys[path] = len(keys)
            walk(node[1], path + "R")

    walk(tree, "")
    return keys


def _build(client, num_data, keys_by_path, scale=1, *, cached=False):
    omap = AVLOmap(
        AvlOmapConfig(num_data=num_data, key_size=8, data_size=8, client=client, distinguishable=True, cached=cached)
    )
    omap.init_server_storage()
    for path in sorted(keys_by_path, key=lambda path: (len(path), path)):
        omap.insert(_key(scale * keys_by_path[path]), b"v")
    return omap


def _work(omap):
    return omap._op_fetches + omap._op_owed


@pytest.mark.parametrize("height", [3, 5, 6, 8])
def test_each_avl_bound_is_reached(client, height):
    tree = _keys_by_path(_fibonacci(height))
    deepest = max(tree, key=len)
    fibonacci_nodes = len(tree)

    omap = _build(client, fibonacci_nodes, tree)
    omap.search(_key(tree[deepest]))
    assert (omap._op_fetches, omap._op_owed) == (height, 0)
    assert _work(omap) == omap._budget

    spine = _keys_by_path(_worst_insert_tree(height))
    omap = _build(client, len(spine) + 1, spine, scale=2)
    assert avl_max_height(len(spine) + 1) == height
    omap.insert(_key(2 * spine["R" * (height - 1)] + 1), b"v")
    assert (omap._op_fetches, omap._op_owed) == (height, height)
    assert _work(omap) == omap._budget

    worst = _keys_by_path(_worst_delete_tree(height))
    omap = _build(client, len(worst), worst)
    assert len(worst) == fibonacci_nodes
    end_of_short_path = max((path for path in worst if set(path) <= {"R"}), key=len)
    omap.delete(_key(worst[end_of_short_path]))
    fetches = avl_delete_fetches(height)
    assert (omap._op_fetches, omap._op_owed) == (fetches, 2 * height - fetches)
    assert _work(omap) == omap._budget


@pytest.mark.parametrize("height", [3, 5, 6, 8])
def test_each_cached_avl_bound_is_reached(client, height):
    spine = _keys_by_path(_worst_insert_tree(height))
    omap = _build(client, len(spine) + 1, spine, scale=2, cached=True)
    omap.insert(_key(2 * spine["R" * (height - 1)] + 1), b"v")
    assert omap._op_fetches == height
    assert omap._budget == height + 1

    worst = _keys_by_path(_worst_delete_tree(height))
    omap = _build(client, len(worst), worst, cached=True)
    end_of_short_path = max((path for path in worst if set(path) <= {"R"}), key=len)
    omap.drain()
    assert omap.delete(_key(worst[end_of_short_path])) == b"v"
    fetches = avl_delete_fetches(height)
    assert omap._op_fetches == omap._budget == fetches == (3 * height - 1) // 2
    assert omap._owed == 2 * height - fetches <= omap._carry_limit
    rest = [path for path in worst if path != end_of_short_path]
    assert [omap.search(_key(worst[path])) for path in rest] == [b"v"] * len(rest)


@pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
def test_cached_maps_match_the_model_and_drain_in_a_fixed_count(client, distinguishable, duplicate_blocks):
    rng = random.Random(11)
    omap = AVLOmap(
        AvlOmapConfig(num_data=64, key_size=8, data_size=8, client=client, cached=True, distinguishable=distinguishable)
    )
    omap.init_server_storage()
    model: dict[bytes, bytes] = {}
    for _ in range(1500):
        key, roll = _key(rng.randrange(64)), rng.random()
        if roll < 0.45 and key not in model:
            omap.insert(key, key)
            model[key] = key
        elif roll < 0.75:
            assert omap.delete(key) == model.pop(key, None)
        else:
            assert omap.search(key) == model.get(key)
        assert omap._owed <= omap._carry_limit

    rounds = client.metrics.rounds
    omap.drain()
    assert client.metrics.rounds - rounds == omap._carry_limit
    assert omap._owed == 0
    rounds = client.metrics.rounds
    omap.drain()
    assert client.metrics.rounds - rounds == omap._carry_limit
    assert all(omap.search(key) == value for key, value in model.items())
    assert not duplicate_blocks(omap)


def _omap(client, num_data: int) -> AVLOmap:
    omap = AVLOmap(AvlOmapConfig(num_data=num_data, key_size=8, data_size=8, client=client))
    omap.init_server_storage([(_key(i), _key(i)) for i in range(0, num_data, 2)])
    return omap


def test_delete_miss_moves_the_root(client):
    omap = _omap(client, num_data=1024)
    moved = 0
    for i in range(1, 41, 2):
        before = omap.root
        assert omap.delete(_key(i)) is None
        assert omap.root is not None and before is not None
        assert omap.root[0] == before[0]
        moved += omap.root[1] != before[1]
    assert moved >= 18


def _record_returns(omap: AVLOmap) -> dict[bytes, tuple[int | None, bytes | None, int]]:
    returned: dict[bytes, tuple[int | None, bytes | None, int]] = {}
    return_to_stash = omap._return_to_stash

    def recording_return(blocks):
        returned.update({node.key: (node.leaf, node.value.l_key, node.value.l_height) for node in blocks})
        return_to_stash(blocks)

    omap._return_to_stash = recording_return
    return returned


def test_rebalance_fetches_get_fresh_leaves(client):
    omap = _omap(client, num_data=2048)
    returned = _record_returns(omap)
    fetched: list[tuple[bytes, int]] = []
    fetch = omap._fetch_for_rebalance

    def recording_fetch(key, leaf, parent_key):
        fetched.append((key, leaf))
        fetch(key=key, leaf=leaf, parent_key=parent_key)

    omap._fetch_for_rebalance = recording_fetch
    keys = list(range(0, 2048, 2))
    random.Random(4).shuffle(keys)
    total = stale = 0
    for i in keys[:600]:
        fetched.clear()
        returned.clear()
        assert omap.delete(_key(i)) == _key(i)
        for key, leaf in fetched:
            if key in returned:
                total += 1
                stale += returned[key][0] == leaf
    assert total > 50
    assert stale <= 2


def test_two_children_delete_keeps_the_height_of_an_empty_key_child(client):
    omap = AVLOmap(AvlOmapConfig(num_data=64, key_size=8, data_size=8, client=client))
    omap.init_server_storage()
    for key in [b"05", b"02", b"08", b"01", b"03", b"09", b""]:
        omap.insert(key, b"v")
    returned = _record_returns(omap)
    assert omap.delete(b"02") == b"v"
    _, l_key, l_height = returned[b"01"]
    assert l_key == b""
    assert l_height == 1
