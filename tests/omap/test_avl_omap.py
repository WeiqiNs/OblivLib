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


def _build(client, num_data, keys_by_path, scale=1):
    omap = AVLOmap(AvlOmapConfig(num_data=num_data, key_size=8, data_size=8, client=client, distinguishable=True))
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

    omap = _build(client, fibonacci_nodes + 1, tree, scale=2)
    omap.insert(_key(2 * tree[deepest] + 1), b"v")
    assert (omap._op_fetches, omap._op_owed) == (height, height)
    assert _work(omap) == omap._budget

    worst = _keys_by_path(_worst_delete_tree(height))
    omap = _build(client, len(worst), worst)
    assert len(worst) == fibonacci_nodes
    end_of_short_path = max((path for path in worst if set(path) <= {"R"}), key=len)
    omap.delete(_key(worst[end_of_short_path]))
    fetches = avl_delete_fetches(height)
    assert (omap._op_fetches, omap._op_owed) == (fetches, fetches - 2)
    assert _work(omap) == 2 * fetches - 2 <= omap._budget


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


def _record_flushes(omap: AVLOmap) -> list[dict[bytes, tuple[int | None, bytes | None, int]]]:
    flushes: list[dict[bytes, tuple[int | None, bytes | None, int]]] = []
    flush = omap._flush_local_to_stash

    def recording_flush():
        flushes.append(
            {key: (node.leaf, node.value.l_key, node.value.l_height) for key, node in omap._local.nodes.items()}
        )
        flush()

    omap._flush_local_to_stash = recording_flush
    return flushes


def test_rebalance_fetches_get_fresh_leaves(client):
    omap = _omap(client, num_data=2048)
    flushes = _record_flushes(omap)
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
        flushes.clear()
        assert omap.delete(_key(i)) == _key(i)
        written = flushes[-1] if flushes else {}
        for key, leaf in fetched:
            if key in written:
                total += 1
                stale += written[key][0] == leaf
    assert total > 50
    assert stale <= 2


def test_two_children_delete_keeps_the_height_of_an_empty_key_child(client):
    omap = AVLOmap(AvlOmapConfig(num_data=64, key_size=8, data_size=8, client=client))
    omap.init_server_storage()
    for key in [b"05", b"02", b"08", b"01", b"03", b"09", b""]:
        omap.insert(key, b"v")
    flushes = _record_flushes(omap)
    assert omap.delete(b"02") == b"v"
    _, l_key, l_height = flushes[-1][b"01"]
    assert l_key == b""
    assert l_height == 1
