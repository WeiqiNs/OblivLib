import pytest

from oblivlib.dependency import AVLNode, AvlOmapConfig
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


def test_a_node_row_stores_the_key_leaf_value_then_the_right_child_then_the_left(client):
    omap = AVLOmap(AvlOmapConfig(num_data=4, key_size=2, data_size=2, client=client))
    node = AVLNode(key=b"k", leaf=1, value=b"v", r_key=b"r", r_leaf=2, r_height=1)
    row = omap._block_layout.encode(node)
    assert row == bytes.fromhex("016b00010176000101720001020100000000000000")
    assert omap._block_layout.decode(row) == node


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


def test_a_wide_batch_holds_its_extra_paths_without_overflowing_the_stash(client):
    omap = AVLOmap(AvlOmapConfig(num_data=1024, key_size=8, data_size=8, client=client, batch_size=128))
    omap.init_server_storage([(_key(i), b"v") for i in range(1024)])
    for start in range(0, 1024, 128):
        assert omap.search_batch([(_key(i), None) for i in range(start, start + 128)]) == [b"v"] * 128
    assert omap._max_stash > omap._stash_capacity


def _record_returns(omap: AVLOmap) -> dict[bytes, tuple[int | None, bytes | None, int]]:
    returned: dict[bytes, tuple[int | None, bytes | None, int]] = {}
    return_to_stash = omap._return_to_stash

    def recording_return(blocks):
        returned.update({node.key: (node.leaf, node.l_key, node.l_height) for node in blocks})
        return_to_stash(blocks)

    omap._return_to_stash = recording_return
    return returned


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
