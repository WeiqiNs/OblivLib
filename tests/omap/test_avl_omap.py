import random

import pytest

from oblivlib.dependency import AvlOmapConfig
from oblivlib.omap import AVLOmap
from oblivlib.omap.avl_omap import avl_max_height


def _key(i: int) -> bytes:
    return f"{i:06d}".encode()


@pytest.mark.parametrize(
    ("num_nodes", "height"), [(1, 1), (2, 2), (3, 2), (4, 3), (6, 3), (7, 4), (53, 7), (54, 8), (87, 8), (88, 9)]
)
def test_height_bound_is_the_tallest_avl_tree_with_that_many_nodes(num_nodes, height):
    assert avl_max_height(num_nodes) == height


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
