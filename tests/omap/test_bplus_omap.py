import random

import pytest

from oblivlib.dependency import AesGcm, BPlusOmapConfig
from oblivlib.dependency.bplus_tree import BPlusInternal, BPlusLeaf
from oblivlib.omap import BPlusOmap
from oblivlib.omap.bplus_omap import bplus_max_levels


@pytest.mark.parametrize(
    ("order", "num_keys", "levels"),
    [(3, 2, 1), (3, 3, 2), (3, 4, 2), (3, 5, 3), (3, 9, 4), (3, 17, 5), (4, 6, 2), (4, 7, 3), (5, 12, 2), (5, 13, 3)],
)
def test_level_bound_is_the_fewest_keys_a_root_split_needs(order, num_keys, levels):
    assert bplus_max_levels(num_keys, order) == levels


def _witness_omap(client, order, num_data, paths_per_round, *, cached=False):
    return BPlusOmap(
        BPlusOmapConfig(
            order=order,
            num_data=num_data,
            key_size=4,
            data_size=4,
            client=client,
            distinguishable=True,
            paths_per_round=paths_per_round,
            cached=cached,
        )
    )


@pytest.mark.parametrize(("order", "num_data", "witness"), [(3, 8, 7), (3, 16, 13), (5, 36, 19), (5, 108, 55)])
def test_sorted_inserts_reach_every_single_path_bplus_bound(client, order, num_data, witness):
    omap = _witness_omap(client, order, num_data, paths_per_round=1)
    height = bplus_max_levels(num_data, order)
    assert bplus_max_levels(num_data + 1, order) == height + 1
    omap.init_server_storage()
    for i in range(witness):
        omap.insert(b"%04d" % i, b"v")
    assert (omap._op_fetches, omap._op_owed) == (height, 2 * height - 2)
    assert omap._op_fetches + omap._op_owed == omap._budget == 3 * height - 2
    omap.search(b"0000")
    assert omap._op_fetches == omap._budget == height
    omap.delete(b"9999")
    assert (omap._op_fetches, omap._op_owed) == (2 * height - 1, 2 * height - 2)
    assert omap._op_fetches + omap._op_owed == omap._budget == 4 * height - 3


@pytest.mark.parametrize(("order", "num_data", "witness"), [(3, 8, 7), (3, 16, 13), (5, 36, 19), (5, 108, 55)])
def test_sorted_inserts_reach_every_two_path_bplus_bound(client, order, num_data, witness):
    omap = _witness_omap(client, order, num_data, paths_per_round=2)
    height = bplus_max_levels(num_data, order)
    omap.init_server_storage()
    for i in range(witness):
        omap.insert(b"%04d" % i, b"v")
    assert (omap._op_fetches, omap._op_owed) == (height, 2 * height - 3)
    assert omap._op_fetches + -(-omap._op_owed // 2) == omap._budget == 2 * height - 1
    omap.search(b"0000")
    assert omap._op_fetches == omap._budget == height
    omap.delete(b"9999")
    assert (omap._op_fetches, omap._op_owed) == (height, 2 * height - 3)
    assert omap._op_fetches + -(-omap._op_owed // 2) == omap._budget == 2 * height - 1


@pytest.mark.parametrize(("order", "num_data", "witness"), [(3, 8, 7), (3, 16, 13), (5, 36, 19), (5, 108, 55)])
@pytest.mark.parametrize(
    ("paths_per_round", "insert_rounds", "insert_carry", "delete_rounds", "delete_carry"),
    [(2, 1, -3, 1, -3), (1, 2, -1, 2, -2)],
    ids=["two_path", "one_path"],
)
def test_sorted_inserts_reach_every_cached_bplus_bound(
    client, order, num_data, witness, paths_per_round, insert_rounds, insert_carry, delete_rounds, delete_carry
):
    omap = _witness_omap(client, order, num_data, paths_per_round, cached=True)
    height = bplus_max_levels(num_data, order)
    omap.init_server_storage()
    for i in range(witness - 1):
        omap.insert(b"%04d" % i, b"v")
    omap.drain()
    omap.insert(b"%04d" % (witness - 1), b"v")
    assert omap._op_fetches == height
    assert omap._budget == insert_rounds * height - (insert_rounds - 1)
    assert omap._owed == (2 if paths_per_round == 2 else 1) * height + insert_carry
    omap.drain()
    omap.delete(b"9999")
    assert omap._budget == delete_rounds * height - (delete_rounds - 1)
    assert omap._owed == 2 * height + delete_carry <= omap._carry_limit == 2 * height - 2
    assert omap.search(b"0000") == b"v"


def _batch_bplus(client, *, batch_size, num_data=64, **config):
    omap = BPlusOmap(
        BPlusOmapConfig(
            order=3, num_data=num_data, key_size=4, data_size=4, client=client, batch_size=batch_size, **config
        )
    )
    omap.init_server_storage()
    return omap


def test_a_batch_of_sorted_keys_splits_one_leaf_repeatedly(client, check_bplus_invariants):
    omap = _batch_bplus(client, batch_size=8)
    pairs = [(b"%04d" % i, b"v") for i in range(8)]
    omap.insert_batch(pairs)
    check_bplus_invariants(omap, dict(pairs))
    omap.insert_batch([(b"%04d" % i, b"w") for i in range(8, 16)])
    check_bplus_invariants(omap, {**dict(pairs), **{b"%04d" % i: b"w" for i in range(8, 16)}})


def test_an_even_order_tree_reaches_its_level_bound_through_deletes(client):
    omap = BPlusOmap(BPlusOmapConfig(order=4, num_data=7, key_size=4, data_size=4, client=client))
    omap.init_server_storage()
    history = [("i", 3), ("i", 34), ("i", 37), ("i", 32), ("i", 5), ("i", 4), ("i", 35), ("d", 32)]
    history += [("i", 36), ("d", 3), ("d", 36), ("d", 34), ("i", 25), ("i", 14), ("i", 8)]
    for op, key in history:
        if op == "i":
            omap.insert(b"%04d" % key, b"v")
        else:
            omap.delete(b"%04d" % key)
    omap.search(b"0004")
    assert omap._op_fetches == bplus_max_levels(7, 4) == 3


def test_an_even_order_tree_stays_valid_through_deletes(client, check_bplus_invariants):
    order = 4
    omap = BPlusOmap(BPlusOmapConfig(num_data=64, key_size=4, data_size=4, client=client, order=order))
    omap.init_server_storage()
    rng = random.Random(order)
    model: dict[bytes, bytes] = {}
    for step in range(400):
        key = b"%02d" % rng.randrange(48)
        if key in model and rng.random() < 0.5:
            assert omap.delete(key) == model.pop(key)
        elif key not in model:
            omap.insert(key, b"%d" % step)
            model[key] = b"%d" % step
        if step % 20 == 0:
            check_bplus_invariants(omap, model)
    check_bplus_invariants(omap, model)


@pytest.mark.parametrize("first_block_id", [0, 2**40])
def test_insert_delete_churn_never_outgrows_the_row(client, first_block_id):
    omap = BPlusOmap(
        BPlusOmapConfig(order=3, num_data=64, key_size=10, data_size=10, client=client, encryptor=AesGcm())
    )
    omap._block_id = first_block_id
    omap.init_server_storage()
    model = {}
    workload = random.Random(f"churn-{first_block_id}")
    for _ in range(1500):
        key = b"%010d" % workload.randrange(48)
        if key in model and workload.random() < 0.5:
            assert omap.delete(key) == model.pop(key)
        elif key not in model:
            model[key] = workload.randbytes(10)
            omap.insert(key, model[key])
        else:
            assert omap.search(key) == model[key]
    assert omap._block_id - first_block_id > 64
    for key, value in model.items():
        assert omap.search(key) == value


def test_a_node_row_keeps_one_size_and_rebuilds_the_node_kind_its_tag_names(client):
    omap = BPlusOmap(BPlusOmapConfig(order=3, num_data=4, key_size=2, data_size=3, client=client))
    layout = omap._block_layout
    leaf = BPlusLeaf(key=7, leaf=1, keys=[b"a"], values=[b"xyz"])
    internal = BPlusInternal(key=2**64 - 1, leaf=0, keys=[b"a", b"b"], children=[(2**64 - 1, 0), (1, 3), (2, 2)])

    rows = [layout.encode(leaf), layout.encode(internal)]

    assert {len(row) for row in rows} == {layout.size}
    assert [row[8 + 1] for row in rows] == [0, 1]
    assert [layout.decode(row) for row in rows] == [leaf, internal]
