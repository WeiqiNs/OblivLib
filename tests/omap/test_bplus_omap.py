import random

import pytest

from oblivlib.dependency import AesGcm, BPlusOmapCachedConfig, BPlusOmapConfig
from oblivlib.omap import BPlusOmap, BPlusOmapCached


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize(
    ("cls", "config_cls"), [(BPlusOmap, BPlusOmapConfig), (BPlusOmapCached, BPlusOmapCachedConfig)]
)
def test_small_order_deletes_keep_the_tree_valid(client, check_bplus_invariants, order, cls, config_cls):
    omap = cls(config_cls(num_data=64, key_size=4, data_size=4, client=client, order=order, stash_scale=20))
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


@pytest.mark.parametrize("num_data", [64, 128])
@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("reverse", [False, True], ids=["sorted", "reverse"])
@pytest.mark.parametrize(
    ("cls", "config_cls"), [(BPlusOmap, BPlusOmapConfig), (BPlusOmapCached, BPlusOmapCachedConfig)]
)
def test_ordered_inserts_and_deletes_keep_the_stash_bounded(client, cls, config_cls, reverse, order, num_data):
    omap = cls(config_cls(order=order, num_data=num_data, key_size=8, data_size=4, client=client))
    omap.init_server_storage()
    keys = [b"%08d" % i for i in range(num_data)]
    keys = keys[::-1] if reverse else keys
    for key in keys:
        omap.insert(key, key[-4:])
    for key in keys[::3]:
        assert omap.search(key) == key[-4:]
    for key in keys:
        assert omap.delete(key) == key[-4:]
    assert omap.root is None
    assert omap._max_stash <= omap._stash_capacity


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
