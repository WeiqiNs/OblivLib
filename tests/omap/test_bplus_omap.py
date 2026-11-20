import random

import pytest

from oblivlib.dependency import BPlusOmapCachedConfig, BPlusOmapConfig
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
