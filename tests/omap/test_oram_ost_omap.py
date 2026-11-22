import pytest

from oblivlib.dependency import AvlOmapConfig, PathOramConfig
from oblivlib.omap import AVLOmap, OramOstOmap
from oblivlib.oram import PathOram


def _avl(n, client):
    return AvlOmapConfig(num_data=n, key_size=10, data_size=10, client=client)


@pytest.mark.parametrize(
    ("oram_num_data", "spare_bytes", "message"),
    [(64, 0, None), (64, -1, "too small"), (32, 0, "ORAM num_data=32 differs")],
    ids=["fits", "undersized", "other-num-data"],
)
def test_the_oram_must_hold_one_root_slot_per_tree(client, oram_num_data, spare_bytes, message):
    ost = _avl(64, client)
    size = OramOstOmap.oram_data_size(AVLOmap, ost) + spare_bytes
    oram = PathOram(PathOramConfig(num_data=oram_num_data, data_size=size, client=client))
    if message is None:
        OramOstOmap(ost=AVLOmap, ost_config=ost, oram=oram)
        return
    with pytest.raises(ValueError, match=message):
        OramOstOmap(ost=AVLOmap, ost_config=ost, oram=oram)
