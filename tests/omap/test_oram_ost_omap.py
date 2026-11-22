import pytest

from oblivlib.dependency import AvlOmapConfig, PathOramConfig
from oblivlib.omap import AVLOmap, OramOstOmap
from oblivlib.oram import PathOram


def _avl(n, client):
    return AvlOmapConfig(num_data=n, key_size=10, data_size=10, client=client)


@pytest.mark.parametrize(
    ("oram_num_data", "spare_bytes", "message"),
    [(64, 0, None), (64, -1, "too small"), (32, 0, "num_data=32, not 64")],
    ids=["fits", "undersized", "other-num-data"],
)
def test_a_mismatched_oram_is_refused(client, oram_num_data, spare_bytes, message):
    ost = _avl(64, client)
    size = OramOstOmap.oram_data_size(AVLOmap, ost) + spare_bytes
    oram = PathOram(PathOramConfig(num_data=oram_num_data, data_size=size, client=client))
    if message is None:
        OramOstOmap(ost=AVLOmap, ost_config=ost, oram=oram)
        return
    with pytest.raises(ValueError, match=message):
        OramOstOmap(ost=AVLOmap, ost_config=ost, oram=oram)


@pytest.mark.parametrize("option", [{"cached": True}, {"batch_size": 2}], ids=["cached", "batched"])
def test_a_tree_map_it_cannot_drive_is_refused(client, option):
    ost = AvlOmapConfig(num_data=64, key_size=10, data_size=10, client=client, **option)
    oram = PathOram(PathOramConfig(num_data=64, data_size=OramOstOmap.oram_data_size(AVLOmap, ost), client=client))
    with pytest.raises(ValueError, match="cached=False and batch_size=1"):
        OramOstOmap(ost=AVLOmap, ost_config=ost, oram=oram)
