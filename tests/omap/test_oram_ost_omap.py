import pytest

from oblivlib.dependency import AvlOmapConfig, ContractError, PathOramConfig
from oblivlib.omap import AVLOmap, OramOstOmap, OramOstOmapConfig
from oblivlib.oram import PathOram


def _avl(n, client):
    return AvlOmapConfig(num_data=n, key_size=10, data_size=10, client=client)


def test_undersized_oram_raises(client):
    ost = _avl(64, client)
    oram = PathOram(PathOramConfig(num_data=64, data_size=OramOstOmap.oram_data_size(AVLOmap, ost) - 1, client=client))
    with pytest.raises(ValueError, match="too small"):
        OramOstOmap(OramOstOmapConfig(num_data=64), ost=AVLOmap, ost_config=ost, oram=oram)


def test_the_map_enforces_its_own_num_data_not_the_inner_maps(client):
    ost = _avl(64, client)
    oram = PathOram(PathOramConfig(num_data=16, data_size=OramOstOmap.oram_data_size(AVLOmap, ost), client=client))
    omap = OramOstOmap(OramOstOmapConfig(num_data=16), ost=AVLOmap, ost_config=ost, oram=oram)
    with pytest.raises(ContractError, match="OramOstOmap .*exceed num_data=16"):
        omap.init_server_storage([(b"%d" % i, b"v") for i in range(17)])
    omap.init_server_storage([(b"%d" % i, b"v") for i in range(15)])
    omap.insert(key=b"15", value=b"v")
    with pytest.raises(ContractError, match="OramOstOmap .*num_data=16"):
        omap.insert(key=b"16", value=b"v")
    assert omap.delete(key=b"0") == b"v"
    omap.insert(key=b"16", value=b"v")


def test_a_tree_map_smaller_than_the_map_is_refused(client):
    ost = _avl(8, client)
    oram = PathOram(PathOramConfig(num_data=16, data_size=OramOstOmap.oram_data_size(AVLOmap, ost), client=client))
    with pytest.raises(ValueError, match="tree map's num_data=8"):
        OramOstOmap(OramOstOmapConfig(num_data=16), ost=AVLOmap, ost_config=ost, oram=oram)
