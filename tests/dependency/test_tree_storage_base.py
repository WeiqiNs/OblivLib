import pytest

from oblivlib.dependency import (
    GroupOmapConfig,
    OramBlock,
    PathOramConfig,
    RecursiveOramConfig,
    StashOverflowError,
)
from oblivlib.omap import GroupOmap
from oblivlib.oram import PathOram, RecursivePathOram


def test_default_names_are_unique_per_config(client):
    first = PathOramConfig(num_data=4, data_size=4, client=client)
    second = PathOramConfig(num_data=4, data_size=4, client=client)
    assert first.name.startswith("po-")
    assert first.name != second.name


def test_stash_overflow_raises(client):
    base = PathOram(PathOramConfig(num_data=4, data_size=4, stash_scale=1, client=client, name="tiny"))
    base._stash = [OramBlock(key=i, leaf=0, value=b"") for i in range(base._stash_capacity)]
    base._check_stash()

    base._stash.append(OramBlock(key=99, leaf=0, value=b""))
    with pytest.raises(StashOverflowError, match="PathOram 'tiny'") as raised:
        base._check_stash()
    assert isinstance(raised.value, MemoryError)


def test_every_tree_of_a_composed_scheme_feeds_the_client_recorder(client, encryptor):
    upper = RecursivePathOram(
        RecursiveOramConfig(
            num_data=64, data_size=GroupOmap.upper_oram_data_size(64), client=client, encryptor=encryptor
        )
    )
    omap = GroupOmap(
        GroupOmapConfig(num_data=64, key_size=8, data_size=8, client=client, encryptor=encryptor), upper_oram=upper
    )
    omap.init_server_storage([(b"k%d" % i, b"v") for i in range(8)])
    assert upper._levels

    with client.measure() as measurement:
        for i in range(8, 16):
            omap.insert(b"k%d" % i, b"w")
        assert [omap.search(b"k%d" % i) for i in range(16)] == [b"v"] * 8 + [b"w"] * 8
        client.flush()

    spent = measurement.result
    assert spent.row_bytes_read > 0
    assert spent.opened_bytes == spent.row_bytes_read
    assert spent.sealed_bytes == spent.row_bytes_written
