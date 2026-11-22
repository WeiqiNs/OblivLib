import pytest

from oblivlib.dependency import GroupOmapConfig, PathOramConfig
from oblivlib.omap import GroupOmap
from oblivlib.oram import PathOram


def _make_omap(client, n=128, distinguishable=False):
    upper = PathOram(
        PathOramConfig(num_data=n, data_size=GroupOmap.upper_oram_data_size(num_data=n), client=client, name="upper")
    )
    return GroupOmap(
        GroupOmapConfig(num_data=n, key_size=10, data_size=64, client=client, distinguishable=distinguishable),
        upper_oram=upper,
    )


def _keys_in_one_bucket(omap, count):
    bucket = omap._hash_key_to_bucket(b"0")
    keys = (b"%d" % i for i in range(10**6))
    return [key for key in keys if omap._hash_key_to_bucket(key) == bucket][:count]


def test_undersized_upper_raises(client):
    n = 128
    too_small = PathOram(PathOramConfig(num_data=n, data_size=8, client=client, name="group_upper"))
    with pytest.raises(ValueError, match="too small"):
        GroupOmap(GroupOmapConfig(num_data=n, key_size=10, data_size=64, client=client), upper_oram=too_small)


def test_an_overfull_initial_bucket_raises(client):
    omap = _make_omap(client, n=64)
    keys = _keys_in_one_bucket(omap, omap._upper_bound + 1)
    with pytest.raises(MemoryError, match="exceeds upper bound"):
        omap.init_server_storage([(key, b"v") for key in keys])


@pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
def test_an_insert_into_a_full_bucket_raises(client, distinguishable):
    omap = _make_omap(client, n=64, distinguishable=distinguishable)
    *full, extra = _keys_in_one_bucket(omap, omap._upper_bound + 1)
    omap.init_server_storage([(key, b"v") for key in full])
    with pytest.raises(MemoryError, match="is full"):
        omap.insert(extra, b"v")


def test_lower_paths_cover_the_whole_leaf_range(client):
    omap = _make_omap(client, n=100)
    leaf_range = omap._lower_oram._leaf_range
    leaves = {omap._slot_leaf(bytes([i % 256, i // 256]) * 16, 1) for i in range(4000)}
    assert leaf_range > 100
    assert max(leaves) >= 100
    assert leaves <= set(range(leaf_range))


def test_search_group_returns_the_bucket_and_reshuffles_it(client):
    omap = _make_omap(client, n=64)
    pairs = [(b"key%d" % i, b"value%d" % i) for i in range(20)]
    omap.init_server_storage(pairs)
    bucket = omap._hash_key_to_bucket(b"key3")
    expected = sorted((key, value) for key, value in pairs if omap._hash_key_to_bucket(key) == bucket)

    assert sorted(omap.search_group(bucket)) == expected
    assert sorted(omap.search_group(bucket)) == expected
    assert sorted(pair for each in range(64) for pair in omap.search_group(each)) == sorted(pairs)
    for key, value in pairs:
        assert omap.search(key) == value


def test_search_group_reads_as_many_lower_paths_as_a_search(recorded_client):
    client, recorder = recorded_client()
    omap = _make_omap(client, n=64)
    omap.init_server_storage([(b"key%d" % i, b"v") for i in range(30)])
    lower = omap._lower_oram._name

    def lower_reads(op):
        before = len(recorder.batches)
        op()
        return [len(read.leaves) for batch in recorder.batches[before:] for read in batch.reads if read.label == lower]

    search = lower_reads(lambda: omap.search(b"key3"))
    assert lower_reads(lambda: omap.search_group(omap._hash_key_to_bucket(b"key5"))) == search
    assert lower_reads(lambda: omap.search_group(omap._hash_key_to_bucket(b"absent"))) == search
