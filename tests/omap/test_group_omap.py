import pytest

from oblivlib.dependency import GroupOmapConfig, PathOramConfig
from oblivlib.omap import GroupOmap
from oblivlib.oram import PathOram


def _make_omap(client, n=128, distinguishable=False):
    config = GroupOmapConfig(num_data=n, key_size=10, data_size=64, client=client, distinguishable=distinguishable)
    upper = PathOram(
        PathOramConfig(num_data=n, data_size=GroupOmap.upper_oram_data_size(config), client=client, name="upper")
    )
    return GroupOmap(config, upper_oram=upper)


@pytest.mark.parametrize(
    ("upper_num_data", "data_size", "message"),
    [(128, 8, "too small"), (64, 64, "num_data=64, not 128")],
    ids=["undersized", "other-num-data"],
)
def test_a_mismatched_upper_oram_is_refused(client, upper_num_data, data_size, message):
    upper = PathOram(PathOramConfig(num_data=upper_num_data, data_size=data_size, client=client, name="group_upper"))
    with pytest.raises(ValueError, match=message):
        GroupOmap(GroupOmapConfig(num_data=128, key_size=10, data_size=64, client=client), upper_oram=upper)


def test_slot_leaves_span_the_leaf_range_beyond_num_data(client):
    omap = _make_omap(client, n=100)
    leaf_range = omap._buckets.leaf_range
    leaves = {omap._slot_leaf(bytes([i % 256, i // 256]) * 16, 1) for i in range(4000)}
    assert leaf_range > 100
    assert max(leaves) >= 100
    assert leaves <= set(range(leaf_range))


def test_search_bucket_returns_the_bucket_and_reshuffles_it(client):
    omap = _make_omap(client, n=64)
    pairs = [(b"key%d" % i, b"value%d" % i) for i in range(20)]
    omap.init_server_storage(pairs)
    bucket = omap._hash_key_to_bucket(b"key3")
    expected = sorted((key, value) for key, value in pairs if omap._hash_key_to_bucket(key) == bucket)

    assert sorted(omap.search_bucket(bucket)) == expected
    assert sorted(omap.search_bucket(bucket)) == expected
    assert sorted(pair for each in range(64) for pair in omap.search_bucket(each)) == sorted(pairs)
    for key, value in pairs:
        assert omap.search(key) == value


def test_search_bucket_reads_as_many_lower_paths_as_a_search(recorded_client):
    client, recorder = recorded_client()
    omap = _make_omap(client, n=64)
    omap.init_server_storage([(b"key%d" % i, b"v") for i in range(30)])
    lower = omap._buckets.name

    def lower_reads(op):
        before = len(recorder.batches)
        op()
        return [len(read.leaves) for batch in recorder.batches[before:] for read in batch.reads if read.label == lower]

    search = lower_reads(lambda: omap.search(b"key3"))
    assert lower_reads(lambda: omap.search_bucket(omap._hash_key_to_bucket(b"key5"))) == search
    assert lower_reads(lambda: omap.search_bucket(omap._hash_key_to_bucket(b"absent"))) == search
