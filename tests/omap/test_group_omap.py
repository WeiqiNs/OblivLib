import random
from typing import Any

import pytest

from oblivlib.dependency import AesGcm, ContractError, DaOramConfig, GroupOmapConfig, PathOramConfig
from oblivlib.omap import GroupOmap
from oblivlib.oram import DAOram, PathOram


def _make_upper(kind, n, client, encryptor):
    data_size = GroupOmap.upper_oram_data_size(num_data=n)
    if kind == "path":
        return PathOram(
            PathOramConfig(num_data=n, data_size=data_size, client=client, name="group_upper", encryptor=encryptor)
        )
    return DAOram(DaOramConfig(num_data=n, data_size=data_size, client=client, name="group_upper", encryptor=encryptor))


def _make_omap(client, n=128, encryptor=None, upper_kind="path", distinguishable=False):
    return GroupOmap(
        GroupOmapConfig(
            num_data=n, key_size=10, data_size=64, client=client, encryptor=encryptor, distinguishable=distinguishable
        ),
        upper_oram=_make_upper(upper_kind, n, client, encryptor),
    )


class TestGroupOmap:
    @pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
    @pytest.mark.parametrize("upper_kind", ["path", "da"])
    @pytest.mark.parametrize("enc", [False, True])
    def test_oracle(self, upper_kind, enc, distinguishable, client, duplicate_blocks):
        n = 128
        omap = _make_omap(client, n, AesGcm() if enc else None, upper_kind, distinguishable)
        omap.init_server_storage()
        rng = random.Random(f"{upper_kind}-{enc}")
        model, keyspace = {}, [b"%010d" % i for i in range(n)]
        for _ in range(n * 4):
            key = rng.choice(keyspace)
            roll = rng.random()
            if roll < 0.5:
                if key not in model:
                    value = rng.randbytes(64)
                    omap.insert(key=key, value=value)
                    model[key] = value
            elif roll < 0.65:
                assert omap.search(key=key) == model.get(key)
            elif roll < 0.8:
                assert omap.delete(key=key) == model.pop(key, None)
            elif key in model:
                new_value = rng.randbytes(64)
                assert omap.search(key=key, value=new_value) == model[key]
                model[key] = new_value
        for key in keyspace:
            assert omap.search(key=key) == model.get(key)
        assert omap.search(key=b"absent") is None
        assert sorted(pair for bucket in range(n) for pair in omap.search_group(bucket)) == sorted(model.items())
        assert not duplicate_blocks(omap)

    def test_undersized_upper_raises(self, client):
        n = 128
        too_small = PathOram(PathOramConfig(num_data=n, data_size=8, client=client, name="group_upper"))
        with pytest.raises(ValueError, match="too small"):
            GroupOmap(GroupOmapConfig(num_data=n, key_size=10, data_size=64, client=client), upper_oram=too_small)

    def test_with_init(self, client, duplicate_blocks):
        n = 128
        omap = _make_omap(client, n)
        init = [(b"%d" % i, b"%d" % (i * 7)) for i in range(n // 2)]
        omap.init_server_storage(data=init)
        model = dict(init)
        for i in range(n // 2, n):
            omap.insert(key=b"%d" % i, value=b"%d" % (i * 7))
            model[b"%d" % i] = b"%d" % (i * 7)
        with pytest.raises(ContractError, match="num_data=128"):
            omap.insert(key=b"extra", value=b"v")
        assert omap.delete(key=b"0") == b"0"
        omap.insert(key=b"extra", value=b"v")
        model[b"extra"] = b"v"
        del model[b"0"]
        for key, value in model.items():
            assert omap.search(key=key) == value
        assert not duplicate_blocks(omap)

    def test_rejects_contract_violations(self, client):
        omap = _make_omap(client)
        with pytest.raises(ContractError, match="GroupOmap 'group_omap-[0-9a-f]{8}'.*given twice"):
            omap.init_server_storage(data=[(b"k", b"lost"), (b"k", b"kept")])
        omap.init_server_storage(data=[(b"k", b"kept")])
        rounds = client.metrics.rounds

        cases: list[tuple[Any, Any]] = [("k", None), (b"x" * 11, None), (b"k", "v"), (b"k", b"x" * 65)]
        for key, value in cases:
            with pytest.raises(ContractError, match="GroupOmap 'group_omap-[0-9a-f]{8}'"):
                omap.search(key=key, value=value)
            with pytest.raises(ContractError, match="GroupOmap 'group_omap-[0-9a-f]{8}'"):
                omap.insert(key=key, value=value if value is not None else b"v")

        assert client.metrics.rounds == rounds
        assert omap.search(key=b"k") == b"kept"

    def test_write_deferral_preserves_access_sequence(self, assert_deferral_preserves_access):
        def workload(client):
            omap = _make_omap(client, 64)
            omap.init_server_storage(data=[(b"%d" % i, b"%d" % i) for i in range(0, 64, 2)])
            for i in range(1, 20, 2):
                omap.insert(key=b"%d" % i, value=b"%d" % i)
            for i in range(0, 30, 3):
                omap.search(key=b"%d" % i, value=b"%d" % (i + 1))

        rounds_off, rounds_on = assert_deferral_preserves_access(workload)
        assert rounds_on * 2 == rounds_off


@pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
def test_full_bucket_insert_leaves_the_map_usable(client, distinguishable):
    omap = _make_omap(client, n=64, distinguishable=distinguishable)
    pairs = [(b"key%d" % i, b"value%d" % i) for i in range(20)]
    omap.init_server_storage(pairs)
    bound = omap._upper_bound
    omap._upper_bound = 0
    for i in range(5):
        with pytest.raises(MemoryError, match="is full"):
            omap.insert(b"new%d" % i, b"x")
    omap._upper_bound = bound
    for key, value in pairs:
        assert omap.search(key) == value
    omap.insert(b"new0", b"x")
    assert omap.search(b"new0") == b"x"


def test_lower_paths_cover_the_whole_leaf_range(client):
    omap = _make_omap(client, n=100)
    leaf_range = omap._lower_oram._leaf_range
    leaves = {omap._slot_leaf(bytes([i % 256, i // 256]) * 16, 1) for i in range(4000)}
    assert leaf_range > 100
    assert max(leaves) >= 100
    assert leaves <= set(range(leaf_range))


@pytest.mark.parametrize(
    ("lower_method", "operation"),
    [
        ("_read_paths", lambda omap: omap.search(b"key3")),
        ("_read_paths", lambda omap: omap.search_group(omap._hash_key_to_bucket(b"key3"))),
        ("_read_paths", lambda omap: omap.insert(b"new", b"x")),
        ("_read_paths", lambda omap: omap.delete(b"key3")),
    ],
    ids=["search", "search_group", "insert", "delete"],
)
def test_a_failed_lower_access_still_finishes_the_upper_access(fail_once, client, lower_method, operation):
    omap = _make_omap(client, n=64)
    pairs = [(b"key%d" % i, b"value%d" % i) for i in range(20)]
    omap.init_server_storage(pairs)
    fail_once(omap._lower_oram, lower_method)

    with pytest.raises(RuntimeError, match="injected"):
        operation(omap)

    for key, value in pairs:
        assert omap.search(key) == value
    assert omap.search(b"new") is None
    omap.insert(b"new", b"x")
    assert omap.search(b"new") == b"x"


def test_search_group_returns_the_bucket_and_reshuffles_it(client):
    omap = _make_omap(client, n=64)
    pairs = [(b"key%d" % i, b"value%d" % i) for i in range(20)]
    omap.init_server_storage(pairs)
    bucket = omap._hash_key_to_bucket(b"key3")
    expected = sorted((key, value) for key, value in pairs if omap._hash_key_to_bucket(key) == bucket)

    assert sorted(omap.search_group(bucket)) == expected
    assert sorted(omap.search_group(bucket)) == expected
    for key, value in pairs:
        assert omap.search(key) == value


@pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
def test_an_insert_whose_write_back_fails_keeps_its_block_reachable(
    fail_once, monkeypatch, client, read_all_blocks, duplicate_blocks, distinguishable
):
    omap = _make_omap(client, n=64, distinguishable=distinguishable)
    omap.init_server_storage([(b"key%d" % i, b"value%d" % i) for i in range(20)])
    lower = omap._lower_oram
    stored = read_all_blocks(lower)[0]
    draws = [stored.leaf]
    fresh = lower._get_new_leaf
    monkeypatch.setattr(lower, "_get_new_leaf", lambda: draws.pop() if draws else fresh())
    fail_once(lower, "_write_paths")

    with pytest.raises(RuntimeError, match="injected"):
        omap.insert(b"new", b"first")

    assert omap.search(stored.key) == stored.value
    assert omap.search(b"new") == b"first"
    assert omap.search(b"new", b"second") == b"first"
    assert omap.search(b"new") == b"second"
    assert not duplicate_blocks(omap)


@pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
def test_every_lower_access_reads_a_fixed_number_of_paths(recorded_client, distinguishable):
    client, recorder = recorded_client()
    omap = _make_omap(client, n=64, distinguishable=distinguishable)
    pairs = [(b"key%d" % i, b"v") for i in range(30)]
    omap.init_server_storage(pairs)
    lower = omap._lower_oram._name
    width = omap._width
    empty = next(
        key
        for key in (b"x%d" % i for i in range(1000))
        if omap._hash_key_to_bucket(key) not in {omap._hash_key_to_bucket(each) for each, _ in pairs}
    )
    ops = {
        "search hit": (lambda: omap.search(b"key3"), width),
        "search empty bucket": (lambda: omap.search(empty), width),
        "search update": (lambda: omap.search(b"key4", b"w"), width),
        "search_group": (lambda: omap.search_group(omap._hash_key_to_bucket(b"key5")), width),
        "delete hit": (lambda: omap.delete(b"key6"), width),
        "delete miss": (lambda: omap.delete(empty), width),
        "insert": (lambda: omap.insert(b"fresh", b"v"), 1 if distinguishable else width),
    }
    for name, (op, expected) in ops.items():
        before = len(recorder.batches)
        op()
        reads = [len(read.leaves) for batch in recorder.batches[before:] for read in batch.reads if read.label == lower]
        assert reads == [expected], name
    assert omap.search(b"key4") == b"w"
    assert omap.search(b"fresh") == b"v"


@pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
def test_an_insert_that_fails_after_placing_its_block_is_neither_lost_nor_doubled(
    monkeypatch, client, duplicate_blocks, distinguishable
):
    omap = _make_omap(client, n=64, distinguishable=distinguishable)
    pairs = [(b"key%d" % i, b"value%d" % i) for i in range(20)]
    omap.init_server_storage(pairs)
    lower = omap._lower_oram
    check = lower._check_stash
    calls = []

    def fail_after_placing():
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("injected")
        check()

    monkeypatch.setattr(lower, "_check_stash", fail_after_placing)
    with pytest.raises(RuntimeError, match="injected"):
        omap.insert(b"new", b"x")
    monkeypatch.setattr(lower, "_check_stash", check)

    bucket_id = omap._hash_key_to_bucket(b"new")
    stored = omap._upper_oram.operate_on_key_without_eviction(key=bucket_id)
    omap._upper_oram.eviction_with_update_stash(key=bucket_id, value=stored)
    in_bucket = sum(omap._hash_key_to_bucket(key) == bucket_id for key, _ in pairs)
    assert omap._metadata.decode(stored)[1] == in_bucket + 1
    for key, value in pairs:
        assert omap.search(key) == value
    assert omap.search(b"new") == b"x"
    bucket = omap.search_group(omap._hash_key_to_bucket(b"new"))
    assert [key for key, _ in bucket].count(b"new") == 1
    assert not duplicate_blocks(omap)
