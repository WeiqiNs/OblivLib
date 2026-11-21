import random
from typing import Any

import pytest

from oblivlib.dependency import (
    AesGcm,
    AvlOmapConfig,
    BPlusOmapConfig,
    ContractError,
    DaOramConfig,
    PathOramConfig,
    RecursiveOramConfig,
)
from oblivlib.omap import AVLOmap, BPlusOmap, OramOstOmap, OramOstOmapConfig
from oblivlib.oram import DAOram, PathOram, RecursivePathOram


def _make_ods(kind, n, client, encryptor=None):
    if kind == "avl":
        return AVLOmap(AvlOmapConfig(num_data=n, key_size=10, data_size=10, client=client, encryptor=encryptor))
    return BPlusOmap(
        BPlusOmapConfig(
            order=40,
            num_data=n,
            key_size=10,
            data_size=10,
            client=client,
            encryptor=encryptor,
        )
    )


def _make_oram(kind, n, client, ost, encryptor=None):
    data_size = OramOstOmap.oram_data_size(ost)
    if kind == "da":
        return DAOram(DaOramConfig(num_data=n, data_size=data_size, client=client, encryptor=encryptor))
    if kind == "path":
        return PathOram(PathOramConfig(num_data=n, data_size=data_size, client=client, encryptor=encryptor))
    return RecursivePathOram(RecursiveOramConfig(num_data=n, data_size=data_size, client=client, encryptor=encryptor))


def _make_omap(ods_kind, oram_kind, client, n=64):
    ost = _make_ods(ods_kind, n, client)
    return OramOstOmap(OramOstOmapConfig(num_data=n), ost=ost, oram=_make_oram(oram_kind, n, client, ost))


class TestOramOstOmapOracle:
    @pytest.mark.parametrize("ods_kind", ["avl", "bplus"])
    @pytest.mark.parametrize("oram_kind", ["da", "path", "recursive"])
    def test_oracle(self, ods_kind, oram_kind, client, duplicate_blocks):
        n = 64
        omap = _make_omap(ods_kind, oram_kind, client, n)
        omap.init_server_storage()
        rng = random.Random(f"{ods_kind}-{oram_kind}")
        model = {}
        keyspace = [b"k%d" % i for i in range(n)]
        for _ in range(n * 4):
            key = rng.choice(keyspace)
            roll = rng.random()
            if roll < 0.45:
                if key not in model:
                    value = b"v%d" % rng.randint(0, 10**6)
                    omap.insert(key=key, value=value)
                    model[key] = value
            elif roll < 0.6:
                assert omap.delete(key=key) == model.pop(key, None)
            else:
                assert omap.search(key=key) == model.get(key)
        for key in keyspace:
            assert omap.search(key=key) == model.get(key)
        assert not duplicate_blocks(omap)


class TestOramOstOmapInit:
    @pytest.mark.parametrize("ods_kind", ["avl", "bplus"])
    @pytest.mark.parametrize("oram_kind", ["da", "path", "recursive"])
    def test_with_init(self, ods_kind, oram_kind, client):
        n = 64
        omap = _make_omap(ods_kind, oram_kind, client, n)
        keys = [b"k%d" % i for i in range(n)]
        values = [b"v%d" % i for i in range(n)]

        omap.init_server_storage(data=list(zip(keys[: n // 2], values[: n // 2], strict=True)))
        for key, value in zip(keys[n // 2 :], values[n // 2 :], strict=True):
            omap.insert(key=key, value=value)

        for key, value in zip(keys, values, strict=True):
            assert omap.search(key=key) == value

    @pytest.mark.parametrize("ods_kind", ["avl", "bplus"])
    def test_encryption_round_trip(self, ods_kind, client, encryptor):
        n = 64
        ost = _make_ods(ods_kind, n, client, encryptor=encryptor)
        omap = OramOstOmap(
            OramOstOmapConfig(num_data=n), ost=ost, oram=_make_oram("da", n, client, ost, encryptor=AesGcm())
        )
        omap.init_server_storage()
        keys = [bytes([i]) * 10 for i in range(n)]
        for key in keys:
            omap.insert(key=key, value=key)
        for key in keys:
            assert omap.search(key=key) == key


def test_undersized_oram_raises(client):
    ost = _make_ods("avl", 64, client)
    oram = PathOram(PathOramConfig(num_data=64, data_size=OramOstOmap.oram_data_size(ost) - 1, client=client))
    with pytest.raises(ValueError, match="too small"):
        OramOstOmap(OramOstOmapConfig(num_data=64), ost=ost, oram=oram)


def test_rejects_contract_violations(client):
    omap = _make_omap("avl", "path", client)
    with pytest.raises(ContractError, match="OramOstOmap 'avl-[0-9a-f]{8}'.*given twice"):
        omap.init_server_storage(data=[(b"k", b"lost"), (b"k", b"kept")])
    omap.init_server_storage(data=[(b"k", b"kept")])
    rounds = client.metrics.rounds

    cases: list[tuple[Any, Any]] = [("k", b"v"), (b"x" * 11, b"v"), (b"k", "v"), (b"k", b"x" * 11)]
    for key, value in cases:
        with pytest.raises(ContractError, match="OramOstOmap 'avl-[0-9a-f]{8}'"):
            omap.search(key=key, value=value)
        with pytest.raises(ContractError, match="OramOstOmap 'avl-[0-9a-f]{8}'"):
            omap.insert(key=key, value=value)

    assert client.metrics.rounds == rounds
    assert omap.search(key=b"k") == b"kept"


def test_write_deferral_preserves_access_sequence(assert_deferral_preserves_access):
    def workload(client):
        omap = _make_omap("avl", "da", client)
        omap.init_server_storage(data=[(b"%d" % i, b"%d" % i) for i in range(0, 64, 2)])
        for i in range(1, 20, 2):
            omap.insert(key=b"%d" % i, value=b"%d" % i)
        for i in range(0, 30, 3):
            omap.search(key=b"%d" % i)

    rounds_off, rounds_on = assert_deferral_preserves_access(workload)
    assert rounds_on < rounds_off


def test_a_failed_tree_op_still_finishes_the_oram_access(fail_once, client):
    omap = _make_omap("avl", "path", client)
    omap.init_server_storage(data=[(b"k%d" % i, b"v%d" % i) for i in range(20)])
    fail_once(omap._ost, "_insert")

    with pytest.raises(RuntimeError, match="injected"):
        omap.insert(key=b"new", value=b"x")

    for i in range(20):
        assert omap.search(key=b"k%d" % i) == b"v%d" % i
    omap.insert(key=b"new", value=b"x")
    assert omap.search(key=b"new") == b"x"


def test_an_insert_past_num_data_is_refused_before_any_access(client):
    omap = _make_omap("avl", "path", client, n=16)
    omap.init_server_storage()
    for i in range(16):
        omap.insert(key=b"%d" % i, value=b"v")
    rounds = client.metrics.rounds
    with pytest.raises(ContractError, match="num_data=16"):
        omap.insert(key=b"16", value=b"v")
    assert client.metrics.rounds == rounds
    assert [omap.search(key=b"%d" % i) for i in range(16)] == [b"v"] * 16


def test_the_map_enforces_its_own_num_data_not_the_inner_maps(client):
    ost = _make_ods("avl", 64, client)
    omap = OramOstOmap(OramOstOmapConfig(num_data=16), ost=ost, oram=_make_oram("path", 16, client, ost))
    with pytest.raises(ContractError, match="OramOstOmap .*exceed num_data=16"):
        omap.init_server_storage([(b"%d" % i, b"v") for i in range(17)])
    omap.init_server_storage([(b"%d" % i, b"v") for i in range(15)])
    omap.insert(key=b"15", value=b"v")
    with pytest.raises(ContractError, match="OramOstOmap .*num_data=16"):
        omap.insert(key=b"16", value=b"v")
    assert omap.delete(key=b"0") == b"v"
    omap.insert(key=b"16", value=b"v")
