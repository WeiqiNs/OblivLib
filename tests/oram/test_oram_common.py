import random

import pytest

from oblivlib.dependency import UNSET, DaOramConfig, FreecursiveOramConfig, PathOramConfig, RecursiveOramConfig
from oblivlib.oram import DAOram, FreecursiveOram, PathOram, RecursivePathOram


class TestOramCommon:
    def test_with_init(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage(data={i: b"%d" % (i * 2) for i in range(num_data)})

        for i in range(num_data):
            assert oram.operate_on_key(key=i) == b"%d" % (i * 2)

    def test_operations_match_a_dict_model(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage()
        model = dict.fromkeys(range(num_data), b"")
        rng = random.Random(4)

        key = 0
        for step in range(num_data * 4):
            key = key if rng.random() < 0.25 else rng.randrange(num_data)
            value = rng.choice([UNSET, b"", b"%d" % step])
            assert oram.operate_on_key(key, value) == model[key]
            if isinstance(value, bytes):
                model[key] = value

    def test_operate_then_evict(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key_without_eviction(key=i)
            oram.eviction_with_update_stash(key=i, value=b"%d" % i)

        for i in range(num_data):
            assert oram.operate_on_key(key=i) == b"%d" % i

    def test_split_access_matches_fused_access(self, make_oram, seeded_access):
        def run(split):
            def workload(client):
                oram = make_oram(num_data=64, data_size=10, client=client)
                oram.init_server_storage()
                rng = random.Random(8)
                for step in range(80):
                    key = rng.randrange(64)
                    if split:
                        oram.operate_on_key_without_eviction(key)
                        oram.eviction_with_update_stash(key, b"%d" % step)
                    else:
                        oram.operate_on_key(key, b"%d" % step)

            return seeded_access(workload)[0]

        assert run(split=True) == run(split=False)

    def test_write_deferral_preserves_access_sequence(self, make_oram, assert_deferral_preserves_access):
        def workload(client):
            oram = make_oram(num_data=64, data_size=10, client=client)
            oram.init_server_storage()
            rng = random.Random(5)
            for step in range(60):
                oram.operate_on_key(key=rng.randrange(64), value=b"%d" % step if step % 2 else UNSET)
            oram.operate_on_key_without_eviction(key=3)
            oram.eviction_with_update_stash(key=3, value=b"1")

        rounds_off, rounds_on = assert_deferral_preserves_access(workload)
        assert rounds_on * 2 == rounds_off


@pytest.mark.parametrize(
    ("make", "level_count"),
    [
        (lambda client: RecursivePathOram(RecursiveOramConfig(num_data=256, data_size=10, client=client)), 3),
        (lambda client: DAOram(DaOramConfig(num_data=256, data_size=10, client=client, num_ic=4)), 2),
        (lambda client: FreecursiveOram(FreecursiveOramConfig(num_data=256, data_size=10, client=client, num_ic=4)), 2),
    ],
    ids=["recursive", "da", "freecursive"],
)
def test_max_stash_adds_every_level_peak(make, level_count, client):
    oram = make(client)
    oram.init_server_storage()
    for key in range(0, 256, 9):
        oram.operate_on_key(key, b"v")

    assert len(oram._levels) == level_count
    assert all(level._max_stash >= 1 for level in oram._levels)
    assert oram.max_stash > oram._max_stash


def test_eviction_without_a_pending_access_raises(client):
    oram = PathOram(PathOramConfig(num_data=64, data_size=10, client=client))
    oram.init_server_storage()
    oram.operate_on_key_without_eviction(1)
    oram.eviction_with_update_stash(1, b"a")

    with pytest.raises(ValueError, match="no access is waiting"):
        oram.eviction_with_update_stash(1, b"b")
    assert oram.operate_on_key(1) == b"a"


def test_recursive_pos_map_compressed(num_data, client):
    oram = RecursivePathOram(RecursiveOramConfig(num_data=num_data, data_size=10, client=client))
    oram.init_server_storage()
    assert len(oram._pos_map) <= 10


_RESETTING_COUNTER_ORAMS = pytest.mark.parametrize(
    "make_counter_oram",
    [
        lambda client: DAOram(DaOramConfig(num_data=256, data_size=10, client=client, num_ic=4, ic_length=2)),
        lambda client: FreecursiveOram(
            FreecursiveOramConfig(num_data=256, data_size=10, client=client, num_ic=4, ic_length=4, reset_prob=0.5)
        ),
        lambda client: FreecursiveOram(
            FreecursiveOramConfig(num_data=256, data_size=10, client=client, num_ic=4, ic_length=2, reset_method="hard")
        ),
    ],
    ids=["da", "freecursive_prob", "freecursive_hard"],
)


@_RESETTING_COUNTER_ORAMS
def test_write_deferral_preserves_counter_resets(make_counter_oram, assert_deferral_preserves_access):
    def workload(client):
        oram = make_counter_oram(client)
        oram.init_server_storage()
        rng = random.Random(6)
        for step in range(120):
            oram.operate_on_key(key=rng.randrange(256), value=b"%d" % step)
        for key in range(4):
            oram.operate_on_key_without_eviction(key=key)
            oram.eviction_with_update_stash(key=key, value=b"%d" % key)

    assert_deferral_preserves_access(workload)


@_RESETTING_COUNTER_ORAMS
@pytest.mark.parametrize("defer_writes", [True, False])
def test_split_access_matches_fused_access_through_counter_resets(make_counter_oram, defer_writes, seeded_access):
    def run(split):
        def workload(client):
            oram = make_counter_oram(client)
            oram.init_server_storage()
            rng = random.Random(9)
            for step in range(200):
                key = rng.randrange(16)
                if split:
                    oram.operate_on_key_without_eviction(key)
                    oram.eviction_with_update_stash(key, b"%d" % step)
                else:
                    oram.operate_on_key(key, b"%d" % step)

        return seeded_access(workload, defer_writes=defer_writes)

    assert run(split=True) == run(split=False)
