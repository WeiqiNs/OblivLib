import random
import secrets
from typing import Any

import pytest

from oblivlib.dependency import (
    ContractError,
    DaOramConfig,
    FreecursiveOramConfig,
    PathOramConfig,
    RecursiveOramConfig,
)
from oblivlib.oram import DAOram, FreecursiveOram, PathOram, RecursivePathOram
from oblivlib.oram.tree_base_oram import pos_map_level_count

RECURSIVE_POS_MAP_ORAMS = [
    (RecursivePathOram, RecursiveOramConfig),
    (DAOram, DaOramConfig),
    (FreecursiveOram, FreecursiveOramConfig),
]


@pytest.mark.parametrize("n", [100, 1025])
def test_non_power_of_two_round_trip(make_oram, client, n):
    oram = make_oram(num_data=n, data_size=10, client=client)
    oram.init_server_storage()

    for i in range(n):
        oram.operate_on_key(key=i, value=b"%d" % i)

    for i in range(n):
        assert oram.operate_on_key(key=i) == b"%d" % i


def test_init_path_overflow_goes_to_stash(client):
    oram = PathOram(PathOramConfig(num_data=4, data_size=10, client=client, bucket_size=1))
    oram._pos_map = dict.fromkeys(range(4), 0)
    oram.init_server_storage(data={i: b"v%d" % i for i in range(4)})

    assert [data.key for data in oram._stash] == [3]
    for i in range(4):
        assert oram.operate_on_key(key=i) == b"v%d" % i


def test_init_accepts_one_shot_stream(make_oram, client, test_file, encryptor):
    oram = make_oram(num_data=64, data_size=10, client=client, encryptor=encryptor, build_file=test_file)
    oram.init_server_storage(data=((key, b"v%d" % key) for key in range(64) if key % 3))

    assert not test_file.exists()
    for key in range(64):
        assert oram.operate_on_key(key=key) == (b"v%d" % key if key % 3 else b"")


@pytest.mark.parametrize(
    ("pairs", "match"),
    [
        ([(1, b"a"), (1, b"b")], "given twice"),
        ([(64, b"a")], r"\[0, 64\)"),
        ([(-1, b"a")], r"\[0, 64\)"),
        ([(1, "a")], "must be bytes"),
        ([(1, b"x" * 11)], "data_size is 10"),
    ],
)
def test_init_rejects_duplicate_keys_and_contract_violations(client, pairs, match):
    oram = PathOram(PathOramConfig(num_data=64, data_size=10, client=client))
    with pytest.raises(ContractError, match=match):
        oram.init_server_storage(data=iter(pairs))


@pytest.mark.parametrize(("cls", "config_cls"), RECURSIVE_POS_MAP_ORAMS)
def test_num_data_at_most_on_chip_raises(cls, config_cls, client):
    with pytest.raises(ValueError):
        cls(config_cls(num_data=8, data_size=10, client=client))


@pytest.mark.parametrize(
    ("num_data", "on_chip", "ratio", "expected"),
    [(40, 10, 4, 1), (41, 10, 4, 2), (125, 1, 5, 3)],
)
def test_pos_map_level_count_is_exact(num_data, on_chip, ratio, expected):
    assert pos_map_level_count(num_data=num_data, on_chip=on_chip, ratio=ratio) == expected


def test_da_oram_reset_block_left_in_pos_map_stash(monkeypatch, client):
    monkeypatch.setattr(secrets, "randbelow", random.Random(9).randrange)
    oram = DAOram(
        DaOramConfig(num_data=1024, data_size=8, client=client, bucket_size=1, stash_scale=1000, prf_key=b"\x00" * 32)
    )
    model = {i: b"%d" % i for i in range(1024)}
    oram.init_server_storage(data=dict(model))

    workload = random.Random(9)
    for step in range(3000):
        key = workload.randrange(1024)
        assert oram.operate_on_key(key=key, value=b"%d" % step) == model[key]
        model[key] = b"%d" % step


def test_rejects_non_bytes_oversized_and_out_of_range(make_oram, client):
    oram = make_oram(num_data=64, data_size=10, client=client)
    oram.init_server_storage()
    oram.operate_on_key(key=5, value=b"kept")
    rounds = client.metrics.rounds

    cases: list[tuple[Any, Any]] = [
        (5, "hello"),
        (5, b"x" * 11),
        (5, None),
        (-1, b"v"),
        (64, b"v"),
        ("0", b"v"),
        (True, b"v"),
    ]
    for key, value in cases:
        with pytest.raises(ContractError, match=f"{type(oram).__name__} '"):
            oram.operate_on_key(key=key, value=value)

    assert client.metrics.rounds == rounds
    assert oram.operate_on_key(key=5) == b"kept"


def test_new_access_while_one_is_pending_raises(make_oram, client):
    oram = make_oram(num_data=64, data_size=10, client=client)
    oram.init_server_storage({3: b"three"})
    assert oram.operate_on_key_without_eviction(3) == b"three"
    with pytest.raises(ValueError, match="still waiting for its eviction"):
        oram.operate_on_key(4)
    with pytest.raises(ValueError, match="still waiting for its eviction"):
        oram.operate_on_key_without_eviction(4)
    oram.eviction_with_update_stash(3, b"tres")
    assert oram.operate_on_key(3) == b"tres"


@pytest.mark.parametrize(("cls", "config_cls"), [(DAOram, DaOramConfig), (FreecursiveOram, FreecursiveOramConfig)])
@pytest.mark.parametrize("num_data", [4096, 100_000])
def test_position_map_leaves_are_independent_of_data_leaves(cls, config_cls, num_data, client):
    oram = cls(config_cls(num_data=num_data, data_size=10, client=client))
    level = oram._levels[-1]
    leaf_range = 2 ** (level._level - 1)
    matches = sum(
        level._prf_leaf(key=key, gc=0, ic=0, level=level._level, num_data=level._num_data)
        == oram._prf_leaf(key=key, gc=0, ic=0, level=oram._level, num_data=oram._num_data) % leaf_range
        for key in range(level._num_data)
    )
    assert matches <= max(3, 3 * level._num_data // leaf_range)
