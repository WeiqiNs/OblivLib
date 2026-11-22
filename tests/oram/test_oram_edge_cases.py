import random
import secrets
from typing import Any

import pytest

from oblivlib.dependency import (
    AesGcm,
    CapacityError,
    DaOramConfig,
    FreecursiveOramConfig,
    MulPathOramConfig,
    PathOramConfig,
    RecursiveOramConfig,
    StaticOramConfig,
    config,
)
from oblivlib.oram import DAOram, FreecursiveOram, MulPathOram, PathOram, RecursivePathOram, StaticOram
from oblivlib.oram.tree_base_oram import pos_map_level_count

_TINY_COUNTERS: dict[str, Any] = {"num_data": 64, "data_size": 4}

RECURSIVE_POS_MAP_ORAMS = [
    (RecursivePathOram, RecursiveOramConfig),
    (DAOram, DaOramConfig),
    (FreecursiveOram, FreecursiveOramConfig),
]


def test_init_path_overflow_goes_to_stash(client):
    oram = PathOram(PathOramConfig(num_data=4, data_size=10, client=client, bucket_size=1))
    oram._pos_map = dict.fromkeys(range(4), 0)
    oram.init_server_storage(data={i: b"v%d" % i for i in range(4)})

    assert [data.key for data in oram._stash] == [3]
    for i in range(4):
        assert oram.operate_on_key(key=i) == b"v%d" % i


@pytest.mark.parametrize(("cls", "config_cls"), RECURSIVE_POS_MAP_ORAMS)
def test_num_data_at_most_on_chip_raises(cls, config_cls, client):
    cls(config_cls(num_data=11, data_size=10, client=client))
    with pytest.raises(ValueError, match=r"num_data \(8\) must be greater than"):
        cls(config_cls(num_data=8, data_size=10, client=client))


def test_recursive_pos_map_compressed(num_data, client):
    oram = RecursivePathOram(RecursiveOramConfig(num_data=num_data, data_size=10, client=client))
    oram.init_server_storage()
    assert len(oram._pos_map) <= 10


@pytest.mark.parametrize(
    ("num_data", "on_chip", "ratio", "expected"),
    [(40, 10, 4, 1), (41, 10, 4, 2), (125, 1, 5, 3)],
)
def test_pos_map_level_count_is_exact(num_data, on_chip, ratio, expected):
    assert pos_map_level_count(num_data=num_data, on_chip=on_chip, ratio=ratio) == expected


def test_static_oram_never_samples_a_random_leaf(monkeypatch, client):
    config = StaticOramConfig(num_data=64, data_size=10, client=client)

    def no_random_leaf(_: int) -> int:
        raise AssertionError("StaticOram drew a random leaf.")

    monkeypatch.setattr(secrets, "randbelow", no_random_leaf)
    oram = StaticOram(config)
    oram.init_server_storage(data={3: b"three"})
    assert oram.operate_on_key(3, b"tres") == b"three"
    assert oram.operate_on_key(3) == b"tres"


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


def test_da_oram_resets_in_a_partly_filled_last_block(client):
    oram = DAOram(DaOramConfig(num_data=102, data_size=8, client=client, num_ic=4, ic_length=2, on_chip_mem=4))
    oram.init_server_storage()
    model: dict[int, bytes] = {}
    workload = random.Random(11)
    for step in range(1500):
        key = workload.choice([96, 97, 98, 99, 100, 101])
        assert oram.operate_on_key(key=key, value=b"%d" % step) == model.get(key, b"")
        model[key] = b"%d" % step


@pytest.mark.parametrize(
    "make",
    [
        lambda client: DAOram(DaOramConfig(num_ic=2, ic_length=1, gc_length=2, **_TINY_COUNTERS, client=client)),
        lambda client: FreecursiveOram(
            FreecursiveOramConfig(
                num_ic=2, ic_length=1, gc_length=2, reset_method="hard", **_TINY_COUNTERS, client=client
            )
        ),
        lambda client: FreecursiveOram(
            FreecursiveOramConfig(num_ic=2, ic_length=1, gc_length=2, reset_prob=1.0, **_TINY_COUNTERS, client=client)
        ),
    ],
    ids=["da", "freecursive_hard", "freecursive_prob"],
)
def test_a_group_counter_that_overflows_its_field_raises_capacity_error(monkeypatch, client, make):
    monkeypatch.setattr(config, "MIN_GC_LENGTH", 1)
    oram = make(client)
    oram.init_server_storage()
    for step in range(3):
        assert oram.operate_on_key(key=0, value=b"%d" % step) == (b"" if step == 0 else b"%d" % (step - 1))
    with pytest.raises(CapacityError, match="bit field 0"):
        for _ in range(500):
            oram.operate_on_key(key=0)


_ROW_LAYOUT_CONFIG: dict[str, Any] = {"num_data": 64, "data_size": 16, "encryptor": AesGcm()}


@pytest.mark.parametrize(
    ("make", "expected_rows"),
    [
        pytest.param(lambda **kw: PathOram(PathOramConfig(**kw)), [105], id="path"),
        pytest.param(lambda **kw: StaticOram(StaticOramConfig(**kw)), [105], id="static"),
        pytest.param(lambda **kw: MulPathOram(MulPathOramConfig(**kw)), [105], id="mul_path"),
        pytest.param(lambda **kw: RecursivePathOram(RecursiveOramConfig(**kw)), [105, 53, 53], id="recursive"),
        pytest.param(lambda **kw: DAOram(DaOramConfig(num_ic=4, **kw)), [105, 85], id="da"),
        pytest.param(lambda **kw: FreecursiveOram(FreecursiveOramConfig(num_ic=4, **kw)), [105, 89], id="freecursive"),
    ],
)
def test_row_size_is_the_declared_block_layout_of_every_tree(make, expected_rows, client):
    oram = make(client=client, **_ROW_LAYOUT_CONFIG)
    assert [oram._cipher.row_bytes] + [level._cipher.row_bytes for level in oram._pos_map_levels()] == expected_rows
    oram.init_server_storage({key: b"\xff" * 16 for key in range(64)})
    for key in (0, 63, 0):
        assert oram.operate_on_key(key=key) == b"\xff" * 16


@pytest.mark.parametrize(("cls", "config_cls"), [(DAOram, DaOramConfig), (FreecursiveOram, FreecursiveOramConfig)])
@pytest.mark.parametrize("num_data", [4096, 100_000])
def test_position_map_leaves_are_independent_of_data_leaves(cls, config_cls, num_data, client):
    oram = cls(config_cls(num_data=num_data, data_size=10, client=client, prf_key=bytes(32)))
    level = oram._levels[-1]
    leaf_range = 2 ** (level._level - 1)
    matches = sum(
        level._own_counters.leaf(key, gc=0, ic=0) == oram._own_counters.leaf(key, gc=0, ic=0) % leaf_range
        for key in range(level.num_data)
    )
    assert matches <= max(3, 3 * level.num_data // leaf_range)
