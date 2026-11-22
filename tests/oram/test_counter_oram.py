from typing import Any

import pytest

from oblivlib.dependency import CapacityError, DaOramConfig, FreecursiveOramConfig, config
from oblivlib.oram import DAOram, FreecursiveOram

_TINY_COUNTERS: dict[str, Any] = {"num_data": 64, "data_size": 4}


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
