from typing import Any

import pytest

from oblivlib.dependency import (
    AesGcm,
    DaOramConfig,
    FreecursiveOramConfig,
    MulPathOramConfig,
    PathOramConfig,
    RecursiveOramConfig,
    StaticOramConfig,
)
from oblivlib.oram import DAOram, FreecursiveOram, MulPathOram, PathOram, RecursivePathOram, StaticOram
from oblivlib.oram.tree_base_oram import pos_map_level_count

RECURSIVE_POS_MAP_ORAMS = [
    (RecursivePathOram, RecursiveOramConfig),
    (DAOram, DaOramConfig),
    (FreecursiveOram, FreecursiveOramConfig),
]


@pytest.mark.parametrize(("cls", "config_cls"), RECURSIVE_POS_MAP_ORAMS)
def test_num_data_at_most_on_chip_mem_is_refused(cls, config_cls, client):
    cls(config_cls(num_data=11, data_size=10, client=client))
    with pytest.raises(ValueError, match=r"num_data \(8\) must be greater than"):
        cls(config_cls(num_data=8, data_size=10, client=client))


@pytest.mark.parametrize(
    ("num_data", "on_chip", "ratio", "expected"),
    [(40, 10, 4, 1), (41, 10, 4, 2), (125, 1, 5, 3)],
)
def test_pos_map_level_count_is_exact(num_data, on_chip, ratio, expected):
    assert pos_map_level_count(num_data=num_data, on_chip=on_chip, ratio=ratio) == expected


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
