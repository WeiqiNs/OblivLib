from typing import Any

import pytest

from oblivlib.dependency import (
    AccessBudget,
    AvlOmapConfig,
    BPlusOmapConfig,
    DaOramConfig,
    FreecursiveOramConfig,
    OramOstOmapConfig,
    PathOramConfig,
    RecursiveOramConfig,
)

_ORAM: dict[str, Any] = {"num_data": 16, "data_size": 10}
_OMAP: dict[str, Any] = {**_ORAM, "key_size": 4}


@pytest.mark.parametrize(
    ("config_cls", "valid", "invalid", "message"),
    [
        (PathOramConfig, _ORAM, {"num_data": 0}, "num_data"),
        (PathOramConfig, _ORAM, {"data_size": 0}, "data_size"),
        (PathOramConfig, _ORAM, {"bucket_size": 0}, "bucket_size"),
        (PathOramConfig, _ORAM, {"stash_scale": 0}, "stash_scale"),
        (PathOramConfig, _ORAM, {"build_file": "build.tree"}, "build_file requires an encryptor"),
        (RecursiveOramConfig, _ORAM, {"on_chip_mem": 0}, "on_chip_mem"),
        (RecursiveOramConfig, _ORAM, {"compression_ratio": 1}, "compression_ratio"),
        (DaOramConfig, _ORAM, {"num_ic": 1}, "num_ic must be >= 2"),
        (DaOramConfig, _ORAM, {"ic_length": 0}, "ic_length"),
        (DaOramConfig, _ORAM, {"gc_length": 63}, "gc_length must be >= 64"),
        (FreecursiveOramConfig, _ORAM, {"gc_length": 63}, "gc_length must be >= 64"),
        (DaOramConfig, _ORAM, {"on_chip_mem": 0}, "on_chip_mem"),
        (DaOramConfig, _ORAM, {"num_ic": 8, "ic_length": 2}, r"2\*\*ic_length"),
        (FreecursiveOramConfig, _ORAM, {"on_chip_size": 0}, "on_chip_size"),
        (FreecursiveOramConfig, _ORAM, {"reset_prob": 0.0}, "reset_prob"),
        (FreecursiveOramConfig, _ORAM, {"reset_prob": 1.5}, "reset_prob"),
        (FreecursiveOramConfig, _ORAM, {"reset_method": "soft"}, "reset_method"),
        (AvlOmapConfig, _OMAP, {"key_size": 0}, "key_size"),
        (BPlusOmapConfig, {**_OMAP, "order": 3}, {"order": 2}, "order"),
        (AvlOmapConfig, _OMAP, {"insert_budget": AccessBudget(1, 9)}, r"a must be >= 2"),
        (AvlOmapConfig, _OMAP, {"delete_budget": AccessBudget(3, -2)}, r"a \+ b >= 2"),
        (BPlusOmapConfig, {**_OMAP, "order": 3}, {"delete_budget": AccessBudget(3, 9)}, r"a must be >= 4"),
        (AvlOmapConfig, {**_OMAP, "cached": True}, {"delete_budget": AccessBudget(1, 9)}, r"a must be >= 3/2"),
        (BPlusOmapConfig, {**_OMAP, "order": 3}, {"cached": True}, "cached"),
        (AvlOmapConfig, _OMAP, {"batch_size": 0}, "batch_size must be >= 1"),
        (BPlusOmapConfig, {**_OMAP, "order": 3}, {"batch_size": 2}, "batch_size"),
        (OramOstOmapConfig, {"num_data": 16}, {"num_data": 0}, "num_data"),
    ],
)
def test_config_rejects_each_invalid_field(client, config_cls, valid, invalid, message):
    if config_cls is not OramOstOmapConfig:
        valid = {**valid, "client": client}
    config_cls(**valid)
    with pytest.raises(ValueError, match=message):
        config_cls(**{**valid, **invalid})


def test_a_budget_override_may_raise_a_minimum_but_keeps_the_others(client):
    config = AvlOmapConfig(**_OMAP, client=client, delete_budget=AccessBudget(3, -1), search_budget=AccessBudget(1, 5))
    assert [config.budget(op).rounds(10) for op in ("search", "insert", "delete")] == [15, 20, 29]
    textbook = BPlusOmapConfig.textbook(**_OMAP, client=client, order=3)
    assert [textbook.budget(op).rounds(10) for op in ("search", "insert", "delete")] == [20, 30, 37]
    cached = AvlOmapConfig(**_OMAP, client=client, cached=True)
    assert [cached.budget(op).rounds(height) for op in ("search", "insert", "delete") for height in (7, 8)] == [
        7,
        8,
        8,
        9,
        10,
        11,
    ]
