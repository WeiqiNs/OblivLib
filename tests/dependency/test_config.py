from typing import Any

import pytest

from oblivlib.dependency import (
    AccessBudget,
    AvlOmapConfig,
    BPlusOmapConfig,
    DaOramConfig,
    FreecursiveOramConfig,
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
        (DaOramConfig, _ORAM, {"num_ic": 8, "ic_length": 2}, r"2\*\*ic_length"),
        (FreecursiveOramConfig, _ORAM, {"reset_prob": 0.0}, "reset_prob"),
        (FreecursiveOramConfig, _ORAM, {"reset_prob": 1.5}, "reset_prob"),
        (FreecursiveOramConfig, _ORAM, {"reset_method": "soft"}, "reset_method"),
        (AvlOmapConfig, _OMAP, {"key_size": 0}, "key_size"),
        (BPlusOmapConfig, {**_OMAP, "order": 3}, {"order": 2}, "order"),
        (AvlOmapConfig, _OMAP, {"insert_budget": AccessBudget(1, 9)}, r"a must be >= 2"),
        (AvlOmapConfig, _OMAP, {"delete_budget": AccessBudget(3, -2)}, r"a \+ b >= 2"),
        (BPlusOmapConfig, {**_OMAP, "order": 3}, {"delete_budget": AccessBudget(1, 9)}, r"a must be >= 2"),
        (
            BPlusOmapConfig,
            {**_OMAP, "order": 3, "paths_per_round": 1},
            {"delete_budget": AccessBudget(3, 9)},
            r"a must be >= 4",
        ),
        (BPlusOmapConfig, {**_OMAP, "order": 3}, {"paths_per_round": 3}, "paths_per_round must be 1 or 2"),
        (AvlOmapConfig, _OMAP, {"paths_per_round": 2}, "paths_per_round"),
        (AvlOmapConfig, {**_OMAP, "cached": True}, {"delete_budget": AccessBudget(1, 9)}, r"a must be >= 3/2"),
        (AvlOmapConfig, _OMAP, {"batch_size": 0}, "batch_size must be >= 1"),
        (AvlOmapConfig, _OMAP, {"batch_size": 17}, "exceeds the 16 leaves"),
    ],
)
def test_config_rejects_each_invalid_field(client, config_cls, valid, invalid, message):
    valid = {**valid, "client": client}
    config_cls(**valid)
    with pytest.raises(ValueError, match=message):
        config_cls(**{**valid, **invalid})


def test_a_budget_override_may_raise_a_minimum_but_keeps_the_others(client):
    config = AvlOmapConfig(**_OMAP, client=client, delete_budget=AccessBudget(3, -1), search_budget=AccessBudget(1, 5))
    assert [config.budget(op).rounds(10) for op in ("search", "insert", "delete")] == [15, 20, 29]
    bplus = BPlusOmapConfig(**_OMAP, client=client, order=3)
    assert [bplus.budget(op).rounds(10) for op in ("search", "insert", "delete")] == [10, 19, 19]
    cached_bplus = [BPlusOmapConfig(**_OMAP, client=client, order=3, cached=True, paths_per_round=p) for p in (2, 1)]
    assert [[c.budget(op).rounds(10) for op in ("search", "insert", "delete")] for c in cached_bplus] == [
        [10, 10, 10],
        [10, 19, 19],
    ]
    batched = [
        BPlusOmapConfig(**_OMAP, client=client, order=3, cached=True, paths_per_round=p, batch_size=k)
        for p, k in ((2, 4), (2, 5), (1, 2))
    ]
    assert [[c.budget(op).rounds(10) for op in ("search", "insert", "delete")] for c in batched] == [
        [10, 10, 10],
        [10, 19, 10],
        [10, 28, 19],
    ]
    cached = AvlOmapConfig(**_OMAP, client=client, cached=True)
    assert [cached.budget(op).rounds(height) for op in ("search", "insert", "delete") for height in (7, 8)] == [
        7,
        8,
        8,
        9,
        10,
        11,
    ]
