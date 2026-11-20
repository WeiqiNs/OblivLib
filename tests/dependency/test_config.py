import pytest

from oblivlib.dependency import (
    AvlOmapConfig,
    BPlusOmapConfig,
    DaOramConfig,
    FreecursiveOramConfig,
    OramOstOmapConfig,
    PathOramConfig,
    RecursiveOramConfig,
)

_ORAM = {"num_data": 16, "data_size": 10}
_OMAP = {**_ORAM, "key_size": 4}


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
        (OramOstOmapConfig, {"num_data": 16}, {"num_data": 0}, "num_data"),
    ],
)
def test_config_rejects_each_invalid_field(client, config_cls, valid, invalid, message):
    if config_cls is not OramOstOmapConfig:
        valid = {**valid, "client": client}
    config_cls(**valid)
    with pytest.raises(ValueError, match=message):
        config_cls(**{**valid, **invalid})
