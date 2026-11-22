import pytest

from oblivlib.dependency import (
    OramBlock,
    PathOramConfig,
    StashOverflowError,
)
from oblivlib.oram import PathOram


def test_default_names_are_unique_per_config(client):
    first = PathOramConfig(num_data=4, data_size=4, client=client)
    second = PathOramConfig(num_data=4, data_size=4, client=client)
    assert first.name.startswith("po-")
    assert first.name != second.name


def test_stash_overflow_raises(client):
    base = PathOram(PathOramConfig(num_data=4, data_size=4, stash_scale=1, client=client, name="tiny"))
    base._stash = [OramBlock(key=i, leaf=0, value=b"") for i in range(base._stash_capacity)]
    base._check_stash()

    base._stash.append(OramBlock(key=99, leaf=0, value=b""))
    with pytest.raises(StashOverflowError, match="PathOram 'tiny'") as raised:
        base._check_stash()
    assert isinstance(raised.value, MemoryError)


def test_require_in_stash_finds_the_block_or_names_the_missing_key(client):
    base = PathOram(PathOramConfig(num_data=4, data_size=4, client=client, name="stash"))
    block = OramBlock(key=1, leaf=0, value=b"")
    base._stash = [OramBlock(key=0, leaf=0, value=b""), block]
    assert base._require_in_stash(1) is block
    with pytest.raises(KeyError, match="Key 2 not found"):
        base._require_in_stash(2)
