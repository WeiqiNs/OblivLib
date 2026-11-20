import pytest

from oblivlib.dependency import (
    Data,
    PathOramConfig,
    StashOverflowError,
)
from oblivlib.dependency.tree_storage_base import TreeStorageBase


def test_default_names_are_unique_per_config(client):
    first = PathOramConfig(num_data=4, data_size=4, client=client)
    second = PathOramConfig(num_data=4, data_size=4, client=client)
    assert first.name.startswith("po-")
    assert first.name != second.name


def test_stash_overflow_raises(client):
    base = TreeStorageBase(PathOramConfig(num_data=4, data_size=4, stash_scale=1, client=client, name="tiny"))
    base._stash = [Data(key=i, leaf=0) for i in range(base._stash_capacity)]
    base._check_stash()

    base._stash.append(Data(key=99, leaf=0))
    with pytest.raises(StashOverflowError, match="TreeStorageBase 'tiny'") as raised:
        base._check_stash()
    assert isinstance(raised.value, MemoryError)


def test_require_in_stash_finds_the_block_or_names_the_missing_key(client):
    base = TreeStorageBase(PathOramConfig(num_data=4, data_size=4, client=client, name="stash"))
    block = Data(key=1, leaf=0)
    base._stash = [Data(key=0, leaf=0), block]
    assert base._require_in_stash(1) is block
    with pytest.raises(KeyError, match="Key 2 not found"):
        base._require_in_stash(2)


def test_a_write_back_that_fails_while_sealing_is_staged_by_the_next_read(fail_once, client, read_all_blocks):
    base = TreeStorageBase(PathOramConfig(num_data=16, data_size=4, client=client, name="unit"))
    base._host_tree(base._build_tree([Data(key=key, leaf=key % 8, value=b"%d" % key) for key in range(16)]))
    base._read_paths([0])
    fail_once(base._cipher, "seal_path")

    with pytest.raises(RuntimeError, match="injected"):
        base._write_paths()

    base._read_paths([1])
    base._write_paths()
    held = sorted((data.key, data.value) for data in read_all_blocks(base) + base._stash)
    assert held == [(key, b"%d" % key) for key in range(16)]
