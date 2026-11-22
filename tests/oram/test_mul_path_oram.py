import random
from typing import Any

import pytest

from oblivlib.dependency import ContractError, MulPathOramConfig
from oblivlib.dependency.types import UNSET
from oblivlib.oram import MulPathOram


def _mul_path(client, num_data=64):
    return MulPathOram(MulPathOramConfig(num_data=num_data, data_size=10, client=client, stash_scale=70))


def test_operate_on_keys_matches_a_dict_model(client, duplicate_blocks):
    oram = _mul_path(client, num_data=256)
    model = {key: b"%d" % key for key in range(256)}
    oram.init_server_storage(data=dict(model))
    rng = random.Random(3)

    for step in range(200):
        keys = rng.sample(range(256), rng.randint(1, 10))
        batch = {key: rng.choice([UNSET, b"", b"s%d" % step]) for key in keys}
        assert oram.operate_on_keys(batch) == {key: model[key] for key in keys}
        for key, value in batch.items():
            if isinstance(value, bytes):
                model[key] = value
    assert oram.operate_on_keys(dict.fromkeys(range(10), UNSET)) == {key: model[key] for key in range(10)}
    assert not duplicate_blocks(oram)


def test_operate_on_keys_reads_every_key_in_one_batch(recorded_client):
    client, recorder = recorded_client()
    oram = _mul_path(client)
    oram.init_server_storage()
    before = len(recorder.batches)
    oram.operate_on_keys({1: b"a", 2: b"b", 3: UNSET})
    reads = [len(read.leaves) for batch in recorder.batches[before:] for read in batch.reads]
    assert len(reads) == 1
    assert 1 <= reads[0] <= 3


def test_batch_contract_violations_raise_before_any_message(recorded_client):
    client, recorder = recorded_client()
    oram = _mul_path(client)
    oram.init_server_storage(data={1: b"one"})
    sent = len(recorder.messages)

    batches: list[dict[int, Any]] = [{64: UNSET}, {1: "one"}, {1: b"x" * 11}, {1: b"uno", 64: UNSET}]
    for batch in batches:
        with pytest.raises(ContractError, match="MulPathOram '"):
            oram.operate_on_keys(batch)
    assert oram.operate_on_keys({}) == {}
    assert len(recorder.messages) == sent
    assert oram.operate_on_keys({1: UNSET}) == {1: b"one"}
