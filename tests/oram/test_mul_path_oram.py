import random

import pytest

from oblivlib.dependency import UNSET, ContractError, MulPathOramConfig
from oblivlib.oram import MulPathOram


class TestMulPathOram:
    def test_batch_write(self, num_data, client):
        oram = MulPathOram(MulPathOramConfig(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3))
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=b"%d" % i)

        updates = {0: b"100", 1: b"101", 2: b"102"}
        old_values = oram.operate_on_keys(key_value_map=updates)

        assert old_values[0] == b"0"
        assert old_values[1] == b"1"
        assert old_values[2] == b"2"

        for key, new_value in updates.items():
            assert oram.operate_on_key(key=key) == new_value

    def test_random_batch_operations(self, num_data, client):
        oram = MulPathOram(MulPathOramConfig(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=10))
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=b"%d" % i)

        rng = random.Random(3)
        for _ in range(10):
            batch_size = rng.randint(2, min(10, num_data))
            keys = rng.sample(range(num_data), batch_size)

            results = oram.operate_on_keys(key_value_map={k: UNSET for k in keys})

            assert len(results) == batch_size
            for key in keys:
                assert results[key] == b"%d" % key

    def test_explicit_paths_leave_position_map_untouched(self, client):
        oram = MulPathOram(MulPathOramConfig(num_data=64, data_size=10, client=client, stash_scale_multiplier=3))
        oram.init_server_storage(data={k: b"%d" % k for k in range(64)})
        positions = dict(oram._pos_map)
        keys = [1, 2, 3]
        moved = {k: (positions[k] + 1) % oram._leaf_range for k in keys}

        read = oram.operate_on_keys(
            dict.fromkeys(keys, UNSET), key_path_map={k: positions[k] for k in keys}, new_path_map=moved
        )
        assert read == {k: b"%d" % k for k in keys}
        assert oram._pos_map == positions

        oram.operate_on_keys_without_eviction({1: b"one"}, key_path_map={1: moved[1]}, new_path_map={1: positions[1]})
        oram.eviction_for_mul_keys()
        assert oram._pos_map == positions
        assert oram.operate_on_key(key=1) == b"one"

    @pytest.mark.parametrize("bad_leaf", [64, -5, "1"])
    @pytest.mark.parametrize("path_map", ["key_path_map", "new_path_map"])
    @pytest.mark.parametrize("split", [False, True])
    def test_out_of_range_leaves_raise_before_any_state_changes(self, client, bad_leaf, path_map, split):
        oram = MulPathOram(MulPathOramConfig(num_data=64, data_size=10, client=client, stash_scale_multiplier=3))
        oram.init_server_storage(data={1: b"one"})
        positions = dict(oram._pos_map)
        rounds = client.metrics.rounds
        operate = oram.operate_on_keys_without_eviction if split else oram.operate_on_keys

        with pytest.raises(ContractError, match=r"a leaf must be an int in \[0, 64\)"):
            operate({1: b"uno"}, **{path_map: {1: bad_leaf}})

        assert client.metrics.rounds == rounds
        assert oram._pos_map == positions
        assert oram.operate_on_keys({1: UNSET}) == {1: b"one"}

    def test_batch_contract_violations_raise_before_any_round(self, client):
        oram = MulPathOram(MulPathOramConfig(num_data=64, data_size=10, client=client, stash_scale_multiplier=3))
        oram.init_server_storage(data={1: b"one"})
        rounds = client.metrics.rounds

        for batch in ({64: UNSET}, {1: "one"}, {1: b"x" * 11}):
            with pytest.raises(ContractError):
                oram.operate_on_keys(batch)
            with pytest.raises(ContractError):
                oram.operate_on_keys_without_eviction(batch)
        assert oram.operate_on_keys({}) == {}
        assert client.metrics.rounds == rounds

        oram.operate_on_keys_without_eviction({1: UNSET})
        for updates in ({64: b"v"}, {1: "one"}, {1: b"x" * 11}):
            with pytest.raises(ContractError):
                oram.eviction_for_mul_keys(updates)
        with pytest.raises(KeyError, match="Key 2 not found"):
            oram.eviction_for_mul_keys({1: b"uno", 2: b"two"})
        oram.eviction_for_mul_keys()
        assert oram.operate_on_keys({1: UNSET, 2: UNSET}) == {1: b"one", 2: b""}
        with pytest.raises(ValueError, match="no access is waiting"):
            oram.eviction_for_mul_keys({1: b"uno"})
        assert oram.operate_on_keys({1: UNSET}) == {1: b"one"}

    def test_without_eviction(self, num_data, client):
        oram = MulPathOram(MulPathOramConfig(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3))
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=b"%d" % i)

        keys_to_read = [0, 1, 2]
        results = oram.operate_on_keys_without_eviction(key_value_map={k: UNSET for k in keys_to_read})

        assert results[0] == b"0"
        assert results[1] == b"1"
        assert results[2] == b"2"

        oram.eviction_for_mul_keys(updates={0: b"100", 1: b"101"})

        assert oram.operate_on_key(key=0) == b"100"
        assert oram.operate_on_key(key=1) == b"101"
        assert oram.operate_on_key(key=2) == b"2"
