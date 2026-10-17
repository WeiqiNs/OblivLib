import random

from oblivlib.dependency import UNSET, MulPathOramConfig
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

        for _ in range(10):
            batch_size = random.randint(2, min(10, num_data))
            keys = random.sample(range(num_data), batch_size)

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
