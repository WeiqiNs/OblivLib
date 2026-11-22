import random
import secrets

from oblivlib.dependency import (
    DaOramConfig,
)
from oblivlib.oram import DAOram


def test_da_oram_matches_a_dict_model_while_reset_blocks_wait_in_a_stash(monkeypatch, client):
    monkeypatch.setattr(secrets, "randbelow", random.Random(9).randrange)
    oram = DAOram(
        DaOramConfig(num_data=1024, data_size=8, client=client, bucket_size=1, stash_scale=1000, prf_key=b"\x00" * 32)
    )
    model = {i: b"%d" % i for i in range(1024)}
    oram.init_server_storage(data=dict(model))

    workload = random.Random(9)
    for step in range(3000):
        key = workload.randrange(1024)
        assert oram.operate_on_key(key=key, value=b"%d" % step) == model[key]
        model[key] = b"%d" % step


def test_da_oram_resets_in_a_partly_filled_last_block(client):
    oram = DAOram(DaOramConfig(num_data=102, data_size=8, client=client, num_ic=4, ic_length=2, on_chip_mem=4))
    oram.init_server_storage()
    model: dict[int, bytes] = {}
    workload = random.Random(11)
    for step in range(1500):
        key = workload.choice([96, 97, 98, 99, 100, 101])
        assert oram.operate_on_key(key=key, value=b"%d" % step) == model.get(key, b"")
        model[key] = b"%d" % step
