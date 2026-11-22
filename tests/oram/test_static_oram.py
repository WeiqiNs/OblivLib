import secrets

from oblivlib.dependency import StaticOramConfig
from oblivlib.oram import StaticOram


def test_static_oram_never_samples_a_random_leaf(monkeypatch, client):
    config = StaticOramConfig(num_data=64, data_size=10, client=client)

    def no_random_leaf(_: int) -> int:
        raise AssertionError("StaticOram drew a random leaf.")

    monkeypatch.setattr(secrets, "randbelow", no_random_leaf)
    oram = StaticOram(config)
    oram.init_server_storage(data={3: b"three"})
    assert oram.operate_on_key(3, b"tres") == b"three"
    assert oram.operate_on_key(3) == b"tres"
