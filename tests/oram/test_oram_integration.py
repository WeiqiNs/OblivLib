import pytest

from oblivlib.dependency import RecursiveOramConfig
from oblivlib.oram import RecursivePathOram


@pytest.mark.parametrize("mode", ["adopt", "attach", "stream"])
def test_build_file_handover_modes(mode, tmp_path, encryptor, handover_client):
    client = handover_client(mode)
    build_file = tmp_path / "build.tree"
    oram = RecursivePathOram(
        RecursiveOramConfig(num_data=128, data_size=10, client=client, encryptor=encryptor, build_file=build_file)
    )
    with client.measure() as init:
        oram.init_server_storage(data={key: b"%d" % (key * 3) for key in range(128)})

    assert not build_file.exists()
    assert init.result.rows_sealed > 0
    assert (init.result.row_bytes_written > 0) is (mode == "stream")
    for key in range(128):
        assert oram.operate_on_key(key=key, value=b"%d" % key) == b"%d" % (key * 3)
    for key in range(128):
        assert oram.operate_on_key(key=key) == b"%d" % key
    client.close()
