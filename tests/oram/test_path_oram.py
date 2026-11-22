from oblivlib.dependency import (
    PathOramConfig,
)
from oblivlib.oram import PathOram


def test_init_blocks_that_overflow_a_path_stay_in_the_stash(client):
    oram = PathOram(PathOramConfig(num_data=4, data_size=10, client=client, bucket_size=1))
    oram._pos_map = dict.fromkeys(range(4), 0)
    oram.init_server_storage(data={i: b"v%d" % i for i in range(4)})

    assert [data.key for data in oram._stash] == [3]
    for i in range(4):
        assert oram.operate_on_key(key=i) == b"v%d" % i
