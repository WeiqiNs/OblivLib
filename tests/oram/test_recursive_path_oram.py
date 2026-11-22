from oblivlib.dependency import (
    RecursiveOramConfig,
)
from oblivlib.oram import RecursivePathOram


def test_recursive_oram_keeps_at_most_on_chip_mem_positions_on_the_client(num_data, client):
    oram = RecursivePathOram(RecursiveOramConfig(num_data=num_data, data_size=10, client=client))
    oram.init_server_storage()
    assert len(oram._pos_map) <= 10
