import pytest

from oblivlib.dependency import AesGcm, CapacityError, Data
from oblivlib.dependency.layout import Bytes, UInt, block_layout
from oblivlib.dependency.path_cipher import PlainPathCipher, SealedPathCipher, make_path_cipher

_DATA_SIZE = 16
_BLOCK = block_layout(UInt(6), UInt(6), Bytes(_DATA_SIZE))


def _blocks(count: int, value_size: int = _DATA_SIZE) -> list[Data]:
    return [Data(key=63 - i, leaf=63 - i, value=bytes([i + 1]) * value_size) for i in range(count)]


def test_sealed_rows_have_one_length_whatever_the_occupancy():
    cipher = SealedPathCipher(_BLOCK, AesGcm(), bucket_size=3)
    rows = [cipher.seal_bucket(_blocks(count, value_size)) for count in (0, 1, 3) for value_size in (0, _DATA_SIZE)]
    assert {len(row) for row in rows} == {cipher.row_bytes}


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "sealed"])
def test_path_round_trips_and_absent_rows_open_empty(encrypted):
    cipher = make_path_cipher(_BLOCK, AesGcm() if encrypted else None, bucket_size=3)
    path = {0: _blocks(3), 1: [], 3: _blocks(1)}
    assert cipher.open_path(cipher.seal_path(path)) == path
    assert cipher.open_path({0: b"", 2: cipher.seal_bucket([])}) == {0: [], 2: []}


def test_plain_empty_bucket_is_not_absent():
    assert PlainPathCipher(_BLOCK, bucket_size=3).seal_bucket([]) != b""


def test_sealed_row_length_is_the_bucket_layout_plus_the_cipher_overhead():
    block_bytes = 1 + 1 + (1 + _DATA_SIZE)
    assert SealedPathCipher(_BLOCK, AesGcm(), bucket_size=3).row_bytes == 1 + 3 * block_bytes + 28


def test_sealed_bucket_rejects_more_blocks_or_bytes_than_a_row_holds():
    cipher = SealedPathCipher(_BLOCK, AesGcm(), bucket_size=3)
    with pytest.raises(CapacityError, match="at most 3"):
        cipher.seal_bucket(_blocks(4))
    with pytest.raises(CapacityError, match="16-byte field"):
        cipher.seal_bucket(_blocks(3, value_size=_DATA_SIZE + 1))
