import pytest

from oblivlib.dependency import AesGcm, CapacityError
from oblivlib.dependency.layout import Bytes, Col, Record, UInt
from oblivlib.dependency.metrics import Recorder
from oblivlib.dependency.path_cipher import SealedPathCipher, make_path_cipher
from oblivlib.dependency.types import OramBlock

_DATA_SIZE = 16
_BLOCK = Record(
    OramBlock,
    [
        Col(lambda block: block.key, UInt(6)),
        Col(lambda block: block.leaf, UInt(6)),
        Col(lambda block: block.value, Bytes(_DATA_SIZE)),
    ],
)


def _blocks(count: int, value_size: int = _DATA_SIZE) -> list[OramBlock]:
    return [OramBlock(key=63 - i, leaf=63 - i, value=bytes([i + 1]) * value_size) for i in range(count)]


def test_sealed_rows_have_one_length_whatever_the_occupancy():
    cipher = SealedPathCipher(_BLOCK, AesGcm(), bucket_size=3, recorder=Recorder())
    rows = [cipher.seal_bucket(_blocks(count, value_size)) for count in (0, 1, 3) for value_size in (0, _DATA_SIZE)]
    assert {len(row) for row in rows} == {cipher.row_bytes}


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "sealed"])
def test_path_round_trips_and_is_recorded_except_absent_rows(encrypted):
    recorder = Recorder()
    cipher = make_path_cipher(_BLOCK, AesGcm() if encrypted else None, bucket_size=3, recorder=recorder)
    path = {0: _blocks(3), 1: [], 3: _blocks(1)}
    rows = cipher.seal_path(path)
    empty = cipher.seal_bucket([])
    assert cipher.open_path(rows) == path
    assert cipher.open_path({0: b"", 2: empty}) == {0: [], 2: []}

    row_bytes = sum(len(row) for row in rows.values()) + len(empty)
    assert (recorder.rows_sealed, recorder.rows_opened) == (4, 4)
    assert recorder.sealed_bytes == recorder.opened_bytes == row_bytes
    assert min(recorder.encode_ns, recorder.decode_ns, recorder.encrypt_ns, recorder.decrypt_ns) >= 0
    if not encrypted:
        assert recorder.encrypt_ns == recorder.decrypt_ns == 0


def test_sealed_row_length_is_the_bucket_layout_plus_the_cipher_overhead():
    block_bytes = 1 + 1 + (1 + _DATA_SIZE)
    assert SealedPathCipher(_BLOCK, AesGcm(), bucket_size=3, recorder=Recorder()).row_bytes == 1 + 3 * block_bytes + 28


def test_sealed_bucket_rejects_more_blocks_or_bytes_than_a_row_holds():
    cipher = SealedPathCipher(_BLOCK, AesGcm(), bucket_size=3, recorder=Recorder())
    with pytest.raises(CapacityError, match="at most 3"):
        cipher.seal_bucket(_blocks(4))
    with pytest.raises(CapacityError, match="16-byte field"):
        cipher.seal_bucket(_blocks(3, value_size=_DATA_SIZE + 1))
