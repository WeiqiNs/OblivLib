import time
from abc import ABC, abstractmethod
from typing import override

from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.layout import Field, Seq
from oblivlib.dependency.metrics import Recorder
from oblivlib.dependency.types import Bucket, PathData, PathRows


class PathCipher[B](ABC):
    """Turns a scheme's buckets into server rows and back. A bucket is ``Seq(block, bucket_size)`` of
    the scheme's block layout, so every bucket encodes to the same length; an absent row (``b""``)
    opens as an empty bucket. ``row_bytes`` is the fixed row length, or ``None`` when the server keeps
    the rows in a variable-row store."""

    def __init__(self, block: Field[B], *, bucket_size: int, recorder: Recorder):
        self._bucket: Seq[B] = Seq(block, bucket_size)
        self._recorder: Recorder = recorder

    @property
    @abstractmethod
    def row_bytes(self) -> int | None: ...

    @abstractmethod
    def seal_path(self, path: PathData[B]) -> PathRows: ...

    @abstractmethod
    def open_path(self, rows: PathRows) -> PathData[B]: ...

    def seal_bucket(self, blocks: Bucket[B]) -> bytes:
        return self.seal_path({0: blocks})[0]

    def open_bucket(self, row: bytes) -> Bucket[B]:
        return self.open_path({0: row})[0]

    def _encode(self, path: PathData[B]) -> PathRows:
        return {index: self._bucket.encode(bucket) for index, bucket in path.items()}

    def _decode(self, rows: PathRows, plaintexts: PathRows) -> PathData[B]:
        return {index: self._bucket.decode(plaintexts[index]) if row else [] for index, row in rows.items()}


class PlainPathCipher[B](PathCipher[B]):
    """Rows are the encoded buckets themselves, unsealed: for debugging, memory servers only."""

    @property
    @override
    def row_bytes(self) -> None:
        return None

    @override
    def seal_path(self, path: PathData[B]) -> PathRows:
        start = time.perf_counter_ns()
        rows = self._encode(path)
        encode_ns = time.perf_counter_ns() - start
        self._recorder.record_seal(rows=len(rows), sealed_bytes=_size(rows), encode_ns=encode_ns, encrypt_ns=0)
        return rows

    @override
    def open_path(self, rows: PathRows) -> PathData[B]:
        start = time.perf_counter_ns()
        path = self._decode(rows, rows)
        decode_ns = time.perf_counter_ns() - start
        self._recorder.record_open(rows=_present(rows), opened_bytes=_size(rows), decrypt_ns=0, decode_ns=decode_ns)
        return path


class SealedPathCipher[B](PathCipher[B]):
    """Each row is ``enc(encoded bucket)``; the bucket layout is fixed, so every row has one length."""

    def __init__(self, block: Field[B], encryptor: Encryptor, *, bucket_size: int, recorder: Recorder):
        super().__init__(block, bucket_size=bucket_size, recorder=recorder)
        self._encryptor: Encryptor = encryptor
        self._row_bytes: int = encryptor.ciphertext_length(self._bucket.size)

    @property
    @override
    def row_bytes(self) -> int:
        return self._row_bytes

    @override
    def seal_path(self, path: PathData[B]) -> PathRows:
        start = time.perf_counter_ns()
        encoded = self._encode(path)
        encoded_at = time.perf_counter_ns()
        rows = {index: self._encryptor.enc(plaintext) for index, plaintext in encoded.items()}
        self._recorder.record_seal(
            rows=len(rows),
            sealed_bytes=_size(rows),
            encode_ns=encoded_at - start,
            encrypt_ns=time.perf_counter_ns() - encoded_at,
        )
        return rows

    @override
    def open_path(self, rows: PathRows) -> PathData[B]:
        start = time.perf_counter_ns()
        plaintexts = {index: self._encryptor.dec(row) for index, row in rows.items() if row}
        decrypted_at = time.perf_counter_ns()
        path = self._decode(rows, plaintexts)
        self._recorder.record_open(
            rows=len(plaintexts),
            opened_bytes=_size(rows),
            decrypt_ns=decrypted_at - start,
            decode_ns=time.perf_counter_ns() - decrypted_at,
        )
        return path


def _size(rows: PathRows) -> int:
    return sum(len(row) for row in rows.values())


def _present(rows: PathRows) -> int:
    return sum(1 for row in rows.values() if row)


def make_path_cipher[B](
    block: Field[B], encryptor: Encryptor | None, *, bucket_size: int, recorder: Recorder
) -> PathCipher[B]:
    if encryptor is None:
        return PlainPathCipher(block, bucket_size=bucket_size, recorder=recorder)
    return SealedPathCipher(block, encryptor, bucket_size=bucket_size, recorder=recorder)
