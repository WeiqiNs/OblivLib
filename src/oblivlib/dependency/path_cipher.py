from abc import ABC, abstractmethod
from typing import override

from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.layout import Field, Seq
from oblivlib.dependency.types import Bucket, PathData, PathRows


class PathCipher[B](ABC):
    """Turns a scheme's buckets into server rows and back. A bucket is ``Seq(block, bucket_size)`` of
    the scheme's block layout, so every bucket encodes to the same length; an absent row (``b""``)
    opens as an empty bucket. ``row_bytes`` is the fixed row length, or ``None`` when the server keeps
    the rows in a variable-row store."""

    def __init__(self, block: Field[B], bucket_size: int):
        self._bucket: Seq[B] = Seq(block, bucket_size)

    @property
    @abstractmethod
    def row_bytes(self) -> int | None: ...

    @abstractmethod
    def seal_bucket(self, blocks: Bucket[B]) -> bytes: ...

    @abstractmethod
    def _open_row(self, row: bytes) -> bytes: ...

    def open_bucket(self, row: bytes) -> Bucket[B]:
        if row == b"":
            return []
        return self._bucket.decode(self._open_row(row))

    def seal_path(self, path: PathData[B]) -> PathRows:
        return {index: self.seal_bucket(bucket) for index, bucket in path.items()}

    def open_path(self, rows: PathRows) -> PathData[B]:
        return {index: self.open_bucket(row) for index, row in rows.items()}


class PlainPathCipher[B](PathCipher[B]):
    """Rows are the encoded buckets themselves, unsealed: for debugging, memory servers only."""

    @property
    @override
    def row_bytes(self) -> None:
        return None

    @override
    def seal_bucket(self, blocks: Bucket[B]) -> bytes:
        return self._bucket.encode(blocks)

    @override
    def _open_row(self, row: bytes) -> bytes:
        return row


class SealedPathCipher[B](PathCipher[B]):
    """Each row is ``enc(encoded bucket)``; the bucket layout is fixed, so every row has one length."""

    def __init__(self, block: Field[B], encryptor: Encryptor, bucket_size: int):
        super().__init__(block, bucket_size)
        self._encryptor: Encryptor = encryptor
        self._row_bytes: int = encryptor.ciphertext_length(self._bucket.size)

    @property
    @override
    def row_bytes(self) -> int:
        return self._row_bytes

    @override
    def seal_bucket(self, blocks: Bucket[B]) -> bytes:
        return self._encryptor.enc(self._bucket.encode(blocks))

    @override
    def _open_row(self, row: bytes) -> bytes:
        return self._encryptor.dec(row)


def make_path_cipher[B](block: Field[B], encryptor: Encryptor | None, bucket_size: int) -> PathCipher[B]:
    if encryptor is None:
        return PlainPathCipher(block, bucket_size)
    return SealedPathCipher(block, encryptor, bucket_size)
