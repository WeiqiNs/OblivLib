import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import override

from oblivlib.dependency.heap_index import leaf_index, path_to_root, tree_size
from oblivlib.dependency.path_cipher import PathCipher
from oblivlib.dependency.types import Block, Bucket


@dataclass(frozen=True)
class MemoryImage:
    level: int
    row_bytes: int | None
    rows: list[bytes]


@dataclass(frozen=True)
class FileImage:
    """A built tree on disk in the server's format: ``2**level - 1`` sealed rows of ``row_bytes`` each,
    in heap order from the root."""

    level: int
    row_bytes: int
    path: Path

    def rows(self) -> Iterator[bytes]:
        with self.path.open("rb") as file:
            while row := file.read(self.row_bytes):
                yield row


TreeImage = MemoryImage | FileImage


@dataclass(frozen=True)
class BuildResult[B]:
    image: TreeImage
    overflow: list[B]


class _Buckets[B](ABC):
    @abstractmethod
    def read(self, index: int) -> Bucket[B]: ...

    @abstractmethod
    def write(self, index: int, bucket: Bucket[B]) -> None: ...

    @abstractmethod
    def finish(self) -> TreeImage: ...

    @abstractmethod
    def close(self) -> None: ...


class _MemoryBuckets[B](_Buckets[B]):
    def __init__(self, level: int, cipher: PathCipher[B]):
        self._level: int = level
        self._cipher: PathCipher[B] = cipher
        self._buckets: list[Bucket[B]] = [[] for _ in range(tree_size(level))]

    @override
    def read(self, index: int) -> Bucket[B]:
        return self._buckets[index]

    @override
    def write(self, index: int, bucket: Bucket[B]) -> None:
        self._buckets[index] = bucket

    @override
    def finish(self) -> MemoryImage:
        rows = [self._cipher.seal_bucket(bucket) for bucket in self._buckets]
        return MemoryImage(level=self._level, row_bytes=self._cipher.row_bytes, rows=rows)

    @override
    def close(self) -> None:
        pass


class _FileBuckets[B](_Buckets[B]):
    def __init__(self, level: int, cipher: PathCipher[B], path: Path):
        assert cipher.row_bytes is not None
        self._level: int = level
        self._cipher: PathCipher[B] = cipher
        self._row_bytes: int = cipher.row_bytes
        self._path: Path = path
        self._fd: int = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            for index in range(tree_size(level)):
                self.write(index, [])
        except BaseException:
            self.close()
            raise

    @override
    def read(self, index: int) -> Bucket[B]:
        return self._cipher.open_bucket(os.pread(self._fd, self._row_bytes, index * self._row_bytes))

    @override
    def write(self, index: int, bucket: Bucket[B]) -> None:
        _ = os.pwrite(self._fd, self._cipher.seal_bucket(bucket), index * self._row_bytes)

    @override
    def finish(self) -> FileImage:
        self.close()
        return FileImage(level=self._level, row_bytes=self._row_bytes, path=self._path)

    @override
    def close(self) -> None:
        os.close(self._fd)


def build_tree[B: Block[object]](
    blocks: Iterable[B], *, level: int, bucket_size: int, cipher: PathCipher[B], build_file: Path | None
) -> BuildResult[B]:
    """Place each block in the deepest non-full bucket on its leaf's path and seal every bucket into a
    row, in memory or, with ``build_file``, in that file (sealed throughout, never plaintext on disk).
    Blocks whose whole path is full come back as ``overflow``."""
    buckets: _Buckets[B] = (
        _MemoryBuckets(level, cipher) if build_file is None else _FileBuckets(level, cipher, build_file)
    )
    overflow: list[B] = []
    try:
        for block in blocks:
            for index in path_to_root(leaf_index(block.leaf, level)):
                bucket = buckets.read(index)
                if len(bucket) < bucket_size:
                    bucket.append(block)
                    buckets.write(index, bucket)
                    break
            else:
                overflow.append(block)
    except BaseException:
        buckets.close()
        raise
    return BuildResult(image=buckets.finish(), overflow=overflow)
