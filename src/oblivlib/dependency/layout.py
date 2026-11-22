from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Protocol, override

from oblivlib.dependency.errors import CapacityError

Buffer = bytes | bytearray | memoryview


class Field[T](ABC):
    @property
    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def write(self, value: T, out: bytearray) -> None: ...

    @abstractmethod
    def read(self, data: Buffer, offset: int) -> tuple[T, int]: ...

    def encode(self, value: T) -> bytes:
        out = bytearray()
        self.write(value, out)
        return bytes(out)

    def decode(self, data: Buffer) -> T:
        if len(data) != self.size:
            raise ValueError(f"{self!r}: expected {self.size} bytes, got {len(data)}.")
        return self.read(data, 0)[0]


class UInt(Field[int]):
    def __init__(self, bits: int):
        if bits < 1:
            raise ValueError(f"UInt needs at least 1 bit, got {bits}.")
        self._bits: int = bits
        self._size: int = (bits + 7) // 8
        self._limit: int = 1 << bits

    @classmethod
    def covering(cls, max_value: int) -> "UInt":
        return cls(max(1, max_value.bit_length()))

    @property
    @override
    def size(self) -> int:
        return self._size

    @override
    def write(self, value: int, out: bytearray) -> None:
        if not 0 <= value < self._limit:
            raise CapacityError(f"{value} does not fit a {self._bits}-bit unsigned field.")
        out += value.to_bytes(self._size, "big")

    @override
    def read(self, data: Buffer, offset: int) -> tuple[int, int]:
        end = offset + self._size
        return int.from_bytes(data[offset:end], "big"), end

    @override
    def __repr__(self) -> str:
        return f"UInt({self._bits})"


class Bytes(Field[bytes]):
    def __init__(self, max_size: int):
        self._max_size: int = max_size
        self._length: UInt = UInt.covering(max_size)

    @property
    @override
    def size(self) -> int:
        return self._length.size + self._max_size

    @override
    def write(self, value: bytes, out: bytearray) -> None:
        if len(value) > self._max_size:
            raise CapacityError(f"{len(value)} bytes do not fit a {self._max_size}-byte field.")
        self._length.write(len(value), out)
        out += value
        out += bytes(self._max_size - len(value))

    @override
    def read(self, data: Buffer, offset: int) -> tuple[bytes, int]:
        length, start = self._length.read(data, offset)
        return bytes(data[start : start + length]), start + self._max_size

    @override
    def __repr__(self) -> str:
        return f"Bytes({self._max_size})"


class FixedBytes(Field[bytes]):
    def __init__(self, size: int):
        self._size: int = size

    @property
    @override
    def size(self) -> int:
        return self._size

    @override
    def write(self, value: bytes, out: bytearray) -> None:
        if len(value) != self._size:
            raise CapacityError(f"{len(value)} bytes do not fit an exactly {self._size}-byte field.")
        out += value

    @override
    def read(self, data: Buffer, offset: int) -> tuple[bytes, int]:
        end = offset + self._size
        return bytes(data[offset:end]), end

    @override
    def __repr__(self) -> str:
        return f"FixedBytes({self._size})"


class Optional[T](Field[T | None]):
    def __init__(self, field: Field[T]):
        self._field: Field[T] = field

    @property
    @override
    def size(self) -> int:
        return 1 + self._field.size

    @override
    def write(self, value: T | None, out: bytearray) -> None:
        if value is None:
            out += bytes(self.size)
            return
        out.append(1)
        self._field.write(value, out)

    @override
    def read(self, data: Buffer, offset: int) -> tuple[T | None, int]:
        if data[offset] == 0:
            return None, offset + self.size
        return self._field.read(data, offset + 1)

    @override
    def __repr__(self) -> str:
        return f"Optional({self._field!r})"


class Array[T](Field[list[T]]):
    def __init__(self, field: Field[T], count: int):
        self._field: Field[T] = field
        self._count: int = count

    @property
    @override
    def size(self) -> int:
        return self._count * self._field.size

    @override
    def write(self, value: Sequence[T], out: bytearray) -> None:
        if len(value) != self._count:
            raise CapacityError(f"{len(value)} items do not fit an array of exactly {self._count}.")
        for item in value:
            self._field.write(item, out)

    @override
    def read(self, data: Buffer, offset: int) -> tuple[list[T], int]:
        items: list[T] = []
        for _ in range(self._count):
            item, offset = self._field.read(data, offset)
            items.append(item)
        return items, offset

    @override
    def __repr__(self) -> str:
        return f"Array({self._field!r}, {self._count})"


class Seq[T](Field[list[T]]):
    def __init__(self, field: Field[T], max_count: int):
        self._field: Field[T] = field
        self._max_count: int = max_count
        self._count: UInt = UInt.covering(max_count)

    @property
    @override
    def size(self) -> int:
        return self._count.size + self._max_count * self._field.size

    @override
    def write(self, value: Sequence[T], out: bytearray) -> None:
        if len(value) > self._max_count:
            raise CapacityError(f"{len(value)} items do not fit a sequence of at most {self._max_count}.")
        self._count.write(len(value), out)
        for item in value:
            self._field.write(item, out)
        out += bytes((self._max_count - len(value)) * self._field.size)

    @override
    def read(self, data: Buffer, offset: int) -> tuple[list[T], int]:
        count, offset = self._count.read(data, offset)
        end = offset + self._max_count * self._field.size
        items: list[T] = []
        for _ in range(count):
            item, offset = self._field.read(data, offset)
            items.append(item)
        return items, end

    @override
    def __repr__(self) -> str:
        return f"Seq({self._field!r}, {self._max_count})"


class Pair[A, B](Field[tuple[A, B]]):
    def __init__(self, first: Field[A], second: Field[B]):
        self._first: Field[A] = first
        self._second: Field[B] = second

    @property
    @override
    def size(self) -> int:
        return self._first.size + self._second.size

    @override
    def write(self, value: tuple[A, B], out: bytearray) -> None:
        first, second = value
        self._first.write(first, out)
        self._second.write(second, out)

    @override
    def read(self, data: Buffer, offset: int) -> tuple[tuple[A, B], int]:
        first, offset = self._first.read(data, offset)
        second, offset = self._second.read(data, offset)
        return (first, second), offset

    @override
    def __repr__(self) -> str:
        return f"Pair({self._first!r}, {self._second!r})"


class Column[T](Protocol):
    @property
    def size(self) -> int: ...

    def write(self, block: T, out: bytearray) -> None: ...

    def read(self, data: Buffer, offset: int) -> tuple[object, int]: ...


class Col[T, V]:
    def __init__(self, get: Callable[[T], V], field: Field[V]):
        self._get: Callable[[T], V] = get
        self._field: Field[V] = field

    @property
    def size(self) -> int:
        return self._field.size

    def write(self, block: T, out: bytearray) -> None:
        self._field.write(self._get(block), out)

    def read(self, data: Buffer, offset: int) -> tuple[object, int]:
        return self._field.read(data, offset)

    @override
    def __repr__(self) -> str:
        return repr(self._field)


class Record[T](Field[T]):
    def __init__(self, cls: type[T], columns: Sequence[Column[T]]):
        self._cls: type[T] = cls
        self._build: Callable[..., T] = cls
        self._columns: tuple[Column[T], ...] = tuple(columns)
        self._size: int = sum(column.size for column in self._columns)

    @property
    @override
    def size(self) -> int:
        return self._size

    @override
    def write(self, value: T, out: bytearray) -> None:
        for column in self._columns:
            column.write(value, out)

    @override
    def read(self, data: Buffer, offset: int) -> tuple[T, int]:
        values: list[object] = []
        for column in self._columns:
            item, offset = column.read(data, offset)
            values.append(item)
        return self._build(*values), offset

    @override
    def __repr__(self) -> str:
        return f"Record({self._cls.__name__}, {list(self._columns)!r})"


class BitFields:
    def __init__(self, widths: Sequence[int]):
        if any(width < 1 for width in widths):
            raise ValueError(f"every bit field needs at least 1 bit, got {list(widths)}.")
        self._widths: tuple[int, ...] = tuple(widths)
        self._bits: int = sum(self._widths)
        shifts: list[int] = []
        remaining = self._bits
        for width in self._widths:
            remaining -= width
            shifts.append(remaining)
        self._shifts: tuple[int, ...] = tuple(shifts)

    def get(self, word: int, index: int) -> int:
        return (word >> self._shifts[index]) & ((1 << self._widths[index]) - 1)

    def set(self, word: int, index: int, value: int) -> int:
        width = self._widths[index]
        if not 0 <= value < 1 << width:
            raise CapacityError(f"{value} does not fit bit field {index}, which is {width} bits wide.")
        shift = self._shifts[index]
        return word & ~(((1 << width) - 1) << shift) | value << shift
