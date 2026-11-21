from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, override

from oblivlib.dependency.errors import CapacityError
from oblivlib.dependency.types import Data, FieldTuple

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
        self._bits = bits
        self._size = (bits + 7) // 8
        self._limit = 1 << bits

    @classmethod
    def covering(cls, max_value: int) -> "UInt":
        return cls(max(1, max_value.bit_length()))

    @property
    def bits(self) -> int:
        return self._bits

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
        self._max_size = max_size
        self._length = UInt.covering(max_size)

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
        self._size = size

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
        self._field = field

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
        self._field = field
        self._count = count

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
        items = []
        for _ in range(self._count):
            item, offset = self._field.read(data, offset)
            items.append(item)
        return items, offset

    @override
    def __repr__(self) -> str:
        return f"Array({self._field!r}, {self._count})"


class Seq[T](Field[list[T]]):
    def __init__(self, field: Field[T], max_count: int):
        self._field = field
        self._max_count = max_count
        self._count = UInt.covering(max_count)

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
        items = []
        for _ in range(count):
            item, offset = self._field.read(data, offset)
            items.append(item)
        return items, end

    @override
    def __repr__(self) -> str:
        return f"Seq({self._field!r}, {self._max_count})"


class Record(Field[Any]):
    def __init__(self, fields: Sequence[Field[Any]], cls: type[FieldTuple] | None = None):
        self._fields = tuple(fields)
        self._cls = cls
        self._size = sum(field.size for field in self._fields)

    @property
    @override
    def size(self) -> int:
        return self._size

    @override
    def write(self, value: Any, out: bytearray) -> None:
        values = value.to_fields() if self._cls is not None else value
        if len(values) != len(self._fields):
            raise CapacityError(f"{len(values)} values do not fit a record of {len(self._fields)} fields.")
        for field, item in zip(self._fields, values, strict=True):
            field.write(item, out)

    @override
    def read(self, data: Buffer, offset: int) -> tuple[Any, int]:
        values = []
        for field in self._fields:
            item, offset = field.read(data, offset)
            values.append(item)
        if self._cls is None:
            return tuple(values), offset
        return self._cls.from_fields(values), offset

    @override
    def __repr__(self) -> str:
        name = "tuple" if self._cls is None else self._cls.__name__
        return f"Record({name}, {list(self._fields)!r})"


class Variant(Field[Any]):
    def __init__(self, fields: Sequence[Field[Any]], select: Callable[[Any], int]):
        self._fields = tuple(fields)
        self._select = select
        self._width = max(field.size for field in self._fields)

    @property
    @override
    def size(self) -> int:
        return 1 + self._width

    @override
    def write(self, value: Any, out: bytearray) -> None:
        tag = self._select(value)
        out.append(tag)
        start = len(out)
        self._fields[tag].write(value, out)
        out += bytes(self._width - (len(out) - start))

    @override
    def read(self, data: Buffer, offset: int) -> tuple[Any, int]:
        value, _ = self._fields[data[offset]].read(data, offset + 1)
        return value, offset + self.size

    @override
    def __repr__(self) -> str:
        return f"Variant({list(self._fields)!r})"


class BitFields:
    def __init__(self, widths: Sequence[int]):
        if any(width < 1 for width in widths):
            raise ValueError(f"every bit field needs at least 1 bit, got {list(widths)}.")
        self._widths = tuple(widths)
        self._bits = sum(self._widths)
        shifts = []
        remaining = self._bits
        for width in self._widths:
            remaining -= width
            shifts.append(remaining)
        self._shifts = tuple(shifts)

    @property
    def bits(self) -> int:
        return self._bits

    def get(self, word: int, index: int) -> int:
        return (word >> self._shifts[index]) & ((1 << self._widths[index]) - 1)

    def set(self, word: int, index: int, value: int) -> int:
        width = self._widths[index]
        if not 0 <= value < 1 << width:
            raise CapacityError(f"{value} does not fit bit field {index}, which is {width} bits wide.")
        shift = self._shifts[index]
        return word & ~(((1 << width) - 1) << shift) | value << shift


def block_layout(key: Field[Any], leaf: Field[int], value: Field[Any]) -> Record:
    return Record((key, leaf, value), Data)
