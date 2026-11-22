from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol


class Unset(Enum):
    UNSET = "UNSET"


UNSET: Final = Unset.UNSET


class Block[K](Protocol):
    leaf: int

    @property
    def key(self) -> K: ...


@dataclass
class OramBlock:
    key: int
    leaf: int
    value: bytes


@dataclass
class KeyedBlock:
    key: bytes
    leaf: int
    value: bytes


type Bucket[B] = list[B]
type PathData[B] = dict[int, list[B]]
PathRows = dict[int, bytes]

PosMap = dict[int, int]
DataMap = dict[int, bytes]
InitData = Mapping[int, bytes] | Sequence[tuple[int, bytes]] | Iterator[tuple[int, bytes]]
