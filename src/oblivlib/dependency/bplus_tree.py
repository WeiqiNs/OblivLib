from dataclasses import dataclass


@dataclass
class BPlusLeaf:
    key: int
    leaf: int
    keys: list[bytes]
    values: list[bytes]


@dataclass
class BPlusInternal:
    key: int
    leaf: int
    keys: list[bytes]
    children: list[tuple[int, int]]


BPlusNode = BPlusLeaf | BPlusInternal
