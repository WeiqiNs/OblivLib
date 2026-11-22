from dataclasses import dataclass


@dataclass
class AVLNode:
    key: bytes
    leaf: int
    value: bytes
    r_key: bytes | None = None
    r_leaf: int | None = None
    r_height: int = 0
    l_key: bytes | None = None
    l_leaf: int | None = None
    l_height: int = 0
