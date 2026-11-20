from collections.abc import Iterable, Sequence
from typing import Any

from oblivlib.dependency.errors import ContractError


def require_oram_key(owner: str, key: Any, num_data: int) -> int:
    if type(key) is not int or not 0 <= key < num_data:
        raise ContractError(f"{owner}: a key must be an int in [0, {num_data}), got {key!r}.")
    return key


def require_leaf(owner: str, leaf: Any, leaf_range: int) -> int:
    if type(leaf) is not int or not 0 <= leaf < leaf_range:
        raise ContractError(f"{owner}: a leaf must be an int in [0, {leaf_range}), got {leaf!r}.")
    return leaf


def require_omap_key(owner: str, key: Any, key_size: int) -> bytes:
    if type(key) is not bytes:
        raise ContractError(f"{owner}: a key must be bytes, got {type(key).__name__}.")
    if len(key) > key_size:
        raise ContractError(f"{owner}: a key is {len(key)} bytes, key_size is {key_size}.")
    return key


def require_value(owner: str, value: Any, data_size: int) -> bytes:
    if type(value) is not bytes:
        raise ContractError(f"{owner}: a value must be bytes, got {type(value).__name__}.")
    if len(value) > data_size:
        raise ContractError(f"{owner}: a value is {len(value)} bytes, data_size is {data_size}.")
    return value


def require_omap_init(owner: str, pairs: Iterable[tuple[Any, Any]], *, key_size: int, data_size: int) -> None:
    seen: set[bytes] = set()
    for key, value in pairs:
        require_omap_key(owner, key, key_size)
        require_value(owner, value, data_size)
        if key in seen:
            raise ContractError(f"{owner}: initial key {key!r} is given twice.")
        seen.add(key)


def require_batch(owner: str, keys: Sequence[Any], batch_size: int, *, distinct: bool) -> None:
    if len(keys) > batch_size:
        raise ContractError(f"{owner}: a batch holds {len(keys)} keys, batch_size is {batch_size}.")
    if distinct and len(set(keys)) < len(keys):
        raise ContractError(f"{owner}: a batch gives a key twice.")
