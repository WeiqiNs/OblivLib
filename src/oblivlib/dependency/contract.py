from collections.abc import Iterable, Sequence

from oblivlib.dependency.errors import ContractError


def require_oram_key(owner: str, key: object, num_data: int) -> None:
    if type(key) is not int or not 0 <= key < num_data:
        raise ContractError(f"{owner}: a key must be an int in [0, {num_data}), got {key!r}.")


def require_omap_key(owner: str, key: object, key_size: int) -> None:
    if type(key) is not bytes:
        raise ContractError(f"{owner}: a key must be bytes, got {type(key).__name__}.")
    if len(key) > key_size:
        raise ContractError(f"{owner}: a key is {len(key)} bytes, key_size is {key_size}.")


def require_value(owner: str, value: object, data_size: int) -> None:
    if type(value) is not bytes:
        raise ContractError(f"{owner}: a value must be bytes, got {type(value).__name__}.")
    if len(value) > data_size:
        raise ContractError(f"{owner}: a value is {len(value)} bytes, data_size is {data_size}.")


def require_omap_init(owner: str, pairs: Iterable[tuple[object, object]], *, key_size: int, data_size: int) -> None:
    seen: set[object] = set()
    for key, value in pairs:
        require_omap_key(owner, key, key_size)
        require_value(owner, value, data_size)
        if key in seen:
            raise ContractError(f"{owner}: initial key {key!r} is given twice.")
        seen.add(key)


def require_batch(owner: str, keys: Sequence[object], batch_size: int, *, distinct: bool) -> None:
    if len(keys) > batch_size:
        raise ContractError(f"{owner}: a batch holds {len(keys)} keys, batch_size is {batch_size}.")
    if distinct and len(set(keys)) < len(keys):
        raise ContractError(f"{owner}: a batch gives a key twice.")
