from collections.abc import Iterable, Mapping, Sequence

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


def require_omap_pair(
    owner: str, key: object, value: object, *, key_size: int, data_size: int, value_required: bool
) -> None:
    require_omap_key(owner, key, key_size)
    if value_required or value is not None:
        require_value(owner, value, data_size)


def require_room(owner: str, *, held: int, fresh: int, capacity: int) -> None:
    if held + fresh > capacity:
        raise ContractError(f"{owner}: {held} held and {fresh} more exceed num_data={capacity}.")


def require_vertex(owner: str, vertex: object) -> None:
    require_oram_key(owner, vertex, 1 << 64)


def require_neighbors(owner: str, vertex: int, neighbors: Sequence[int], max_degree: int) -> None:
    for neighbor in neighbors:
        require_vertex(owner, neighbor)
    if len(neighbors) > max_degree:
        raise ContractError(f"{owner}: vertex {vertex} has {len(neighbors)} neighbors, max_degree is {max_degree}.")
    if len(set(neighbors)) != len(neighbors):
        raise ContractError(f"{owner}: vertex {vertex} lists a neighbor twice.")
    if vertex in neighbors:
        raise ContractError(f"{owner}: vertex {vertex} lists itself as a neighbor.")


def require_graph(
    owner: str,
    graph: Mapping[int, Sequence[int]],
    values: Mapping[int, bytes],
    *,
    capacity: int,
    max_degree: int,
    data_size: int,
) -> None:
    require_room(owner, held=0, fresh=len(graph), capacity=capacity)
    for vertex, neighbors in graph.items():
        require_vertex(owner, vertex)
        require_neighbors(owner, vertex, neighbors, max_degree)
        for neighbor in neighbors:
            if vertex not in graph.get(neighbor, ()):
                raise ContractError(f"{owner}: edge ({vertex}, {neighbor}) is not listed both ways.")
    for value in values.values():
        require_value(owner, value, data_size)


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
