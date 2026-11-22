import secrets
import struct
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import override

from oblivlib.dependency import ContractError
from oblivlib.dependency.config import AvlOmapConfig, GraphOsConfig, OramConfig
from oblivlib.dependency.contract import require_value
from oblivlib.dependency.layout import Bytes, Col, Field, Pair, Record, UInt
from oblivlib.dependency.tree_storage_base import TreeStorageBase, leaf_field
from oblivlib.omap import AVLOmap

_KEY_SIZE = 17
_ID_BITS = 64
_ID = struct.Struct(">Q")

_Slot = bytes | None


def _vertex_key(vertex: int) -> bytes:
    return b"V" + _ID.pack(vertex)


def _edge_key(vertex: int, index: int) -> bytes:
    return b"E" + _ID.pack(vertex) + _ID.pack(index)


def _slot_key(vertex: int, neighbor: int) -> bytes:
    return b"P" + _ID.pack(vertex) + _ID.pack(neighbor)


@dataclass
class _Entry:
    key: bytes
    leaf: int
    value: bytes


class _ValueStore(TreeStorageBase[OramConfig, _Entry]):
    @property
    @override
    def _block_layout(self) -> Field[_Entry]:
        return Record(
            _Entry,
            [
                Col(lambda block: block.key, Bytes(_KEY_SIZE)),
                Col(lambda block: block.leaf, self._leaf_field),
                Col(lambda block: block.value, Bytes(self.data_size)),
            ],
        )

    def new_leaf(self) -> int:
        return self._get_new_leaf()

    def host(self, entries: list[_Entry]) -> None:
        self._host_tree(self._build_tree(entries))

    def read_padded(self, leaves: Sequence[int], width: int) -> None:
        paths = set(leaves)
        while len(paths) < width:
            paths.add(self._get_new_leaf())
        self._read_paths(sorted(paths))

    def take(self, keys: Iterable[bytes]) -> dict[bytes, _Entry]:
        wanted = set(keys)
        return {block.key: block for block in self._stash if block.key in wanted}

    def add(self, entries: Iterable[_Entry]) -> None:
        self._stash.extend(entries)
        self._check_stash()

    def drop(self, keys: Iterable[bytes]) -> None:
        doomed = set(keys)
        self._stash[:] = [block for block in self._stash if block.key not in doomed]

    def write_back(self) -> None:
        self._write_paths()


class GraphOS:
    def __init__(self, config: GraphOsConfig):
        self._config: GraphOsConfig = config
        self._max_degree: int = config.max_degree
        self._batch: int = config.batch_size
        self._size: int = 0
        self._leaf: UInt = leaf_field(config.num_entries)

        self._value_tree: _ValueStore = _ValueStore(
            replace(
                config,
                num_data=config.num_entries,
                data_size=self._value_width,
                name=f"{config.name}_values",
                stash_scale=config.stash_scale * self._batch,
            )
        )
        self._omap: AVLOmap = AVLOmap(
            AvlOmapConfig(
                num_data=config.num_entries,
                key_size=_KEY_SIZE,
                data_size=self._leaf.size,
                client=config.client,
                name=f"{config.name}_omap",
                build_file=config.build_file,
                bucket_size=config.bucket_size,
                stash_scale=config.stash_scale,
                encryptor=config.encryptor,
                distinguishable=True,
                batch_size=self._batch,
            )
        )

    @cached_property
    def _vertex_record(self) -> Pair[bytes, int]:
        return Pair(Bytes(self._config.data_size), UInt.covering(self._max_degree))

    @cached_property
    def _id(self) -> Field[int]:
        return UInt(_ID_BITS)

    @cached_property
    def _index(self) -> Field[int]:
        return UInt.covering(self._max_degree - 1)

    @property
    def _value_width(self) -> int:
        return max(self._vertex_record.size, self._id.size, self._index.size)

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._config.name!r}"

    @property
    def size(self) -> int:
        return self._size

    def _require_vertex(self, vertex: object) -> None:
        if type(vertex) is not int or not 0 <= vertex < 1 << _ID_BITS:
            raise ContractError(f"{self._identity}: a vertex must be an int in [0, 2**64), got {vertex!r}.")

    def _require_count(self, name: str, count: object) -> None:
        if type(count) is not int or count < 0:
            raise ContractError(f"{self._identity}: {name} must be an int >= 0, got {count!r}.")

    def _require_value(self, value: object) -> None:
        require_value(self._identity, value, self._config.data_size)

    def _require_neighbors(self, vertex: int, neighbors: Sequence[int]) -> None:
        for neighbor in neighbors:
            self._require_vertex(neighbor)
        if len(neighbors) > self._max_degree:
            raise ContractError(
                f"{self._identity}: vertex {vertex} has {len(neighbors)} neighbors, max_degree is {self._max_degree}."
            )
        if len(set(neighbors)) != len(neighbors):
            raise ContractError(f"{self._identity}: vertex {vertex} lists a neighbor twice.")
        if vertex in neighbors:
            raise ContractError(f"{self._identity}: vertex {vertex} lists itself as a neighbor.")

    def _require_graph(self, graph: Mapping[int, Sequence[int]], values: Mapping[int, bytes]) -> None:
        if len(graph) > self._config.num_data:
            raise ContractError(f"{self._identity}: {len(graph)} vertices exceed num_data {self._config.num_data}.")
        for vertex, neighbors in graph.items():
            self._require_vertex(vertex)
            self._require_neighbors(vertex, neighbors)
            for neighbor in neighbors:
                if vertex not in graph.get(neighbor, ()):
                    raise ContractError(f"{self._identity}: edge ({vertex}, {neighbor}) is not listed both ways.")
        for vertex, value in values.items():
            if vertex not in graph:
                raise ContractError(f"{self._identity}: value given for vertex {vertex}, which is not in the graph.")
            self._require_value(value)

    def _encode_vertex(self, value: bytes, degree: int) -> bytes:
        return self._vertex_record.encode((value, degree))

    def _decode_vertex(self, data: bytes) -> tuple[bytes, int]:
        return self._vertex_record.decode(data)

    def init_server_storage(
        self, graph: Mapping[int, Sequence[int]] | None = None, values: Mapping[int, bytes] | None = None
    ) -> None:
        graph = graph or {}
        values = values or {}
        self._require_graph(graph, values)

        entries: list[tuple[bytes, bytes]] = []
        for vertex, neighbors in graph.items():
            entries.append((_vertex_key(vertex), self._encode_vertex(values.get(vertex, b""), len(neighbors))))
            for index, neighbor in enumerate(neighbors):
                entries.append((_edge_key(vertex, index), self._id.encode(neighbor)))
                entries.append((_slot_key(vertex, neighbor), self._index.encode(index)))

        leaves = {key: self._value_tree.new_leaf() for key, _ in entries}
        self._omap.init_server_storage([(key, self._leaf.encode(leaf)) for key, leaf in leaves.items()])
        self._value_tree.host([_Entry(key=key, leaf=leaves[key], value=value) for key, value in entries])
        self._size = len(graph)

    def _chunks[T](self, items: Sequence[T]) -> list[Sequence[T]]:
        return [items[start : start + self._batch] for start in range(0, len(items), self._batch)]

    def _fetch(self, keys: Sequence[_Slot]) -> dict[bytes, _Entry]:
        lanes = [(key, self._value_tree.new_leaf()) for key in keys if key is not None]
        found = self._omap.search_batch([(key, self._leaf.encode(leaf)) for key, leaf in lanes])

        located: dict[bytes, int] = {}
        old_leaves: list[int] = []
        for (key, leaf), old_leaf in zip(lanes, found, strict=True):
            if old_leaf is not None:
                located[key] = leaf
                old_leaves.append(self._leaf.decode(old_leaf))

        self._value_tree.read_padded(old_leaves, len(keys))
        blocks = self._value_tree.take(located)
        for key, leaf in located.items():
            blocks[key].leaf = leaf
        return blocks

    def _commit(self) -> None:
        self._value_tree.write_back()

    def _read(self, keys: Sequence[_Slot]) -> dict[bytes, bytes]:
        values: dict[bytes, bytes] = {}
        for chunk in self._chunks(keys):
            values.update({key: block.value for key, block in self._fetch(chunk).items()})
            self._commit()
        return values

    def _add(self, entries: Sequence[tuple[bytes, bytes] | None]) -> None:
        for chunk in self._chunks(entries):
            placed = [
                _Entry(key=key, leaf=self._value_tree.new_leaf(), value=value) for key, value in filter(None, chunk)
            ]
            self._omap.insert_batch([(entry.key, self._leaf.encode(entry.leaf)) for entry in placed])
            self._value_tree.read_padded([], len(chunk))
            self._value_tree.add(placed)
            self._commit()

    def _remove(self, keys: Sequence[_Slot]) -> None:
        for chunk in self._chunks(keys):
            old_leaves = [self._omap.delete(key) for key in chunk]
            self._value_tree.read_padded(
                [self._leaf.decode(leaf) for leaf in old_leaves if leaf is not None], len(chunk)
            )
            self._value_tree.drop(key for key in chunk if key is not None)
            self._commit()

    def _vertices(self, vertices: Sequence[int | None]) -> dict[int, tuple[bytes, int]]:
        found = self._read([None if vertex is None else _vertex_key(vertex) for vertex in vertices])
        return {
            vertex: self._decode_vertex(found[_vertex_key(vertex)])
            for vertex in vertices
            if vertex is not None and _vertex_key(vertex) in found
        }

    def _neighbors_of(self, frontier: Sequence[int | None], degrees: Mapping[int, int]) -> list[int | None]:
        keys: list[_Slot] = [
            _edge_key(vertex, index) if vertex is not None and index < degrees.get(vertex, 0) else None
            for vertex in frontier
            for index in range(self._max_degree)
        ]
        found = self._read(keys)
        return [None if key is None else self._id.decode(found[key]) for key in keys]

    @staticmethod
    def _degrees(info: Mapping[int, tuple[bytes, int]]) -> dict[int, int]:
        return {vertex: degree for vertex, (_, degree) in info.items()}

    @staticmethod
    def _values(info: Mapping[int, tuple[bytes, int]]) -> dict[int, bytes]:
        return {vertex: value for vertex, (value, _) in info.items()}

    def lookup(self, vertices: Sequence[int]) -> dict[int, bytes]:
        for vertex in vertices:
            self._require_vertex(vertex)
        return self._values(self._vertices(vertices))

    def neighbor_values(self, vertices: Sequence[int]) -> dict[int, bytes]:
        for vertex in vertices:
            self._require_vertex(vertex)
        info = self._vertices(vertices)
        return self._values(self._vertices(self._neighbors_of(vertices, self._degrees(info))))

    def hop_neighborhood(self, vertex: int, hops: int) -> dict[int, bytes]:
        self._require_vertex(vertex)
        self._require_count("hops", hops)
        info = self._vertices([vertex])
        frontier: list[int | None] = [vertex if vertex in info else None]
        reach = 1
        for _ in range(hops):
            reach *= self._max_degree
            reached = self._neighbors_of(frontier, self._degrees(info))
            fresh = [neighbor for neighbor in dict.fromkeys(reached) if neighbor is not None and neighbor not in info]
            width = min(reach, self._size)
            assert len(fresh) <= width
            layer = self._vertices([*fresh, *[None] * (width - len(fresh))])
            info.update(layer)
            frontier = [neighbor if neighbor in layer else None for neighbor in fresh]
            frontier += [None] * (width - len(frontier))
        return self._values(info)

    def random_walk(self, vertex: int, steps: int) -> dict[int, bytes]:
        self._require_vertex(vertex)
        self._require_count("steps", steps)
        info = self._vertices([vertex])
        current: int | None = vertex if vertex in info else None
        degrees = self._degrees(info)
        for _ in range(steps):
            layer = self._vertices(self._neighbors_of([current], degrees))
            info.update(layer)
            degrees = self._degrees(layer)
            current = secrets.choice(list(layer)) if layer else None
        return self._values(info)

    def _traverse(
        self, next_vertex: Callable[[], int | None], expand: Callable[[int, list[int]], None]
    ) -> dict[int, bytes]:
        order: dict[int, bytes] = {}
        for _ in range(self._size):
            vertex = next_vertex()
            info = self._vertices([vertex])
            found = vertex if vertex in info else None
            neighbors = self._neighbors_of([found], self._degrees(info))
            if found is not None:
                order[found] = info[found][0]
                expand(found, [neighbor for neighbor in neighbors if neighbor is not None])
        return order

    def bfs(self, source: int) -> dict[int, bytes]:
        self._require_vertex(source)
        queue = deque([source])
        discovered = {source}

        def next_vertex() -> int | None:
            return queue.popleft() if queue else None

        def expand(_: int, neighbors: list[int]) -> None:
            for neighbor in neighbors:
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    queue.append(neighbor)

        return self._traverse(next_vertex, expand)

    def dfs(self, source: int) -> dict[int, bytes]:
        self._require_vertex(source)
        visited: set[int] = set()
        stack: list[Iterator[int]] = [iter([source])]

        def next_vertex() -> int | None:
            while stack:
                for neighbor in stack[-1]:
                    if neighbor not in visited:
                        return neighbor
                del stack[-1]
            return None

        def expand(vertex: int, neighbors: list[int]) -> None:
            visited.add(vertex)
            stack.append(iter(neighbors))

        return self._traverse(next_vertex, expand)

    def insert(self, vertex: int, value: bytes, neighbors: Sequence[int]) -> None:
        self._require_vertex(vertex)
        self._require_value(value)
        self._require_neighbors(vertex, neighbors)
        if self._size >= self._config.num_data:
            raise ContractError(f"{self._identity}: the graph already holds num_data {self._config.num_data}.")

        vertex_key = _vertex_key(vertex)
        padding: list[_Slot] = [None] * (self._max_degree - len(neighbors))
        blocks = self._fetch([vertex_key, *(_vertex_key(neighbor) for neighbor in neighbors), *padding])

        problem = None
        if vertex_key in blocks:
            problem = f"vertex {vertex} already exists"
        for neighbor in neighbors:
            block = blocks.get(_vertex_key(neighbor))
            if problem is None and block is None:
                problem = f"neighbor {neighbor} does not exist"
            elif problem is None and block is not None and self._decode_vertex(block.value)[1] >= self._max_degree:
                problem = f"neighbor {neighbor} already has max_degree {self._max_degree} neighbors"
        linked: Sequence[int] = neighbors if problem is None else []
        old_degrees: dict[int, int] = {}
        for neighbor in linked:
            block = blocks[_vertex_key(neighbor)]
            neighbor_value, degree = self._decode_vertex(block.value)
            old_degrees[neighbor] = degree
            block.value = self._encode_vertex(neighbor_value, degree + 1)
        self._commit()

        entries: list[tuple[bytes, bytes] | None] = (
            [(vertex_key, self._encode_vertex(value, len(neighbors)))] if problem is None else [None]
        )
        for index, neighbor in enumerate(linked):
            entries += [
                (_edge_key(neighbor, old_degrees[neighbor]), self._id.encode(vertex)),
                (_slot_key(neighbor, vertex), self._index.encode(old_degrees[neighbor])),
                (_edge_key(vertex, index), self._id.encode(neighbor)),
                (_slot_key(vertex, neighbor), self._index.encode(index)),
            ]
        entries += [None] * (4 * (self._max_degree - len(linked)))
        self._add(entries)
        if problem is not None:
            raise ValueError(f"{self._identity}: {problem}.")
        self._size += 1

    def delete(self, vertex: int) -> None:
        self._require_vertex(vertex)
        vertex_key = _vertex_key(vertex)
        blocks = self._fetch([vertex_key])
        exists = vertex_key in blocks
        degree = self._decode_vertex(blocks[vertex_key].value)[1] if exists else 0
        self._commit()

        slots = self._neighbors_of([vertex], {vertex: degree})
        neighbors = [neighbor for neighbor in slots if neighbor is not None]

        blocks = self._fetch(
            [None if u is None else _vertex_key(u) for u in slots]
            + [None if u is None else _slot_key(u, vertex) for u in slots]
        )
        position: dict[int, int] = {}
        last: dict[int, int] = {}
        for neighbor in neighbors:
            block = blocks[_vertex_key(neighbor)]
            neighbor_value, neighbor_degree = self._decode_vertex(block.value)
            block.value = self._encode_vertex(neighbor_value, neighbor_degree - 1)
            last[neighbor] = neighbor_degree - 1
            position[neighbor] = self._index.decode(blocks[_slot_key(neighbor, vertex)].value)
        self._commit()

        moved_keys: list[_Slot] = [
            None if u is None or position[u] == last[u] else _edge_key(u, position[u]) for u in slots
        ]
        last_keys: list[_Slot] = [None if u is None or position[u] == last[u] else _edge_key(u, last[u]) for u in slots]
        blocks = self._fetch(moved_keys + last_keys)
        moved: dict[int, int] = {}
        for neighbor in neighbors:
            if position[neighbor] != last[neighbor]:
                tail = self._id.decode(blocks[_edge_key(neighbor, last[neighbor])].value)
                blocks[_edge_key(neighbor, position[neighbor])].value = self._id.encode(tail)
                moved[neighbor] = tail
        self._commit()

        blocks = self._fetch([_slot_key(u, moved[u]) if u in moved else None for u in slots])
        for neighbor, tail in moved.items():
            blocks[_slot_key(neighbor, tail)].value = self._index.encode(position[neighbor])
        self._commit()

        self._remove(
            [vertex_key if exists else None]
            + [None if u is None else _edge_key(vertex, index) for index, u in enumerate(slots)]
            + [None if u is None else _slot_key(vertex, u) for u in slots]
            + [None if u is None else _edge_key(u, last[u]) for u in slots]
            + [None if u is None else _slot_key(u, vertex) for u in slots]
        )
        if not exists:
            raise KeyError(f"{self._identity}: vertex {vertex} does not exist.")
        self._size -= 1
