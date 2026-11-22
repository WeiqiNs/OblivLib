import secrets
import struct
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cached_property
from typing import Any

from oblivlib.dependency import ContractError, Data
from oblivlib.dependency.config import AvlOmapConfig, GraphOsConfig, OramConfig
from oblivlib.dependency.contract import require_value
from oblivlib.dependency.layout import Bytes, Field, Record, UInt
from oblivlib.dependency.tree_storage_base import KeyedStore
from oblivlib.omap import AVLOmap

KEY_SIZE = 17
_ID_BITS = 64
_ID = struct.Struct(">Q")

Slot = bytes | None


def _vertex_key(vertex: int) -> bytes:
    return b"V" + _ID.pack(vertex)


def _edge_key(vertex: int, index: int) -> bytes:
    return b"E" + _ID.pack(vertex) + _ID.pack(index)


def _slot_key(vertex: int, neighbor: int) -> bytes:
    return b"P" + _ID.pack(vertex) + _ID.pack(neighbor)


class GraphOS:
    def __init__(self, config: GraphOsConfig):
        self._config: GraphOsConfig = config
        self._max_deg: int = config.max_deg
        self._batch: int = 2 * config.max_deg + 1
        self._num_vertices: int = 0

        self._store = KeyedStore(
            OramConfig(
                num_data=config.num_entries,
                data_size=self._value_width,
                client=config.client,
                name=f"{config.name}_values",
                bucket_size=config.bucket_size,
                stash_scale=config.stash_scale * self._batch,
                encryptor=config.encryptor,
            ),
            key_size=KEY_SIZE,
        )
        self._omap = AVLOmap(
            AvlOmapConfig(
                num_data=config.num_entries,
                key_size=KEY_SIZE,
                data_size=self._leaf.size,
                client=config.client,
                name=f"{config.name}_omap",
                bucket_size=config.bucket_size,
                stash_scale=config.stash_scale,
                encryptor=config.encryptor,
                distinguishable=True,
                batch_size=self._batch,
            )
        )

    @cached_property
    def _vertex_record(self) -> Record:
        return Record((Bytes(self._config.data_size), UInt.covering(self._max_deg)))

    @cached_property
    def _id(self) -> Field[int]:
        return UInt(_ID_BITS)

    @cached_property
    def _index(self) -> Field[int]:
        return UInt.covering(self._max_deg - 1)

    @property
    def _value_width(self) -> int:
        return max(self._vertex_record.size, self._id.size, self._index.size)

    @property
    def _leaf(self) -> Field[int]:
        return self._store._leaf_field

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._config.name!r}"

    @property
    def num_vertices(self) -> int:
        return self._num_vertices

    def _require_vertex(self, vertex: object) -> int:
        if type(vertex) is not int or not 0 <= vertex < 1 << _ID_BITS:
            raise ContractError(f"{self._identity}: a vertex must be an int in [0, 2**64), got {vertex!r}.")
        return vertex

    def _require_count(self, name: str, count: object) -> int:
        if type(count) is not int or count < 0:
            raise ContractError(f"{self._identity}: {name} must be an int >= 0, got {count!r}.")
        return count

    def _require_value(self, value: object) -> None:
        require_value(self._identity, value, self._config.data_size)

    def _require_neighbors(self, vertex: int, neighbors: Sequence[int]) -> None:
        for neighbor in neighbors:
            self._require_vertex(neighbor)
        if len(neighbors) > self._max_deg:
            raise ContractError(
                f"{self._identity}: vertex {vertex} has {len(neighbors)} neighbors, max_deg is {self._max_deg}."
            )
        if len(set(neighbors)) != len(neighbors):
            raise ContractError(f"{self._identity}: vertex {vertex} lists a neighbor twice.")
        if vertex in neighbors:
            raise ContractError(f"{self._identity}: vertex {vertex} lists itself as a neighbor.")

    def _require_graph(self, graph: Mapping[int, Sequence[int]], values: Mapping[int, bytes]) -> None:
        if len(graph) > self._config.num_vertices:
            raise ContractError(
                f"{self._identity}: {len(graph)} vertices exceed num_vertices {self._config.num_vertices}."
            )
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

        leaves = {key: self._store._get_new_leaf() for key, _ in entries}
        self._omap.init_server_storage([(key, self._leaf.encode(leaf)) for key, leaf in leaves.items()])
        self._store._host_tree(
            self._store._build_tree(Data(key=key, leaf=leaves[key], value=value) for key, value in entries)
        )
        self._num_vertices = len(graph)

    def _padded(self, leaves: Sequence[int], width: int) -> list[int]:
        paths = set(leaves)
        while len(paths) < width:
            paths.add(self._store._get_new_leaf())
        return sorted(paths)

    def _chunks[T](self, items: Sequence[T]) -> list[Sequence[T]]:
        return [items[start : start + self._batch] for start in range(0, len(items), self._batch)]

    def _fetch(self, keys: Sequence[Slot]) -> dict[bytes, Data]:
        new_leaves: dict[bytes, int] = {}
        pairs: list[tuple[Any, Any]] = []
        for key in keys:
            if key is None:
                pairs.append((None, None))
                continue
            new_leaves[key] = self._store._get_new_leaf()
            pairs.append((key, self._leaf.encode(new_leaves[key])))

        located: dict[bytes, int] = {}
        old_leaves: list[int] = []
        for (key, _), old_leaf in zip(pairs, self._omap.search_batch(pairs), strict=True):
            if key is not None and old_leaf is not None:
                located[key] = new_leaves[key]
                old_leaves.append(self._leaf.decode(old_leaf))

        self._store._read_paths(self._padded(old_leaves, len(keys)))
        blocks = {block.key: block for block in self._store._stash if block.key in located}
        for key, leaf in located.items():
            blocks[key].leaf = leaf
        return blocks

    def _commit(self) -> None:
        self._store._write_paths()

    def _read(self, keys: Sequence[Slot]) -> dict[bytes, bytes]:
        values: dict[bytes, bytes] = {}
        for chunk in self._chunks(keys):
            values.update({key: block.value for key, block in self._fetch(chunk).items()})
            self._commit()
        return values

    def _add(self, entries: Sequence[tuple[bytes, bytes] | None]) -> None:
        for chunk in self._chunks(entries):
            placed = [(key, value, self._store._get_new_leaf()) for key, value in filter(None, chunk)]
            self._omap.insert_batch([(key, self._leaf.encode(leaf)) for key, _, leaf in placed])
            self._store._read_paths(self._padded([], len(chunk)))
            self._store._stash += [Data(key=key, leaf=leaf, value=value) for key, value, leaf in placed]
            self._store._check_stash()
            self._commit()

    def _remove(self, keys: Sequence[Slot]) -> None:
        for chunk in self._chunks(keys):
            old_leaves = [self._omap.delete(key) for key in chunk]
            doomed = {key for key in chunk if key is not None}
            self._store._read_paths(
                self._padded([self._leaf.decode(leaf) for leaf in old_leaves if leaf is not None], len(chunk))
            )
            self._store._stash = [block for block in self._store._stash if block.key not in doomed]
            self._commit()

    def _vertices(self, vertices: Sequence[int | None]) -> dict[int, tuple[bytes, int]]:
        found = self._read([None if vertex is None else _vertex_key(vertex) for vertex in vertices])
        return {
            vertex: self._decode_vertex(found[_vertex_key(vertex)])
            for vertex in vertices
            if vertex is not None and _vertex_key(vertex) in found
        }

    def _neighbors_of(self, frontier: Sequence[int | None], degrees: Mapping[int, int]) -> list[int | None]:
        keys: list[Slot] = [
            _edge_key(vertex, index) if vertex is not None and index < degrees.get(vertex, 0) else None
            for vertex in frontier
            for index in range(self._max_deg)
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
        for _ in range(hops):
            fresh: list[int | None] = []
            for neighbor in self._neighbors_of(frontier, self._degrees(info)):
                keep = neighbor is not None and neighbor not in info and neighbor not in fresh
                fresh.append(neighbor if keep else None)
            layer = self._vertices(fresh)
            info.update(layer)
            frontier = [neighbor if neighbor in layer else None for neighbor in fresh]
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
        for _ in range(self._num_vertices):
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
                stack.pop()
            return None

        def expand(vertex: int, neighbors: list[int]) -> None:
            visited.add(vertex)
            stack.append(iter(neighbors))

        return self._traverse(next_vertex, expand)

    def insert(self, vertex: int, value: bytes, neighbors: Sequence[int]) -> None:
        self._require_vertex(vertex)
        self._require_value(value)
        self._require_neighbors(vertex, neighbors)
        if self._num_vertices >= self._config.num_vertices:
            raise ContractError(f"{self._identity}: the graph already holds num_vertices {self._config.num_vertices}.")

        vertex_key = _vertex_key(vertex)
        padding: list[Slot] = [None] * (self._max_deg - len(neighbors))
        blocks = self._fetch([vertex_key, *(_vertex_key(neighbor) for neighbor in neighbors), *padding])

        problem = None
        if vertex_key in blocks:
            problem = f"vertex {vertex} already exists"
        for neighbor in neighbors:
            block = blocks.get(_vertex_key(neighbor))
            if problem is None and block is None:
                problem = f"neighbor {neighbor} does not exist"
            elif problem is None and block is not None and self._decode_vertex(block.value)[1] >= self._max_deg:
                problem = f"neighbor {neighbor} already has max_deg {self._max_deg} neighbors"
        if problem is not None:
            self._commit()
            raise ValueError(f"{self._identity}: {problem}.")

        old_degrees: dict[int, int] = {}
        for neighbor in neighbors:
            block = blocks[_vertex_key(neighbor)]
            neighbor_value, degree = self._decode_vertex(block.value)
            old_degrees[neighbor] = degree
            block.value = self._encode_vertex(neighbor_value, degree + 1)
        self._commit()

        entries: list[tuple[bytes, bytes] | None] = [(vertex_key, self._encode_vertex(value, len(neighbors)))]
        for index, neighbor in enumerate(neighbors):
            entries += [
                (_edge_key(neighbor, old_degrees[neighbor]), self._id.encode(vertex)),
                (_slot_key(neighbor, vertex), self._index.encode(old_degrees[neighbor])),
                (_edge_key(vertex, index), self._id.encode(neighbor)),
                (_slot_key(vertex, neighbor), self._index.encode(index)),
            ]
        entries += [None] * (4 * (self._max_deg - len(neighbors)))
        self._add(entries)
        self._num_vertices += 1

    def delete(self, vertex: int) -> None:
        self._require_vertex(vertex)
        vertex_key = _vertex_key(vertex)
        blocks = self._fetch([vertex_key])
        if vertex_key not in blocks:
            self._commit()
            raise KeyError(f"{self._identity}: vertex {vertex} does not exist.")
        _, degree = self._decode_vertex(blocks[vertex_key].value)
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

        moved_keys: list[Slot] = [
            None if u is None or position[u] == last[u] else _edge_key(u, position[u]) for u in slots
        ]
        last_keys: list[Slot] = [None if u is None or position[u] == last[u] else _edge_key(u, last[u]) for u in slots]
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
            [vertex_key]
            + [None if u is None else _edge_key(vertex, index) for index, u in enumerate(slots)]
            + [None if u is None else _slot_key(vertex, u) for u in slots]
            + [None if u is None else _edge_key(u, last[u]) for u in slots]
            + [None if u is None else _slot_key(u, vertex) for u in slots]
        )
        self._num_vertices -= 1
