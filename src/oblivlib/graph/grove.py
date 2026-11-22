import secrets
import struct
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, override

from oblivlib.dependency import ContractError, Data, PathData
from oblivlib.dependency.config import ODS_OPS, AvlOmapConfig, GroveConfig, OdsOp, OramConfig
from oblivlib.dependency.contract import require_value
from oblivlib.dependency.heap_index import compute_level
from oblivlib.dependency.layout import Bytes, Record, Seq, UInt
from oblivlib.dependency.tree_storage_base import KeyedStore, open_paths_together
from oblivlib.graph.meta import DupStore, meta_bucket_size
from oblivlib.omap import AVLOmap

ID_SIZE = 8
_ID = struct.Struct(">Q")
_NEIGHBOR = b"N"
_POSITION = b"P"

Placed = dict[bytes, tuple[int, bytes]]
Target = tuple[int, int] | None


def _key(vertex: int) -> bytes:
    return _ID.pack(vertex)


def _vertex(key: bytes) -> int:
    return _ID.unpack(key[:ID_SIZE])[0]


@dataclass
class Vertex:
    value: bytes
    position: int
    neighbors: dict[int, int] = field(default_factory=dict)


class GroveIndex(AVLOmap):
    def __init__(self, config: AvlOmapConfig, *, meta: DupStore):
        super().__init__(config)
        self._meta = meta
        self.placed: Placed = {}

    def _record(self, blocks: Sequence[Any]) -> None:
        for block in blocks:
            self.placed[block.key] = (block.leaf, block.value.value)

    @override
    def _open_paths(self, leaves: list[int]) -> PathData:
        own, meta = open_paths_together([(self, leaves), (self._meta, leaves)])
        self._meta._absorb(meta)
        return own

    @override
    def _fetch_nodes(self, requests: Sequence[tuple[Any, int | None, Any]]) -> None:
        super()._fetch_nodes(requests)
        for key, leaf, _ in requests:
            assert leaf is not None
            moved = self._meta.take(lambda block_key, key=key: block_key == key, leaf)
            if moved:
                self._local.require(key).value.value = moved[0].value

    @override
    def _return_to_stash(self, blocks: Sequence[Any]) -> None:
        self._record(blocks)
        super()._return_to_stash(blocks)

    @override
    def _build_ods_blocks(self, data: list[Any]) -> tuple[list[Data], Any]:
        blocks, root = super()._build_ods_blocks(data)
        self._record(blocks)
        return blocks, root


class Grove:
    def __init__(self, config: GroveConfig):
        self._config: GroveConfig = config
        self._max_deg: int = config.max_deg
        self._width: int = config.round_width
        self._num_vertices: int = 0
        level = compute_level(config.num_vertices)
        index_batch = config.max_deg + 1

        self._graph = KeyedStore(
            OramConfig(
                num_data=config.num_vertices,
                data_size=self._record.size,
                client=config.client,
                name=f"{config.name}_graph",
                bucket_size=config.bucket_size,
                stash_scale=config.stash_scale * self._width,
                encryptor=config.encryptor,
            ),
            key_size=ID_SIZE,
        )
        self._position_meta = DupStore(
            OramConfig(
                num_data=config.num_vertices,
                data_size=self._graph._leaf_field.size,
                client=config.client,
                name=f"{config.name}_position_meta",
                bucket_size=config.meta_bucket_size
                or meta_bucket_size(insertions=config.num_rounds * self._width, group=1, level=level),
                stash_scale=config.stash_scale * self._width,
                encryptor=config.encryptor,
            ),
            key_size=ID_SIZE,
        )
        self._index = GroveIndex(
            AvlOmapConfig(
                num_data=config.num_vertices,
                key_size=ID_SIZE,
                data_size=self._graph._leaf_field.size,
                client=config.client,
                name=f"{config.name}_index",
                bucket_size=config.bucket_size,
                stash_scale=config.stash_scale,
                encryptor=config.encryptor,
                distinguishable=True,
                batch_size=index_batch,
            ),
            meta=self._position_meta,
        )
        self._index_paths: dict[OdsOp, int] = {
            op: sum(
                self._index._round_width(r)
                for r in range(self._index._config.budget(op).rounds(self._index._max_height))
            )
            for op in ODS_OPS
        }
        round_insertions = self._width * self._max_deg + max(self._index_paths.values())
        self._graph_meta = DupStore(
            OramConfig(
                num_data=config.num_vertices,
                data_size=self._meta_value.size,
                client=config.client,
                name=f"{config.name}_graph_meta",
                bucket_size=config.meta_bucket_size
                or meta_bucket_size(
                    insertions=config.num_rounds * round_insertions,
                    group=self._max_deg + 1,
                    level=level,
                ),
                stash_scale=config.stash_scale * round_insertions,
                encryptor=config.encryptor,
            ),
            key_size=2 * ID_SIZE + 1,
        )

    @cached_property
    def _leaf(self) -> UInt:
        return UInt.covering(self._tombstone - 1)

    @cached_property
    def _record(self) -> Record:
        neighbor = Record((UInt(8 * ID_SIZE), self._leaf))
        return Record((Bytes(self._config.data_size), self._leaf, Seq(neighbor, self._max_deg)))

    @cached_property
    def _meta_value(self) -> UInt:
        return UInt.covering(self._tombstone)

    @property
    def _tombstone(self) -> int:
        return 1 << (compute_level(self._config.num_vertices) - 1)

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._config.name!r}"

    @property
    def num_vertices(self) -> int:
        return self._num_vertices

    def _encode(self, vertex: Vertex) -> bytes:
        return self._record.encode((vertex.value, vertex.position, list(vertex.neighbors.items())))

    def _decode(self, data: bytes) -> Vertex:
        value, position, neighbors = self._record.decode(data)
        return Vertex(value=value, position=position, neighbors=dict(neighbors))

    def _require_vertex(self, vertex: object) -> int:
        if type(vertex) is not int or not 0 <= vertex < 1 << (8 * ID_SIZE):
            raise ContractError(f"{self._identity}: a vertex must be an int in [0, 2**64), got {vertex!r}.")
        return vertex

    def _require_neighbors(self, vertex: int, neighbors: Sequence[int]) -> None:
        for neighbor in neighbors:
            self._require_vertex(neighbor)
        if len(neighbors) > self._max_deg:
            raise ContractError(
                f"{self._identity}: vertex {vertex} has {len(neighbors)} neighbors, max_deg is {self._max_deg}."
            )
        if len(set(neighbors)) != len(neighbors) or vertex in neighbors:
            raise ContractError(f"{self._identity}: vertex {vertex} lists a neighbor twice or itself.")

    def _require_count(self, name: str, count: object) -> int:
        if type(count) is not int or count < 0:
            raise ContractError(f"{self._identity}: {name} must be an int >= 0, got {count!r}.")
        return count

    def init_server_storage(
        self, graph: Mapping[int, Sequence[int]] | None = None, values: Mapping[int, bytes] | None = None
    ) -> None:
        graph = graph or {}
        values = values or {}
        if len(graph) > self._config.num_vertices:
            raise ContractError(f"{self._identity}: {len(graph)} vertices exceed num_vertices.")
        for vertex, neighbors in graph.items():
            self._require_vertex(vertex)
            self._require_neighbors(vertex, neighbors)
            for neighbor in neighbors:
                if vertex not in graph.get(neighbor, ()):
                    raise ContractError(f"{self._identity}: edge ({vertex}, {neighbor}) is not listed both ways.")
        for vertex, value in values.items():
            if vertex not in graph:
                raise ContractError(f"{self._identity}: value given for vertex {vertex}, which is not in the graph.")
            require_value(self._identity, value, self._config.data_size)

        leaves = {vertex: self._graph._get_new_leaf() for vertex in graph}
        self._index.placed = {}
        self._index.init_server_storage([(_key(v), self._leaf.encode(leaf)) for v, leaf in leaves.items()])
        blocks = [
            Data(
                key=_key(vertex),
                leaf=leaves[vertex],
                value=self._encode(
                    Vertex(
                        value=values.get(vertex, b""),
                        position=self._index.placed[_key(vertex)][0],
                        neighbors={neighbor: leaves[neighbor] for neighbor in neighbors},
                    )
                ),
            )
            for vertex, neighbors in graph.items()
        ]
        self._graph._host_tree(self._graph._build_tree(blocks))
        for store in (self._graph_meta, self._position_meta):
            store._host_tree(store._build_tree([]))
        self._index.placed = {}
        self._num_vertices = len(graph)

    def _padded(self, store: KeyedStore, leaves: Sequence[int], width: int) -> list[int]:
        paths = set(leaves)
        while len(paths) < min(width, store._leaf_range):
            paths.add(store._get_new_leaf())
        return sorted(paths)

    def _chunks[T](self, items: Sequence[T]) -> list[Sequence[T]]:
        return [items[start : start + self._width] for start in range(0, len(items), self._width)]

    def _neighbor_dup(self, target: int, leaf: int, source: int, value: int) -> Data:
        return Data(key=_key(target) + _NEIGHBOR + _key(source), leaf=leaf, value=self._meta_value_bytes(value))

    def _position_dup(self, target: int, leaf: int, position: int) -> Data:
        return Data(key=_key(target) + _POSITION, leaf=leaf, value=self._meta_value_bytes(position))

    def _meta_value_bytes(self, number: int) -> bytes:
        return self._meta_value.encode(number)

    def _apply_pending(self, vertex_id: int, vertex: Vertex, leaf: int) -> None:
        prefix = _key(vertex_id)
        for dup in self._graph_meta.take(lambda key: key[:ID_SIZE] == prefix, leaf):
            number = self._meta_value.decode(dup.value)
            if dup.key[ID_SIZE : ID_SIZE + 1] == _POSITION:
                vertex.position = number
            elif number == self._tombstone:
                vertex.neighbors.pop(_vertex(dup.key[ID_SIZE + 1 :]), None)
            else:
                source = _vertex(dup.key[ID_SIZE + 1 :])
                if source in vertex.neighbors:
                    vertex.neighbors[source] = number

    def _access(
        self,
        targets: Sequence[Target],
        placed: Placed,
        *,
        index_paths: int = 0,
        edit: Callable[[dict[int, Vertex]], set[int]] | None = None,
    ) -> tuple[dict[int, Vertex], dict[int, int]]:
        real = {vertex: leaf for vertex, leaf in filter(None, targets)}
        width = len(targets)
        graph_leaves = self._padded(self._graph, list(real.values()), width)
        graph_meta_leaves = graph_leaves + self._graph_meta.rl_leaves(width * self._max_deg + index_paths)
        position_leaves = self._position_meta.rl_leaves(width)
        graph_path, graph_meta_path, position_path = open_paths_together(
            [(self._graph, graph_leaves), (self._graph_meta, graph_meta_leaves), (self._position_meta, position_leaves)]
        )
        self._graph._absorb(graph_path)
        self._graph_meta._absorb(graph_meta_path)
        self._position_meta._absorb(position_path)

        blocks = {
            _vertex(block.key): block
            for block in self._graph._stash
            if _vertex(block.key) in real and block.leaf == real[_vertex(block.key)]
        }
        vertices: dict[int, Vertex] = {}
        for vertex_id, block in blocks.items():
            vertices[vertex_id] = self._decode(block.value)
            self._apply_pending(vertex_id, vertices[vertex_id], real[vertex_id])

        removed = edit(vertices) if edit is not None else set()
        created = {vertex_id for vertex_id in vertices if vertex_id not in blocks}
        for vertex_id in created:
            blocks[vertex_id] = Data(key=_key(vertex_id))
            self._graph._stash.append(blocks[vertex_id])

        new_leaves = {vertex_id: self._graph._get_new_leaf() for vertex_id in vertices if vertex_id not in removed}
        graph_dups: list[Data] = []
        position_dups: list[Data] = []
        for vertex_id, vertex in vertices.items():
            announced = self._tombstone if vertex_id in removed else new_leaves[vertex_id]
            for neighbor, neighbor_leaf in vertex.neighbors.items():
                if neighbor in vertices:
                    if vertex_id in vertices[neighbor].neighbors:
                        if vertex_id in removed:
                            del vertices[neighbor].neighbors[vertex_id]
                        else:
                            vertices[neighbor].neighbors[vertex_id] = announced
                else:
                    graph_dups.append(self._neighbor_dup(neighbor, neighbor_leaf, vertex_id, announced))
        for vertex_id, vertex in vertices.items():
            if vertex_id in removed:
                continue
            entry = placed.get(_key(vertex_id))
            if entry is not None:
                vertex.position = entry[0]
            if vertex_id not in created:
                position_dups.append(
                    Data(key=_key(vertex_id), leaf=vertex.position, value=self._leaf.encode(new_leaves[vertex_id]))
                )
        for key, (position, graph_leaf) in placed.items():
            if _vertex(key) not in vertices:
                graph_dups.append(self._position_dup(_vertex(key), self._leaf.decode(graph_leaf), position))

        self._graph._stash = [block for block in self._graph._stash if _vertex(block.key) not in removed]
        for vertex_id, vertex in vertices.items():
            if vertex_id not in removed:
                blocks[vertex_id].leaf = new_leaves[vertex_id]
                blocks[vertex_id].value = self._encode(vertex)
        self._graph._check_stash()
        self._graph_meta.add(graph_dups)
        self._position_meta.add(position_dups)
        for store in (self._graph, self._graph_meta, self._position_meta):
            store._write_paths()
        return vertices, new_leaves

    def _deliver_index_moves(self, placed: Placed, op: OdsOp) -> None:
        dups = [
            self._position_dup(_vertex(key), self._leaf.decode(graph_leaf), position)
            for key, (position, graph_leaf) in placed.items()
        ]
        self._graph_meta._read_paths(self._graph_meta.rl_leaves(self._index_paths[op]))
        self._graph_meta.add(dups)
        self._graph_meta._write_paths()

    def _locate(self, vertices: Sequence[int | None]) -> tuple[list[Target], Placed]:
        self._index.placed = {}
        pairs: list[tuple[Any, Any]] = [(None if v is None else _key(v), None) for v in vertices]
        found = self._index.search_batch(pairs)
        placed, self._index.placed = self._index.placed, {}
        return [
            None if vertex is None or leaf is None else (vertex, self._leaf.decode(leaf))
            for vertex, leaf in zip(vertices, found, strict=True)
        ], placed

    def _lookup_chunk(self, vertices: Sequence[int | None]) -> tuple[dict[int, Vertex], dict[int, int]]:
        targets, placed = self._locate(vertices)
        seen: set[int] = set()
        unique: list[Target] = []
        for target in targets:
            unique.append(target if target is not None and target[0] not in seen else None)
            if target is not None:
                seen.add(target[0])
        return self._access(unique, placed, index_paths=self._index_paths["search"])

    @staticmethod
    def _gather(found: dict[int, Vertex], layer: Mapping[int, Vertex], moved: Mapping[int, int]) -> None:
        for vertex in found.values():
            for neighbor in vertex.neighbors:
                if neighbor in moved:
                    vertex.neighbors[neighbor] = moved[neighbor]
        found.update(layer)

    def _lookup_all(self, vertices: Sequence[int]) -> dict[int, Vertex]:
        found: dict[int, Vertex] = {}
        for start in range(0, len(vertices), self._index._batch_size):
            self._gather(found, *self._lookup_chunk(vertices[start : start + self._index._batch_size]))
        return found

    def _visit(self, targets: Sequence[Target]) -> dict[int, Vertex]:
        found: dict[int, Vertex] = {}
        for chunk in self._chunks(targets):
            self._gather(found, *self._access(chunk, {}))
        return found

    def _layer(self, frontier: Mapping[int, Vertex], width: int, skip: set[int]) -> list[Target]:
        fresh: dict[int, int] = {}
        for vertex in frontier.values():
            for neighbor, leaf in vertex.neighbors.items():
                if neighbor not in skip:
                    fresh.setdefault(neighbor, leaf)
        width = min(width, self._num_vertices)
        assert len(fresh) <= width
        return [*fresh.items(), *[None] * (width - len(fresh))]

    def lookup(self, vertices: Sequence[int]) -> dict[int, bytes]:
        for vertex in vertices:
            self._require_vertex(vertex)
        return {vertex_id: vertex.value for vertex_id, vertex in self._lookup_all(vertices).items()}

    def neighbor_ids(self, vertices: Sequence[int]) -> dict[int, list[int]]:
        for vertex in vertices:
            self._require_vertex(vertex)
        return {vertex_id: list(vertex.neighbors) for vertex_id, vertex in self._lookup_all(vertices).items()}

    def neighbor_values(self, vertices: Sequence[int]) -> dict[int, bytes]:
        for vertex in vertices:
            self._require_vertex(vertex)
        centers = self._lookup_all(vertices)
        width = len(vertices) * self._max_deg
        slots = self._layer({v: centers[v] for v in dict.fromkeys(vertices) if v in centers}, width, set())
        return {vertex_id: vertex.value for vertex_id, vertex in self._visit(slots).items()}

    def hop_neighborhood(self, vertex: int, hops: int) -> dict[int, bytes]:
        self._require_vertex(vertex)
        self._require_count("hops", hops)
        found = self._lookup_all([vertex])
        frontier = dict(found)
        for hop in range(1, hops + 1):
            layer = self._visit(self._layer(frontier, self._max_deg**hop, set(found)))
            found.update(layer)
            frontier = layer
        return {vertex_id: v.value for vertex_id, v in found.items()}

    def random_walk(self, vertex: int, steps: int) -> dict[int, bytes]:
        self._require_vertex(vertex)
        self._require_count("steps", steps)
        found = self._lookup_all([vertex])
        current = dict(found)
        for _ in range(steps):
            layer = self._visit(self._layer(current, self._max_deg, set()))
            found.update(layer)
            step = secrets.choice(list(layer)) if layer else None
            current = {step: layer[step]} if step is not None else {}
        return {vertex_id: v.value for vertex_id, v in found.items()}

    def _traverse(
        self, source: int, next_vertex: Callable[[], Target], expand: Callable[[int, Vertex], None]
    ) -> dict[int, bytes]:
        self._require_vertex(source)
        order: dict[int, bytes] = {}
        found = self._lookup_all([source])
        for step in range(self._num_vertices):
            if step:
                found = self._access([next_vertex()], {})[0]
            for vertex_id, vertex in found.items():
                order[vertex_id] = vertex.value
                expand(vertex_id, vertex)
        return order

    def bfs(self, source: int) -> dict[int, bytes]:
        queue: deque[tuple[int, int]] = deque()
        discovered = {source}

        def next_vertex() -> Target:
            return queue.popleft() if queue else None

        def expand(_: int, vertex: Vertex) -> None:
            for neighbor, leaf in vertex.neighbors.items():
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    queue.append((neighbor, leaf))

        return self._traverse(source, next_vertex, expand)

    def dfs(self, source: int) -> dict[int, bytes]:
        visited: set[int] = set()
        stack: list[Iterator[tuple[int, int]]] = []

        def next_vertex() -> Target:
            while stack:
                for neighbor, leaf in stack[-1]:
                    if neighbor not in visited:
                        return neighbor, leaf
                stack.pop()
            return None

        def expand(vertex_id: int, vertex: Vertex) -> None:
            visited.add(vertex_id)
            stack.append(iter(list(vertex.neighbors.items())))

        return self._traverse(source, next_vertex, expand)

    def _finish_index_op(self, op: OdsOp, run: Callable[[], Any]) -> Any:
        self._index.placed = {}
        result = run()
        placed, self._index.placed = self._index.placed, {}
        self._deliver_index_moves(placed, op)
        return result

    def insert(self, vertex: int, value: bytes, neighbors: Sequence[int]) -> None:
        self._require_vertex(vertex)
        require_value(self._identity, value, self._config.data_size)
        self._require_neighbors(vertex, neighbors)
        if self._num_vertices >= self._config.num_vertices:
            raise ContractError(f"{self._identity}: the graph already holds num_vertices {self._config.num_vertices}.")

        padding: list[int | None] = [None] * (self._max_deg - len(neighbors))
        targets, placed = self._locate([vertex, *neighbors, *padding])
        problem = None
        if targets[0] is not None:
            problem = f"vertex {vertex} already exists"
        elif any(target is None for target in targets[1 : 1 + len(neighbors)]):
            problem = "a neighbor does not exist"

        def link(found: dict[int, Vertex]) -> set[int]:
            nonlocal problem
            if problem is None and any(len(found[n].neighbors) >= self._max_deg for n in neighbors):
                problem = f"a neighbor already has max_deg {self._max_deg} neighbors"
            if problem is None:
                found[vertex] = Vertex(value=value, position=0, neighbors=dict.fromkeys(neighbors, 0))
                for neighbor in neighbors:
                    found[neighbor].neighbors[vertex] = 0
            return set()

        _, new_leaves = self._access([None, *targets[1:]], placed, index_paths=self._index_paths["search"], edit=link)
        if problem is not None:
            self._finish_index_op("insert", lambda: self._index.insert(None))
            raise ValueError(f"{self._identity}: {problem}.")
        self._finish_index_op("insert", lambda: self._index.insert(_key(vertex), self._leaf.encode(new_leaves[vertex])))
        self._num_vertices += 1

    def delete(self, vertex: int) -> None:
        self._require_vertex(vertex)
        targets, placed = self._locate([vertex])
        self._access(targets, placed, index_paths=self._index_paths["search"], edit=lambda found: set(found))
        if targets[0] is None:
            self._finish_index_op("delete", lambda: self._index.delete(None))
            raise KeyError(f"{self._identity}: vertex {vertex} does not exist.")
        self._finish_index_op("delete", lambda: self._index.delete(_key(vertex)))
        self._num_vertices -= 1

    def update(self, vertex: int, value: bytes) -> None:
        self._require_vertex(vertex)
        require_value(self._identity, value, self._config.data_size)
        targets, placed = self._locate([vertex])

        def change(found: dict[int, Vertex]) -> set[int]:
            if vertex in found:
                found[vertex].value = value
            return set()

        found, _ = self._access(targets, placed, index_paths=self._index_paths["search"], edit=change)
        if vertex not in found:
            raise KeyError(f"{self._identity}: vertex {vertex} does not exist.")

    def _edit_edge(self, u: int, v: int, *, link: bool) -> None:
        self._require_vertex(u)
        self._require_vertex(v)
        if u == v:
            raise ContractError(f"{self._identity}: an edge needs two distinct vertices, got {u} twice.")
        targets, placed = self._locate([u, v])
        problem = None

        def change(found: dict[int, Vertex]) -> set[int]:
            nonlocal problem
            if u not in found or v not in found:
                problem = "an endpoint does not exist"
            elif link and v in found[u].neighbors:
                problem = f"vertices {u} and {v} are already adjacent"
            elif link and max(len(found[u].neighbors), len(found[v].neighbors)) >= self._max_deg:
                problem = f"an endpoint already has max_deg {self._max_deg} neighbors"
            elif not link and v not in found[u].neighbors:
                problem = f"vertices {u} and {v} are not adjacent"
            elif link:
                found[u].neighbors[v] = 0
                found[v].neighbors[u] = 0
            else:
                del found[u].neighbors[v]
                del found[v].neighbors[u]
            return set()

        self._access(targets, placed, index_paths=self._index_paths["search"], edit=change)
        if problem is not None:
            raise ValueError(f"{self._identity}: {problem}.")

    def add_edge(self, u: int, v: int) -> None:
        self._edit_edge(u, v, link=True)

    def remove_edge(self, u: int, v: int) -> None:
        self._edit_edge(u, v, link=False)
