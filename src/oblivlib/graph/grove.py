import secrets
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import override

from oblivlib.dependency.avl_tree import AVLNode
from oblivlib.dependency.config import ODS_OPS, AvlOmapConfig, GroveConfig, OdsOp
from oblivlib.dependency.contract import (
    require_graph,
    require_neighbors,
    require_room,
    require_value,
    require_vertex,
)
from oblivlib.dependency.errors import ContractError
from oblivlib.dependency.heap_index import leaf_count
from oblivlib.dependency.keyed_store import KeyedStore
from oblivlib.dependency.layout import Bytes, Col, Pair, Record, Seq, UInt
from oblivlib.dependency.tree_storage_base import leaf_field, read_together
from oblivlib.dependency.types import KeyedBlock
from oblivlib.graph.meta import DupStore, meta_bucket_size
from oblivlib.omap import AVLOmap
from oblivlib.omap.base_omap import KvList
from oblivlib.omap.ost_base_omap import Root

_ID_SIZE = 8
_NEIGHBOR = b"N"
_POSITION = b"P"

_Placed = dict[bytes, tuple[int, bytes]]
_Target = tuple[int, int] | None


def _key(vertex: int) -> bytes:
    return vertex.to_bytes(_ID_SIZE, "big")


def _vertex(key: bytes) -> int:
    return int.from_bytes(key[:_ID_SIZE], "big")


@dataclass
class _Vertex:
    value: bytes
    position: int
    neighbors: dict[int, int] = field(default_factory=dict)


@dataclass
class _StoredVertex:
    value: bytes
    position: int
    neighbors: list[tuple[int, int]]


class _GroveIndex(AVLOmap):
    def __init__(self, config: AvlOmapConfig, *, meta: DupStore):
        super().__init__(config)
        self._meta: DupStore = meta
        self.placed: _Placed = {}

    def op_paths(self, op: OdsOp) -> int:
        rounds = self._config.budget(op).rounds(self._max_height)
        return sum(self._round_width(index) for index in range(rounds))

    def _record(self, blocks: Sequence[AVLNode]) -> None:
        for block in blocks:
            self.placed[block.key] = (block.leaf, block.value)

    @override
    def _read_paths(self, leaves: list[int]) -> None:
        read_together(self._client, [(self, leaves), (self._meta, leaves)])

    @override
    def _fetch_nodes(self, requests: Sequence[tuple[bytes, int]]) -> None:
        super()._fetch_nodes(requests)
        for key, leaf in requests:
            moved = self._meta.take_at(leaf, lambda block_key, key=key: block_key == key)
            if moved:
                self._local.require(key).value = moved[0].value

    @override
    def _return_to_stash(self, blocks: Sequence[AVLNode]) -> None:
        self._record(blocks)
        super()._return_to_stash(blocks)

    @override
    def _build_ods_blocks(self, data: KvList) -> tuple[list[AVLNode], Root[bytes]]:
        blocks, root = super()._build_ods_blocks(data)
        self._record(blocks)
        return blocks, root


class Grove:
    def __init__(self, config: GroveConfig):
        self._config: GroveConfig = config
        self._max_degree: int = config.max_degree
        self._width: int = config.round_width
        self._size: int = 0
        self._leaf: UInt = leaf_field(config.num_data)
        self._tombstone: int = leaf_count(config.num_data)
        level = self._tombstone.bit_length()
        index_batch = config.max_degree + 1

        self._graph: KeyedStore[GroveConfig] = KeyedStore(
            replace(
                config,
                name=f"{config.name}_graph",
                data_size=self._record.size,
                stash_scale=config.stash_scale * self._width,
            ),
            key_size=_ID_SIZE,
        )
        self._position_meta: DupStore = DupStore(
            replace(
                config,
                name=f"{config.name}_position_meta",
                data_size=self._leaf.size,
                bucket_size=config.meta_bucket_size
                or meta_bucket_size(insertions=config.num_rounds * self._width, group=1, level=level),
                stash_scale=config.stash_scale * self._width,
            ),
            key_size=_ID_SIZE,
        )
        self._index: _GroveIndex = _GroveIndex(
            AvlOmapConfig(
                num_data=config.num_data,
                key_size=_ID_SIZE,
                data_size=self._leaf.size,
                client=config.client,
                name=f"{config.name}_index",
                build_file=config.build_file,
                bucket_size=config.bucket_size,
                stash_scale=config.stash_scale,
                encryptor=config.encryptor,
                distinguishable=True,
                batch_size=index_batch,
            ),
            meta=self._position_meta,
        )
        self._index_batch: int = index_batch
        self._index_paths: dict[OdsOp, int] = {op: self._index.op_paths(op) for op in ODS_OPS}
        round_insertions = self._width * self._max_degree + max(self._index_paths.values())
        self._graph_meta: DupStore = DupStore(
            replace(
                config,
                name=f"{config.name}_graph_meta",
                data_size=self._meta_value.size,
                bucket_size=config.meta_bucket_size
                or meta_bucket_size(
                    insertions=config.num_rounds * round_insertions, group=self._max_degree + 1, level=level
                ),
                stash_scale=config.stash_scale * round_insertions,
            ),
            key_size=2 * _ID_SIZE + 1,
        )

    @cached_property
    def _record(self) -> Record[_StoredVertex]:
        neighbor = Pair(UInt(8 * _ID_SIZE), self._leaf)
        return Record(
            _StoredVertex,
            [
                Col(lambda vertex: vertex.value, Bytes(self._config.data_size)),
                Col(lambda vertex: vertex.position, self._leaf),
                Col(lambda vertex: vertex.neighbors, Seq(neighbor, self._max_degree)),
            ],
        )

    @cached_property
    def _meta_value(self) -> UInt:
        return UInt.covering(self._tombstone)

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._config.name!r}"

    @property
    def size(self) -> int:
        return self._size

    def _encode(self, vertex: _Vertex) -> bytes:
        return self._record.encode(_StoredVertex(vertex.value, vertex.position, list(vertex.neighbors.items())))

    def _decode(self, data: bytes) -> _Vertex:
        stored = self._record.decode(data)
        return _Vertex(value=stored.value, position=stored.position, neighbors=dict(stored.neighbors))

    def _require_vertex(self, vertex: object) -> None:
        require_vertex(self._identity, vertex)

    def _require_value(self, value: object) -> None:
        require_value(self._identity, value, self._config.data_size)

    def init_server_storage(
        self, graph: Mapping[int, Sequence[int]] | None = None, values: Mapping[int, bytes] | None = None
    ) -> None:
        graph = graph or {}
        values = values or {}
        require_graph(
            self._identity,
            graph,
            values,
            capacity=self._config.num_data,
            max_degree=self._max_degree,
            data_size=self._config.data_size,
        )

        leaves = {vertex: self._graph.new_leaf() for vertex in graph}
        self._index.placed = {}
        self._index.init_server_storage([(_key(v), self._leaf.encode(leaf)) for v, leaf in leaves.items()])
        self._graph.host(
            [
                KeyedBlock(
                    key=_key(vertex),
                    leaf=leaves[vertex],
                    value=self._encode(
                        _Vertex(
                            value=values.get(vertex, b""),
                            position=self._index.placed[_key(vertex)][0],
                            neighbors={neighbor: leaves[neighbor] for neighbor in neighbors},
                        )
                    ),
                )
                for vertex, neighbors in graph.items()
            ]
        )
        for store in (self._graph_meta, self._position_meta):
            store.host([])
        self._index.placed = {}
        self._size = len(graph)

    def _chunks[T](self, items: Sequence[T]) -> list[Sequence[T]]:
        return [items[start : start + self._width] for start in range(0, len(items), self._width)]

    def _neighbor_dup(self, target: int, leaf: int, source: int, value: int) -> KeyedBlock:
        return KeyedBlock(key=_key(target) + _NEIGHBOR + _key(source), leaf=leaf, value=self._meta_value.encode(value))

    def _position_dup(self, target: int, leaf: int, position: int) -> KeyedBlock:
        return KeyedBlock(key=_key(target) + _POSITION, leaf=leaf, value=self._meta_value.encode(position))

    def _apply_pending(self, vertex_id: int, vertex: _Vertex, leaf: int) -> None:
        prefix = _key(vertex_id)
        for dup in self._graph_meta.take_at(leaf, lambda key: key[:_ID_SIZE] == prefix):
            number = self._meta_value.decode(dup.value)
            if dup.key[_ID_SIZE : _ID_SIZE + 1] == _POSITION:
                vertex.position = number
            elif number == self._tombstone:
                del vertex.neighbors[_vertex(dup.key[_ID_SIZE + 1 :])]
            else:
                source = _vertex(dup.key[_ID_SIZE + 1 :])
                assert source in vertex.neighbors
                vertex.neighbors[source] = number

    def _access(
        self,
        targets: Sequence[_Target],
        placed: _Placed,
        *,
        index_paths: int = 0,
        edit: Callable[[dict[int, _Vertex]], set[int]] | None = None,
    ) -> tuple[dict[int, _Vertex], dict[int, int]]:
        real = {vertex: leaf for vertex, leaf in filter(None, targets)}
        width = len(targets)
        graph_leaves = self._graph.padded_leaves(real.values(), min(width, self._graph.leaf_range))
        graph_meta_leaves = graph_leaves + self._graph_meta.rl_leaves(width * self._max_degree + index_paths)
        position_leaves = self._position_meta.rl_leaves(width)
        read_together(
            self._config.client,
            [
                (self._graph, graph_leaves),
                (self._graph_meta, graph_meta_leaves),
                (self._position_meta, position_leaves),
            ],
        )

        blocks = {
            _vertex(block.key): block
            for block in self._graph.take(_key(vertex) for vertex in real).values()
            if block.leaf == real[_vertex(block.key)]
        }
        vertices: dict[int, _Vertex] = {}
        for vertex_id, block in blocks.items():
            vertices[vertex_id] = self._decode(block.value)
            self._apply_pending(vertex_id, vertices[vertex_id], real[vertex_id])

        removed: set[int] = edit(vertices) if edit is not None else set()
        created = {vertex_id for vertex_id in vertices if vertex_id not in blocks}
        for vertex_id in created:
            blocks[vertex_id] = KeyedBlock(key=_key(vertex_id), leaf=0, value=b"")
        self._graph.add(blocks[vertex_id] for vertex_id in created)

        new_leaves = {vertex_id: self._graph.new_leaf() for vertex_id in vertices if vertex_id not in removed}
        graph_dups: list[KeyedBlock] = []
        position_dups: list[KeyedBlock] = []
        for vertex_id, vertex in vertices.items():
            announced = self._tombstone if vertex_id in removed else new_leaves[vertex_id]
            for neighbor, neighbor_leaf in vertex.neighbors.items():
                if neighbor in vertices:
                    assert vertex_id not in removed and vertex_id in vertices[neighbor].neighbors
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
                    KeyedBlock(
                        key=_key(vertex_id), leaf=vertex.position, value=self._leaf.encode(new_leaves[vertex_id])
                    )
                )
        for key, (position, graph_leaf) in placed.items():
            if _vertex(key) not in vertices:
                graph_dups.append(self._position_dup(_vertex(key), self._leaf.decode(graph_leaf), position))

        self._graph.drop(_key(vertex_id) for vertex_id in removed)
        for vertex_id, vertex in vertices.items():
            if vertex_id not in removed:
                blocks[vertex_id].leaf = new_leaves[vertex_id]
                blocks[vertex_id].value = self._encode(vertex)
        self._graph_meta.add(graph_dups)
        self._position_meta.add(position_dups)
        for store in (self._graph, self._graph_meta, self._position_meta):
            store.write_back()
        return vertices, new_leaves

    def _deliver_index_moves(self, placed: _Placed, op: OdsOp) -> None:
        dups = [
            self._position_dup(_vertex(key), self._leaf.decode(graph_leaf), position)
            for key, (position, graph_leaf) in placed.items()
        ]
        self._graph_meta.read_rl(self._index_paths[op])
        self._graph_meta.add(dups)
        self._graph_meta.write_back()

    def _locate(self, vertices: Sequence[int | None]) -> tuple[list[_Target], _Placed]:
        self._index.placed = {}
        keys = [_key(vertex) for vertex in vertices if vertex is not None]
        found = dict(zip(keys, self._index.search_batch([(key, None) for key in keys]), strict=True))
        placed, self._index.placed = self._index.placed, {}
        targets: list[_Target] = []
        for vertex in vertices:
            leaf = None if vertex is None else found[_key(vertex)]
            targets.append(None if vertex is None or leaf is None else (vertex, self._leaf.decode(leaf)))
        return targets, placed

    def _lookup_chunk(self, vertices: Sequence[int | None]) -> tuple[dict[int, _Vertex], dict[int, int]]:
        targets, placed = self._locate(vertices)
        seen: set[int] = set()
        unique: list[_Target] = []
        for target in targets:
            unique.append(target if target is not None and target[0] not in seen else None)
            if target is not None:
                seen.add(target[0])
        return self._access(unique, placed, index_paths=self._index_paths["search"])

    @staticmethod
    def _gather(found: dict[int, _Vertex], layer: Mapping[int, _Vertex], moved: Mapping[int, int]) -> None:
        for vertex in found.values():
            for neighbor in vertex.neighbors:
                if neighbor in moved:
                    vertex.neighbors[neighbor] = moved[neighbor]
        found.update(layer)

    def _lookup_all(self, vertices: Sequence[int]) -> dict[int, _Vertex]:
        found: dict[int, _Vertex] = {}
        for start in range(0, len(vertices), self._index_batch):
            self._gather(found, *self._lookup_chunk(vertices[start : start + self._index_batch]))
        return found

    def _visit(self, targets: Sequence[_Target]) -> dict[int, _Vertex]:
        found: dict[int, _Vertex] = {}
        for chunk in self._chunks(targets):
            self._gather(found, *self._access(chunk, {}))
        return found

    def _layer(self, frontier: Mapping[int, _Vertex], width: int, skip: set[int]) -> list[_Target]:
        fresh: dict[int, int] = {}
        for vertex in frontier.values():
            for neighbor, leaf in vertex.neighbors.items():
                if neighbor not in skip:
                    _ = fresh.setdefault(neighbor, leaf)
        width = min(width, self._size)
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
        width = len(vertices) * self._max_degree
        slots = self._layer({v: centers[v] for v in dict.fromkeys(vertices) if v in centers}, width, set())
        return {vertex_id: vertex.value for vertex_id, vertex in self._visit(slots).items()}

    def hop_neighborhood(self, vertex: int, hops: int) -> dict[int, bytes]:
        self._require_vertex(vertex)
        found = self._lookup_all([vertex])
        frontier = dict(found)
        reach = 1
        for _ in range(hops):
            reach *= self._max_degree
            layer = self._visit(self._layer(frontier, reach, set(found)))
            found.update(layer)
            frontier = layer
        return {vertex_id: v.value for vertex_id, v in found.items()}

    def random_walk(self, vertex: int, steps: int) -> dict[int, bytes]:
        self._require_vertex(vertex)
        found = self._lookup_all([vertex])
        current = dict(found)
        for _ in range(steps):
            layer = self._visit(self._layer(current, self._max_degree, set()))
            found.update(layer)
            step = secrets.choice(list(layer)) if layer else None
            current = {step: layer[step]} if step is not None else {}
        return {vertex_id: v.value for vertex_id, v in found.items()}

    def _traverse(
        self, source: int, next_vertex: Callable[[], _Target], expand: Callable[[int, _Vertex], None]
    ) -> dict[int, bytes]:
        self._require_vertex(source)
        order: dict[int, bytes] = {}
        found = self._lookup_all([source])
        for step in range(self._size):
            if step:
                found = self._access([next_vertex()], {})[0]
            for vertex_id, vertex in found.items():
                order[vertex_id] = vertex.value
                expand(vertex_id, vertex)
        return order

    def bfs(self, source: int) -> dict[int, bytes]:
        queue: deque[tuple[int, int]] = deque()
        discovered = {source}

        def next_vertex() -> _Target:
            return queue.popleft() if queue else None

        def expand(_: int, vertex: _Vertex) -> None:
            for neighbor, leaf in vertex.neighbors.items():
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    queue.append((neighbor, leaf))

        return self._traverse(source, next_vertex, expand)

    def dfs(self, source: int) -> dict[int, bytes]:
        visited: set[int] = set()
        stack: list[Iterator[tuple[int, int]]] = []

        def next_vertex() -> _Target:
            while stack:
                for neighbor, leaf in stack[-1]:
                    if neighbor not in visited:
                        return neighbor, leaf
                del stack[-1]
            return None

        def expand(vertex_id: int, vertex: _Vertex) -> None:
            visited.add(vertex_id)
            stack.append(iter(list(vertex.neighbors.items())))

        return self._traverse(source, next_vertex, expand)

    def _finish_index_op[T](self, op: OdsOp, run: Callable[[], T]) -> T:
        self._index.placed = {}
        result = run()
        placed, self._index.placed = self._index.placed, {}
        self._deliver_index_moves(placed, op)
        return result

    def insert(self, vertex: int, value: bytes, neighbors: Sequence[int]) -> None:
        self._require_vertex(vertex)
        self._require_value(value)
        require_neighbors(self._identity, vertex, neighbors, self._max_degree)
        require_room(self._identity, held=self._size, fresh=1, capacity=self._config.num_data)

        padding: list[int | None] = [None] * (self._max_degree - len(neighbors))
        targets, placed = self._locate([vertex, *neighbors, *padding])
        problem = None
        if targets[0] is not None:
            problem = f"vertex {vertex} already exists"
        elif any(target is None for target in targets[1 : 1 + len(neighbors)]):
            problem = "a neighbor does not exist"

        def link(found: dict[int, _Vertex]) -> set[int]:
            nonlocal problem
            if problem is None and any(len(found[n].neighbors) >= self._max_degree for n in neighbors):
                problem = f"a neighbor already has max_degree {self._max_degree} neighbors"
            if problem is None:
                found[vertex] = _Vertex(value=value, position=0, neighbors=dict.fromkeys(neighbors, 0))
                for neighbor in neighbors:
                    found[neighbor].neighbors[vertex] = 0
            return set()

        _, new_leaves = self._access([None, *targets[1:]], placed, index_paths=self._index_paths["search"], edit=link)
        if problem is not None:
            raise ValueError(f"{self._identity}: {problem}.")
        self._finish_index_op("insert", lambda: self._index.insert(_key(vertex), self._leaf.encode(new_leaves[vertex])))
        self._size += 1

    def delete(self, vertex: int) -> bytes | None:
        self._require_vertex(vertex)
        targets, placed = self._locate([vertex])
        found, _ = self._access(targets, placed, index_paths=self._index_paths["search"], edit=lambda found: set(found))
        if targets[0] is None:
            return None
        _ = self._finish_index_op("delete", lambda: self._index.delete(_key(vertex)))
        self._size -= 1
        return found[vertex].value

    def update(self, vertex: int, value: bytes) -> bytes | None:
        self._require_vertex(vertex)
        self._require_value(value)
        targets, placed = self._locate([vertex])
        old: list[bytes] = []

        def change(found: dict[int, _Vertex]) -> set[int]:
            if vertex in found:
                old.append(found[vertex].value)
                found[vertex].value = value
            return set()

        _ = self._access(targets, placed, index_paths=self._index_paths["search"], edit=change)
        return old[0] if old else None

    def _edit_edge(self, u: int, v: int, *, link: bool) -> None:
        self._require_vertex(u)
        self._require_vertex(v)
        if u == v:
            raise ContractError(f"{self._identity}: an edge needs two distinct vertices, got {u} twice.")
        targets, placed = self._locate([u, v])
        problem = None

        def change(found: dict[int, _Vertex]) -> set[int]:
            nonlocal problem
            if u not in found or v not in found:
                problem = "an endpoint does not exist"
            elif link and v in found[u].neighbors:
                problem = f"vertices {u} and {v} are already adjacent"
            elif link and max(len(found[u].neighbors), len(found[v].neighbors)) >= self._max_degree:
                problem = f"an endpoint already has max_degree {self._max_degree} neighbors"
            elif not link and v not in found[u].neighbors:
                problem = f"vertices {u} and {v} are not adjacent"
            elif link:
                found[u].neighbors[v] = 0
                found[v].neighbors[u] = 0
            else:
                del found[u].neighbors[v]
                del found[v].neighbors[u]
            return set()

        _ = self._access(targets, placed, index_paths=self._index_paths["search"], edit=change)
        if problem is not None:
            raise ValueError(f"{self._identity}: {problem}.")

    def add_edge(self, u: int, v: int) -> None:
        self._edit_edge(u, v, link=True)

    def remove_edge(self, u: int, v: int) -> None:
        self._edit_edge(u, v, link=False)
