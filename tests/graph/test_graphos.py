"""Tests for GraphOS: an undirected graph whose entries live in a keyed value tree, located through an AVL OMAP."""

import random
import secrets
from typing import Any

import pytest

from oblivlib.dependency import AesGcm, Client, ContractError, GraphOsConfig
from oblivlib.graph import GraphOS
from oblivlib.graph.graphos import _edge_key, _slot_key, _vertex_key

MAX_DEG = 3


def _make(client, num_vertices=24, max_deg=MAX_DEG, encryptor=None):
    return GraphOS(
        GraphOsConfig(num_vertices=num_vertices, max_deg=max_deg, data_size=8, client=client, encryptor=encryptor)
    )


def _random_graph(rng, num_vertices, max_deg):
    graph = {vertex: [] for vertex in range(num_vertices)}
    for _ in range(num_vertices * max_deg):
        a, b = rng.sample(range(num_vertices), 2)
        if b not in graph[a] and len(graph[a]) < max_deg and len(graph[b]) < max_deg:
            graph[a].append(b)
            graph[b].append(a)
    return graph


def _within(graph, start, num_hop):
    if start not in graph:
        return set()
    seen, frontier = {start}, {start}
    for _ in range(num_hop):
        frontier = {n for v in frontier for n in graph[v]} - seen
        seen |= frontier
    return seen


def _assert_matches_model(graphos, graph, values, read_all_blocks, duplicate_blocks):
    """Every stored entry is exactly the V/E/P encoding of the model, and its OMAP leaf points at it."""
    assert duplicate_blocks(graphos) == {}
    store = graphos._store
    stored = {block.key: block for block in read_all_blocks(store) + store._stash if block.key is not None}
    expected_keys = set()
    for vertex, neighbors in graph.items():
        value, degree = graphos._decode_vertex(stored[_vertex_key(vertex)].value)
        assert (value, degree) == (values[vertex], len(neighbors))
        expected_keys.add(_vertex_key(vertex))
        listed = []
        for index in range(degree):
            neighbor = graphos._id.decode(stored[_edge_key(vertex, index)].value)
            listed.append(neighbor)
            assert graphos._index.decode(stored[_slot_key(vertex, neighbor)].value) == index
            expected_keys |= {_edge_key(vertex, index), _slot_key(vertex, neighbor)}
        assert sorted(listed) == sorted(neighbors)
    assert set(stored) == expected_keys
    for key, block in stored.items():
        assert graphos._leaf.decode(graphos._omap.search(key)) == block.leaf


def _bfs_order(adjacency, source):
    if source not in adjacency:
        return []
    order, discovered, queue = [], {source}, [source]
    while queue:
        vertex = queue.pop(0)
        order.append(vertex)
        for neighbor in adjacency[vertex]:
            if neighbor not in discovered:
                discovered.add(neighbor)
                queue.append(neighbor)
    return order


def _dfs_order(adjacency, source):
    order = []

    def visit(vertex):
        order.append(vertex)
        for neighbor in adjacency[vertex]:
            if neighbor not in order:
                visit(neighbor)

    if source in adjacency:
        visit(source)
    return order


@pytest.fixture
def stored_adjacency(read_all_blocks):
    """Each vertex's neighbors in the order the E entries hold them, which deletes reorder."""

    def read(graphos):
        store = graphos._store
        stored = {block.key: block for block in read_all_blocks(store) + store._stash if block.key is not None}
        adjacency = {}
        for key, block in stored.items():
            if key[:1] == b"V":
                vertex = int.from_bytes(key[1:])
                _, degree = graphos._decode_vertex(block.value)
                adjacency[vertex] = [graphos._id.decode(stored[_edge_key(vertex, i)].value) for i in range(degree)]
        return adjacency

    return read


@pytest.fixture
def matches_model(read_all_blocks, duplicate_blocks):
    def check(graphos, graph, values):
        _assert_matches_model(graphos, graph, values, read_all_blocks, duplicate_blocks)

    return check


class TestGraphOS:
    @pytest.mark.parametrize("enc", [False, True])
    def test_oracle(self, client, enc, matches_model, stored_adjacency):
        rng = random.Random(11 + enc)
        num_vertices = 24
        graph = _random_graph(rng, num_vertices - 6, MAX_DEG)
        values = {vertex: rng.randbytes(8) for vertex in graph}
        graphos = _make(client, num_vertices, encryptor=AesGcm() if enc else None)
        graphos.init_server_storage(graph, values)

        for _ in range(40):
            roll = rng.random()
            vertex = rng.randrange(num_vertices + 2)
            if roll < 0.15:
                queried = rng.sample(range(num_vertices + 2), 3)
                assert graphos.lookup(queried) == {v: values[v] for v in queried if v in graph}
            elif roll < 0.35:
                queried = rng.sample(range(num_vertices + 2), 2)
                expected = {n for v in queried if v in graph for n in graph[v]}
                assert graphos.neighbor(queried) == {n: values[n] for n in expected}
            elif roll < 0.5:
                hops = rng.randrange(3)
                assert graphos.t_hop(vertex, hops) == {v: values[v] for v in _within(graph, vertex, hops)}
            elif roll < 0.6:
                walk = graphos.t_traversal(vertex, 3)
                if vertex in graph:
                    assert set(graph[vertex]) | {vertex} <= set(walk)
                assert set(walk) <= _within(graph, vertex, 3)
                assert all(walk[v] == values[v] for v in walk)
            elif roll < 0.7:
                adjacency = stored_adjacency(graphos)
                assert {v: sorted(n) for v, n in adjacency.items()} == {v: sorted(n) for v, n in graph.items()}
                assert list(graphos.bfs(vertex).items()) == [(v, values[v]) for v in _bfs_order(adjacency, vertex)]
                assert list(graphos.dfs(vertex).items()) == [(v, values[v]) for v in _dfs_order(adjacency, vertex)]
            elif roll < 0.85 and vertex not in graph and len(graph) < num_vertices:
                open_slots = [v for v in graph if len(graph[v]) < MAX_DEG]
                neighbors = rng.sample(open_slots, min(len(open_slots), rng.randrange(MAX_DEG + 1)))
                value = rng.randbytes(rng.randrange(9))
                graphos.insert(vertex, value, neighbors)
                graph[vertex] = list(neighbors)
                for neighbor in neighbors:
                    graph[neighbor].append(vertex)
                values[vertex] = value
            elif vertex in graph:
                graphos.delete(vertex)
                for neighbor in graph.pop(vertex):
                    graph[neighbor].remove(vertex)
                del values[vertex]
            assert graphos.num_vertices == len(graph)

        matches_model(graphos, graph, values)

    def test_lookup_of_repeated_and_many_vertices(self, client):
        graph = {v: [] for v in range(10)}
        graphos = _make(client, 10)
        graphos.init_server_storage(graph, {v: bytes([v]) for v in graph})
        queried = [1, 1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 0, 3]
        assert graphos.lookup(queried) == {v: bytes([v]) for v in queried}
        assert graphos.lookup(queried) == {v: bytes([v]) for v in queried}

    def test_traversals_visit_the_component_in_order(self, client):
        graph = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2, 4], 4: [3], 5: [6], 6: [5]}
        graphos = _make(client, 10)
        graphos.init_server_storage(graph, {v: bytes([v]) for v in graph})
        assert list(graphos.bfs(0)) == [0, 1, 2, 3, 4]
        assert list(graphos.dfs(0)) == [0, 1, 3, 2, 4]
        assert list(graphos.dfs(5)) == [5, 6]
        assert graphos.bfs(9) == {}
        assert graphos.bfs(0)[4] == bytes([4])

    def test_delete_swaps_tail_edge_into_the_hole(self, client, matches_model):
        graph = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
        graphos = _make(client, 8)
        graphos.init_server_storage(graph, dict.fromkeys(graph, b"v"))
        graphos.delete(1)
        graph = {0: [2, 3], 2: [0], 3: [0]}
        matches_model(graphos, graph, dict.fromkeys(graph, b"v"))
        assert graphos.neighbor([0]) == {2: b"v", 3: b"v"}

    def test_empty_graph_grows_from_inserts(self, client, matches_model):
        graphos = _make(client, 4)
        graphos.init_server_storage()
        graphos.insert(5, b"a", [])
        graphos.insert(6, b"b", [5])
        assert graphos.neighbor([5]) == {6: b"b"}
        graphos.delete(5)
        graphos.delete(6)
        assert graphos.lookup([5, 6]) == {}
        matches_model(graphos, {}, {})

    def test_failed_updates_leave_the_graph_usable_and_unchanged(self, client, matches_model):
        graph = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
        values = dict.fromkeys(graph, b"v")
        graphos = _make(client, 8)
        graphos.init_server_storage(graph, values)
        with pytest.raises(ValueError, match="already exists"):
            graphos.insert(1, b"x", [])
        with pytest.raises(ValueError, match="does not exist"):
            graphos.insert(4, b"x", [7])
        with pytest.raises(ValueError, match="max_deg"):
            graphos.insert(4, b"x", [0])
        with pytest.raises(KeyError, match="vertex 7 does not exist"):
            graphos.delete(7)
        assert graphos.neighbor([0]) == {1: b"v", 2: b"v", 3: b"v"}
        matches_model(graphos, graph, values)

    @pytest.mark.parametrize(
        "graph, values",
        [
            ({0: [1]}, {}),
            ({0: [0]}, {}),
            ({0: [1, 1], 1: [0, 0]}, {}),
            ({0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}, {}),
            ({0: []}, {1: b""}),
            ({0: []}, {0: b"123456789"}),
            ({-1: []}, {}),
        ],
    )
    def test_init_rejects_bad_graph(self, client, graph, values):
        graphos = _make(client, 8)
        with pytest.raises(ContractError):
            graphos.init_server_storage(graph, values)

    def test_updates_and_queries_reject_contract_violations_before_any_io(self, client):
        graphos = _make(client, 2)
        graphos.init_server_storage({0: [], 1: []})
        with pytest.raises(ContractError):
            graphos.insert(2, b"x", [])
        graphos = _make(client, 8)
        graphos.init_server_storage({0: []})
        rounds = client.metrics.rounds
        cases: list[tuple[Any, Any, Any]] = [("a", b"", []), (1, b"123456789", []), (1, b"", [1]), (1, b"", [0, 0])]
        for vertex, value, neighbors in cases:
            with pytest.raises(ContractError):
                graphos.insert(vertex, value, neighbors)
        for traversal in (graphos.bfs, graphos.dfs):
            with pytest.raises(ContractError, match="vertex"):
                traversal(-1)
        hops: list[Any] = [-1, 1.5, "2"]
        for num_hop in hops:
            with pytest.raises(ContractError, match="num_hop"):
                graphos.t_hop(0, num_hop)
            with pytest.raises(ContractError, match="num_hop"):
                graphos.t_traversal(0, num_hop)
        assert client.metrics.rounds == rounds


def _request_trace(client, run, *args):
    trace = []
    original = client.add_read_path

    def record(label, leaves):
        leaves = list(leaves)
        trace.append((label.rsplit("_", 1)[-1], len(leaves)))
        original(label, leaves)

    client.add_read_path = record
    run(*args)
    client.add_read_path = original
    return trace


def _value_leaves_read(client, graphos, run, *args):
    requested = []
    original = client.add_read_path

    def record(label, leaves):
        leaves = list(leaves)
        if label == graphos._store._name:
            requested.extend(leaves)
        original(label, leaves)

    client.add_read_path = record
    run(*args)
    client.add_read_path = original
    return requested


class TestGraphOsObliviousness:
    """A query's sequence of (tree, paths requested) depends only on its type, never on the degrees it meets."""

    GRAPH = {0: [1, 2, 3], 1: [0, 4], 2: [0], 3: [0], 4: [1], 5: []}

    def _graphos(self, num_vertices=10):
        client = Client.local()
        graphos = _make(client, num_vertices)
        graphos.init_server_storage(self.GRAPH, dict.fromkeys(self.GRAPH, b"v"))
        return client, graphos

    @pytest.mark.parametrize(
        "query",
        [
            lambda g, v: g.lookup([v]),
            lambda g, v: g.neighbor([v]),
            lambda g, v: g.t_hop(v, 2),
            lambda g, v: g.t_traversal(v, 2),
            lambda g, v: g.bfs(v),
            lambda g, v: g.dfs(v),
        ],
        ids=["lookup", "neighbor", "t_hop", "t_traversal", "bfs", "dfs"],
    )
    def test_query_trace_is_degree_independent(self, query):
        traces = []
        for vertex in [0, 4, 5, 7]:
            client, graphos = self._graphos()
            traces.append(_request_trace(client, query, graphos, vertex))
        assert all(trace == traces[0] for trace in traces)

    def test_delete_trace_is_degree_independent(self):
        traces = []
        for vertex in [0, 4, 5]:
            client, graphos = self._graphos()
            traces.append(_request_trace(client, graphos.delete, vertex))
        assert traces[0] == traces[1] == traces[2]

    def test_insert_trace_is_degree_independent(self):
        traces = []
        for neighbors in [[], [5], [2, 3, 5]]:
            client, graphos = self._graphos()
            traces.append(_request_trace(client, graphos.insert, 9, b"v", neighbors))
        assert traces[0] == traces[1] == traces[2]

    @pytest.mark.parametrize(
        "update",
        [
            lambda g: g.neighbor([7]),
            lambda g: g.insert(9, b"v", []),
            lambda g: g.delete(5),
        ],
        ids=["fetch", "add", "remove"],
    )
    def test_padding_slots_read_fresh_random_leaves(self, monkeypatch, update):
        monkeypatch.setattr(secrets, "randbelow", random.Random(5).randrange)
        reads = []
        for _ in range(20):
            client, graphos = self._graphos(num_vertices=64)
            reads.append(_value_leaves_read(client, graphos, update, graphos))
        first_reads = [leaves[0] for leaves in reads]
        assert len(set(first_reads)) > len(first_reads) // 2
