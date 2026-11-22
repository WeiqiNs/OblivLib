import random
import secrets
from typing import Any

import pytest

from oblivlib.dependency import AesGcm, Client, ContractError, GraphOsConfig
from oblivlib.graph import GraphOS
from oblivlib.graph.graphos import _edge_key, _slot_key, _vertex_key

MAX_DEG = 3


def _make(client, num_vertices=24, max_degree=MAX_DEG, encryptor=None):
    return GraphOS(
        GraphOsConfig(num_data=num_vertices, max_degree=max_degree, data_size=8, client=client, encryptor=encryptor)
    )


def _random_graph(rng, num_vertices, max_deg):
    graph = {vertex: [] for vertex in range(num_vertices)}
    for _ in range(num_vertices * max_deg):
        a, b = rng.sample(range(num_vertices), 2)
        if b not in graph[a] and len(graph[a]) < max_deg and len(graph[b]) < max_deg:
            graph[a].append(b)
            graph[b].append(a)
    return graph


def _within(graph, start, hops):
    if start not in graph:
        return set()
    seen, frontier = {start}, {start}
    for _ in range(hops):
        frontier = {n for v in frontier for n in graph[v]} - seen
        seen |= frontier
    return seen


def _assert_matches_model(graphos, graph, values, read_all_blocks, duplicate_blocks):
    assert duplicate_blocks(graphos) == {}
    store = graphos._value_tree
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

    def read(graphos):
        store = graphos._value_tree
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
    def test_operations_match_a_graph_model(self, client, enc, matches_model, stored_adjacency):
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
                assert graphos.neighbor_values(queried) == {n: values[n] for n in expected}
            elif roll < 0.5:
                hops = rng.randrange(3)
                assert graphos.hop_neighborhood(vertex, hops) == {v: values[v] for v in _within(graph, vertex, hops)}
            elif roll < 0.6:
                walk = graphos.random_walk(vertex, 3)
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
                assert graphos.delete(vertex) == values[vertex]
                for neighbor in graph.pop(vertex):
                    graph[neighbor].remove(vertex)
                del values[vertex]
            assert graphos.size == len(graph)

        matches_model(graphos, graph, values)

    def test_hop_layers_stop_growing_at_the_vertex_count(self, client):
        graph = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1], 4: []}
        graphos = _make(client, 8)
        graphos.init_server_storage(graph, {v: bytes([v]) for v in graph})
        rounds = client.metrics.rounds
        assert graphos.hop_neighborhood(0, 40) == {v: bytes([v]) for v in [0, 1, 2, 3]}
        assert client.metrics.rounds - rounds < 40 * 4 * (graphos._omap._max_height + 2)

    def test_lookup_serves_repeats_and_more_vertices_than_one_batch(self, client):
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
        assert graphos.neighbor_values([0]) == {2: b"v", 3: b"v"}

    def test_empty_graph_grows_from_inserts(self, client, matches_model):
        graphos = _make(client, 4)
        graphos.init_server_storage()
        graphos.insert(5, b"a", [])
        graphos.insert(6, b"b", [5])
        assert graphos.neighbor_values([5]) == {6: b"b"}
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
        with pytest.raises(ValueError, match="max_degree"):
            graphos.insert(4, b"x", [0])
        assert graphos.delete(7) is None
        assert graphos.neighbor_values([0]) == {1: b"v", 2: b"v", 3: b"v"}
        matches_model(graphos, graph, values)

    @pytest.mark.parametrize(
        "graph, values",
        [
            ({0: [1]}, {}),
            ({0: [0]}, {}),
            ({0: [1, 1], 1: [0, 0]}, {}),
            ({0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}, {}),
            ({0: []}, {0: b"123456789"}),
            ({-1: []}, {}),
            ({vertex: [] for vertex in range(9)}, {}),
        ],
    )
    def test_init_rejects_each_malformed_graph(self, client, graph, values):
        graphos = _make(client, 8)
        with pytest.raises(ContractError):
            graphos.init_server_storage(graph, values)

    def test_contract_violations_raise_before_any_round(self, client):
        graphos = _make(client, 2, max_degree=1)
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
            with pytest.raises(ContractError, match="a key must be an int"):
                traversal(-1)
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


def _value_tree_reads(client, graphos, run, *args):
    reads = []
    original = client.add_read_path

    def record(label, leaves):
        leaves = list(leaves)
        if label == graphos._value_tree.name:
            reads.append(leaves)
        original(label, leaves)

    client.add_read_path = record
    run(*args)
    client.add_read_path = original
    return reads


class TestGraphOSObliviousness:
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
            lambda g, v: g.neighbor_values([v]),
            lambda g, v: g.hop_neighborhood(v, 2),
            lambda g, v: g.random_walk(v, 2),
            lambda g, v: g.bfs(v),
            lambda g, v: g.dfs(v),
        ],
        ids=["lookup", "neighbor_values", "hop_neighborhood", "random_walk", "bfs", "dfs"],
    )
    def test_query_trace_is_degree_independent(self, query):
        traces = []
        for vertex in [0, 4, 5, 7]:
            client, graphos = self._graphos()
            traces.append(_request_trace(client, query, graphos, vertex))
        assert all(trace == traces[0] for trace in traces)

    @staticmethod
    def _tolerant(run, *args):
        try:
            run(*args)
        except ValueError:
            pass

    def test_delete_trace_is_degree_and_existence_independent(self):
        traces = []
        for vertex in [0, 4, 5, 7]:
            client, graphos = self._graphos()
            traces.append(_request_trace(client, self._tolerant, graphos.delete, vertex))
        assert all(trace == traces[0] for trace in traces)

    def test_insert_trace_is_degree_and_outcome_independent(self):
        traces = []
        for vertex, neighbors in [(9, []), (9, [5]), (9, [2, 3, 5]), (1, []), (9, [8]), (9, [0])]:
            client, graphos = self._graphos()
            traces.append(_request_trace(client, self._tolerant, graphos.insert, vertex, b"v", neighbors))
        assert all(trace == traces[0] for trace in traces)

    @pytest.mark.parametrize(
        "op",
        [
            lambda g: g.neighbor_values([7]),
            lambda g: g.insert(9, b"v", []),
            lambda g: g.delete(5),
        ],
        ids=["fetch", "add", "remove"],
    )
    def test_an_all_padding_read_draws_fresh_leaves(self, monkeypatch, op):
        monkeypatch.setattr(secrets, "randbelow", random.Random(5).randrange)
        last_reads = []
        for _ in range(20):
            client, graphos = self._graphos(num_vertices=64)
            last_reads.append(_value_tree_reads(client, graphos, op, graphos)[-1][0])
        assert len(set(last_reads)) > len(last_reads) // 2
