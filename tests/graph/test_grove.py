"""Tests for Grove: index-free adjacency over four trees, kept consistent by delayed duplications."""

import random
import secrets
from typing import Any

import pytest

from oblivlib.dependency import AesGcm, Client, ContractError, GroveConfig
from oblivlib.graph import Grove
from oblivlib.graph.grove import ID_SIZE

MAX_DEG = 3


def _make(client, num_vertices=24, max_deg=MAX_DEG, encryptor=None, **kwargs):
    return Grove(
        GroveConfig(
            num_vertices=num_vertices,
            max_deg=max_deg,
            data_size=8,
            client=client,
            encryptor=encryptor,
            num_rounds=1 << 12,
            **kwargs,
        )
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


def _vertex_id(key):
    return int.from_bytes(key[:ID_SIZE], "big")


def _newest(store):
    """The newest copy of every key a store holds: its stash first, then tree copies shallowest first."""
    client = store._client
    client.add_read_path(store._name, range(store._leaf_range))
    path = store._cipher.open_path(client.execute().require(store._name))
    newest = {}
    for block in [*store._stash, *(block for index in sorted(path) for block in path[index])]:
        newest.setdefault(block.key, block)
    return newest


@pytest.fixture
def consistent(read_all_blocks):
    def check(grove, graph, values):
        graph_blocks = {_vertex_id(b.key): b for b in read_all_blocks(grove._graph) + grove._graph._stash}
        assert set(graph_blocks) == set(graph)
        index_blocks = {
            _vertex_id(b.key): b
            for b in read_all_blocks(grove._index) + grove._index._stash + grove._index._local.to_list()
        }
        assert set(index_blocks) == set(graph)
        graph_dups = _newest(grove._graph_meta)
        index_dups = _newest(grove._position_meta)

        for dup in graph_dups.values():
            assert _vertex_id(dup.key) in graph, f"an update is aimed at missing vertex {_vertex_id(dup.key)}"
            assert dup.leaf == graph_blocks[_vertex_id(dup.key)].leaf, "an update is aimed at a stale leaf"
        for dup in index_dups.values():
            assert dup.leaf == index_blocks[_vertex_id(dup.key)].leaf, "an index update is aimed at a stale leaf"

        for vertex, block in graph_blocks.items():
            record = grove._decode(block.value)
            for key, dup in graph_dups.items():
                if _vertex_id(key) != vertex:
                    continue
                number = grove._meta_value.decode(dup.value)
                if key[ID_SIZE : ID_SIZE + 1] == b"P":
                    record.position = number
                elif number == grove._tombstone:
                    record.neighbors.pop(_vertex_id(key[ID_SIZE + 1 :]), None)
                elif _vertex_id(key[ID_SIZE + 1 :]) in record.neighbors:
                    record.neighbors[_vertex_id(key[ID_SIZE + 1 :])] = number
            assert record.value == values[vertex]
            assert sorted(record.neighbors) == sorted(graph[vertex])
            assert all(leaf == graph_blocks[n].leaf for n, leaf in record.neighbors.items())
            assert record.position == index_blocks[vertex].leaf
            node_leaf = index_blocks[vertex].value.value
            if vertex.to_bytes(ID_SIZE, "big") in index_dups:
                node_leaf = index_dups[vertex.to_bytes(ID_SIZE, "big")].value
            assert grove._leaf.decode(node_leaf) == block.leaf

    return check


class TestGrove:
    @pytest.mark.parametrize("enc", [False, True])
    def test_oracle(self, client, enc, consistent, reference_orders):
        rng = random.Random(21 + enc)
        num_vertices = 24
        graph = _random_graph(rng, num_vertices - 6, MAX_DEG)
        values = {vertex: rng.randbytes(8) for vertex in graph}
        grove = _make(client, num_vertices, encryptor=AesGcm() if enc else None)
        grove.init_server_storage(graph, values)
        consistent(grove, graph, values)

        for step in range(100):
            roll = rng.random()
            vertex = rng.randrange(num_vertices + 2)
            if roll < 0.15:
                queried = rng.sample(range(num_vertices + 2), 3)
                assert grove.lookup(queried) == {v: values[v] for v in queried if v in graph}
            elif roll < 0.3:
                queried = rng.sample(range(num_vertices + 2), 2)
                expected = {n for v in queried if v in graph for n in graph[v]}
                assert grove.neighbor_values(queried) == {n: values[n] for n in expected}
            elif roll < 0.42:
                hops = rng.randrange(4)
                assert grove.hop_neighborhood(vertex, hops) == {v: values[v] for v in _within(graph, vertex, hops)}
            elif roll < 0.5:
                walk = grove.random_walk(vertex, 3)
                if vertex in graph:
                    assert set(graph[vertex]) | {vertex} <= set(walk)
                assert set(walk) <= _within(graph, vertex, 3)
                assert all(walk[v] == values[v] for v in walk)
            elif roll < 0.58:
                adjacency = grove.neighbor_ids(list(graph))
                assert {v: sorted(n) for v, n in adjacency.items()} == {v: sorted(n) for v, n in graph.items()}
                bfs_order, dfs_order = reference_orders
                assert list(grove.bfs(vertex).items()) == [(v, values[v]) for v in bfs_order(adjacency, vertex)]
                assert list(grove.dfs(vertex).items()) == [(v, values[v]) for v in dfs_order(adjacency, vertex)]
            elif roll < 0.63 and graph:
                target = rng.choice(sorted(graph))
                value = rng.randbytes(rng.randrange(9))
                grove.update(target, value)
                values[target] = value
            elif roll < 0.68:
                u, v = rng.sample(sorted(graph), 2) if len(graph) > 1 else (0, 1)
                if u in graph and v in graph and v not in graph[u] and max(len(graph[u]), len(graph[v])) < MAX_DEG:
                    grove.add_edge(u, v)
                    graph[u].append(v)
                    graph[v].append(u)
            elif roll < 0.73:
                edges = [(u, v) for u in graph for v in graph[u] if u < v]
                if edges:
                    u, v = rng.choice(edges)
                    grove.remove_edge(u, v)
                    graph[u].remove(v)
                    graph[v].remove(u)
            elif roll < 0.85 and vertex not in graph and len(graph) < num_vertices:
                open_slots = [v for v in graph if len(graph[v]) < MAX_DEG]
                neighbors = rng.sample(open_slots, min(len(open_slots), rng.randrange(MAX_DEG + 1)))
                value = rng.randbytes(rng.randrange(9))
                grove.insert(vertex, value, neighbors)
                graph[vertex] = list(neighbors)
                for neighbor in neighbors:
                    graph[neighbor].append(vertex)
                values[vertex] = value
            elif vertex in graph:
                grove.delete(vertex)
                for neighbor in graph.pop(vertex):
                    graph[neighbor].remove(vertex)
                del values[vertex]
            assert grove.num_vertices == len(graph)
            if step % 10 == 9:
                consistent(grove, graph, values)
        consistent(grove, graph, values)

    def test_neighbor_values_over_more_centers_than_one_index_batch(self, client, consistent):
        graph = {0: [4], 1: [5], 2: [], 3: [6], 4: [0], 5: [1], 6: [3]}
        values = {v: bytes([v]) for v in graph}
        grove = _make(client, 16)
        grove.init_server_storage(graph, values)
        for _ in range(10):
            assert grove.neighbor_values([0, 1, 2, 3, 4, 5, 6]) == {v: bytes([v]) for v in [0, 1, 3, 4, 5, 6]}
        consistent(grove, graph, values)

    def test_repeated_queries_keep_every_pointer_current(self, client, consistent):
        graph = {0: [1, 2, 3], 1: [0, 2], 2: [0, 1], 3: [0]}
        values = {v: bytes([v]) for v in graph}
        grove = _make(client, 8)
        grove.init_server_storage(graph, values)
        for _ in range(40):
            assert grove.neighbor_values([0]) == {1: b"\x01", 2: b"\x02", 3: b"\x03"}
            assert grove.hop_neighborhood(3, 2) == values
        consistent(grove, graph, values)

    def test_traversals_visit_the_component_in_order(self, client, consistent):
        graph = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2, 4], 4: [3], 5: [6], 6: [5]}
        values = {v: bytes([v]) for v in graph}
        grove = _make(client, 10)
        grove.init_server_storage(graph, values)
        assert list(grove.bfs(0)) == [0, 1, 2, 3, 4]
        assert list(grove.dfs(0)) == [0, 1, 3, 2, 4]
        assert list(grove.dfs(5)) == [5, 6]
        assert grove.bfs(9) == {}
        assert grove.bfs(0)[4] == bytes([4])
        consistent(grove, graph, values)

    def test_a_deleted_vertex_leaves_the_index_and_its_neighbors(self, client, consistent):
        graph = {0: [1, 2], 1: [0], 2: [0]}
        grove = _make(client, 8)
        grove.init_server_storage(graph, {v: b"v" for v in graph})
        grove.delete(0)
        assert grove.lookup([0]) == {}
        assert grove.neighbor_ids([1, 2]) == {1: [], 2: []}
        consistent(grove, {1: [], 2: []}, {1: b"v", 2: b"v"})

    def test_failed_updates_leave_the_graph_usable_and_unchanged(self, client, consistent):
        graph = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
        values = dict.fromkeys(graph, b"v")
        grove = _make(client, 8)
        grove.init_server_storage(graph, values)
        with pytest.raises(ValueError, match="already exists"):
            grove.insert(1, b"x", [])
        with pytest.raises(ValueError, match="does not exist"):
            grove.insert(4, b"x", [7])
        with pytest.raises(ValueError, match="max_deg"):
            grove.insert(4, b"x", [0])
        with pytest.raises(KeyError, match="vertex 7 does not exist"):
            grove.delete(7)
        assert grove.neighbor_values([0]) == {1: b"v", 2: b"v", 3: b"v"}
        consistent(grove, graph, values)

    def test_failed_edge_edits_and_updates_leave_the_graph_unchanged(self, client, consistent):
        graph = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0], 4: []}
        values = dict.fromkeys(graph, b"v")
        grove = _make(client, 8)
        grove.init_server_storage(graph, values)
        with pytest.raises(ValueError, match="already adjacent"):
            grove.add_edge(0, 1)
        with pytest.raises(ValueError, match="max_deg"):
            grove.add_edge(0, 4)
        with pytest.raises(ValueError, match="does not exist"):
            grove.add_edge(4, 7)
        with pytest.raises(ValueError, match="not adjacent"):
            grove.remove_edge(1, 4)
        with pytest.raises(KeyError, match="vertex 7 does not exist"):
            grove.update(7, b"x")
        with pytest.raises(ContractError):
            grove.add_edge(4, 4)
        consistent(grove, graph, values)

    def test_edge_edits_rewire_both_endpoints(self, client, consistent):
        graph = {0: [1], 1: [0], 2: []}
        grove = _make(client, 8)
        grove.init_server_storage(graph, dict.fromkeys(graph, b"v"))
        grove.add_edge(1, 2)
        grove.remove_edge(0, 1)
        grove.update(2, b"w")
        assert grove.neighbor_ids([0, 1, 2]) == {0: [], 1: [2], 2: [1]}
        assert grove.neighbor_values([1]) == {2: b"w"}
        consistent(grove, {0: [], 1: [2], 2: [1]}, {0: b"v", 1: b"v", 2: b"w"})

    def test_contract_violations_are_raised_before_any_io(self, client):
        grove = _make(client, 8)
        grove.init_server_storage({0: []})
        rounds = client.metrics.rounds
        cases: list[tuple[Any, Any, Any]] = [("a", b"", []), (1, b"123456789", []), (1, b"", [1]), (1, b"", [0, 0])]
        for vertex, value, neighbors in cases:
            with pytest.raises(ContractError):
                grove.insert(vertex, value, neighbors)
        hops: list[Any] = [-1, 1.5]
        for count in hops:
            with pytest.raises(ContractError, match="hops"):
                grove.hop_neighborhood(0, count)
            with pytest.raises(ContractError, match="steps"):
                grove.random_walk(0, count)
        assert client.metrics.rounds == rounds


def _request_trace(client, run, *args):
    trace = []
    original = client.add_read_path

    def record(label, leaves):
        leaves = list(leaves)
        trace.append((label.split("_", 1)[1], len(leaves)))
        original(label, leaves)

    client.add_read_path = record
    run(*args)
    client.add_read_path = original
    return trace


class TestGroveObliviousness:
    """A query's sequence of (tree, paths requested) depends only on its type, never on degrees or existence."""

    GRAPH = {0: [1, 2, 3], 1: [0, 4], 2: [0], 3: [0], 4: [1], 5: []}

    def _grove(self, monkeypatch, seed):
        monkeypatch.setattr(secrets, "randbelow", random.Random(seed).randrange)
        client = Client.local()
        grove = _make(client, 16)
        grove.init_server_storage(self.GRAPH, dict.fromkeys(self.GRAPH, b"v"))
        return client, grove

    def _same(self, traces):
        assert all(trace == traces[0] for trace in traces)

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
    def test_query_trace_is_degree_independent(self, monkeypatch, query):
        traces = []
        for vertex in [0, 4, 5, 9]:
            client, grove = self._grove(monkeypatch, 3)
            traces.append(_request_trace(client, query, grove, vertex))
        self._same(traces)

    @pytest.mark.parametrize(
        "query",
        [
            lambda g: g.neighbor_values([0, 1, 2, 3, 4, 5]),
            lambda g: g.lookup([0, 1, 2, 3, 4, 5, 0]),
            lambda g: g.hop_neighborhood(1, 3),
            lambda g: g.random_walk(0, 4),
            lambda g: g.bfs(0),
            lambda g: g.dfs(4),
        ],
        ids=["neighbor_values", "lookup", "hop_neighborhood", "random_walk", "bfs", "dfs"],
    )
    def test_every_target_is_found_where_it_is_sought(self, monkeypatch, query):
        """No round seeks a vertex at a path it has left; a stale path would fail to find it and re-read a revealed leaf."""
        for seed in range(10):
            _, grove = self._grove(monkeypatch, seed)
            original = grove._access

            def access(targets, placed, original=original, seed=seed, **kwargs):
                vertices, moved = original(targets, placed, **kwargs)
                sought = {vertex for vertex, _ in filter(None, targets)}
                assert sought <= set(vertices), f"seed {seed}: a vertex was sought at a stale path"
                return vertices, moved

            monkeypatch.setattr(grove, "_access", access)
            query(grove)

    @pytest.mark.parametrize(
        "cases",
        [
            [
                lambda g: g.add_edge(2, 3),
                lambda g: g.add_edge(0, 5),
                lambda g: g.add_edge(0, 1),
                lambda g: g.add_edge(5, 9),
            ],
            [lambda g: g.remove_edge(0, 1), lambda g: g.remove_edge(1, 4), lambda g: g.remove_edge(2, 5)],
            [lambda g: g.update(0, b"x"), lambda g: g.update(5, b"x"), lambda g: g.update(9, b"x")],
        ],
        ids=["add_edge", "remove_edge", "update"],
    )
    def test_edits_have_one_shape_whether_they_succeed_or_fail(self, monkeypatch, cases):
        traces = []
        for case in cases:
            client, grove = self._grove(monkeypatch, 8)

            def run(case=case, grove=grove):
                try:
                    case(grove)
                except (KeyError, ValueError):
                    pass

            traces.append(_request_trace(client, run))
        self._same(traces)

    def test_every_round_reads_its_full_rl_budget(self, monkeypatch):
        client, grove = self._grove(monkeypatch, 6)
        reads = [count for tree, count in _request_trace(client, grove.neighbor_values, [0]) if tree == "graph_meta"]
        k = MAX_DEG
        assert reads == [1 + k + grove._index_paths["search"], k + k * k]

    def test_insert_trace_is_degree_independent(self, monkeypatch):
        traces = []
        for neighbors in [[], [5], [2, 3, 5]]:
            client, grove = self._grove(monkeypatch, 4)
            traces.append(_request_trace(client, grove.insert, 9, b"v", neighbors))
        self._same(traces)

    def test_delete_trace_is_degree_independent(self, monkeypatch):
        traces = []
        for vertex in [0, 4, 5, 9]:
            client, grove = self._grove(monkeypatch, 5)

            def delete(vertex=vertex, grove=grove):
                try:
                    grove.delete(vertex)
                except KeyError:
                    pass

            traces.append(_request_trace(client, delete))
        self._same(traces)
