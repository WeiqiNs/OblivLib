import random
from collections.abc import Callable
from typing import Any

import pytest

from oblivlib.dependency import (
    AesGcm,
    AvlOmapConfig,
    BPlusOmapConfig,
    ContractError,
    GroupOmapConfig,
    PathOramConfig,
)
from oblivlib.omap import AVLOmap, BPlusOmap, GroupOmap, OramOstOmap
from oblivlib.oram import PathOram


def _b(number: int) -> bytes:
    return b"%d" % number


class TestOmapBehavior:
    def test_edge_sizes(self, omap_edge_size, client):
        spec, num_data = omap_edge_size
        omap = spec.make(client=client, num_data=num_data)
        omap.init_server_storage()
        for i in range(num_data):
            omap.insert(key=_b(i), value=_b(i * 10))
        assert [omap.search(key=_b(i)) for i in range(num_data)] == [_b(i * 10) for i in range(num_data)]
        assert [omap.delete(key=_b(i)) for i in range(num_data)] == [_b(i * 10) for i in range(num_data)]
        assert [omap.search(key=_b(i)) for i in range(num_data)] == [None] * num_data

    def test_rejects_contract_violations(self, omap_spec, recorded_client):
        client, recorder = recorded_client()
        omap = omap_spec.make(client=client, num_data=64)
        omap.init_server_storage([(b"1", b"kept")])
        sent = len(recorder.messages)

        cases: list[tuple[Callable[..., Any], dict[str, Any]]] = [
            (omap.insert, {"key": "1", "value": b"v"}),
            (omap.insert, {"key": b"x" * 17, "value": b"v"}),
            (omap.insert, {"key": b"2", "value": "v"}),
            (omap.insert, {"key": b"2", "value": b"x" * 17}),
            (omap.insert, {"key": b"2", "value": None}),
            (omap.search, {"key": 1}),
            (omap.search, {"key": b"1", "value": 7}),
            (omap.search, {"key": b"1", "value": b"x" * 17}),
            (omap.delete, {"key": 1}),
            (omap.delete, {"key": b"x" * 17}),
        ]
        if omap_spec.supports_batch:
            k = omap_spec.batch_size
            cases += [
                (omap.insert_batch, {"pairs": [(b"2", b"v"), (b"2", b"w")]}),
                (omap.insert_batch, {"pairs": [(_b(i), b"v") for i in range(10, 11 + k)]}),
                (omap.insert_batch, {"pairs": [(b"2", "v")]}),
                (omap.search_batch, {"pairs": [(_b(i), None) for i in range(k + 1)]}),
                (omap.search_batch, {"pairs": [(1, None)]}),
            ]
        for operation, kwargs in cases:
            with pytest.raises(ContractError, match=f"{omap_spec.name} '"):
                operation(**kwargs)

        assert len(recorder.messages) == sent
        assert omap.search(key=b"1") == b"kept"

    def test_inserts_stop_at_num_data_until_a_delete_frees_room(self, omap_spec, recorded_client):
        client, recorder = recorded_client()
        omap = omap_spec.make(client=client, num_data=16)
        omap.init_server_storage([(_b(i), b"v") for i in range(14)])
        omap.insert(key=b"14", value=b"v")
        omap.insert(key=b"15", value=b"v")
        sent = len(recorder.messages)
        with pytest.raises(ContractError, match="num_data=16"):
            omap.insert(key=b"16", value=b"v")
        if omap_spec.supports_batch:
            with pytest.raises(ContractError, match="num_data=16"):
                omap.insert_batch([(b"16", b"v")])
        assert len(recorder.messages) == sent
        assert omap.delete(key=b"absent") is None
        with pytest.raises(ContractError, match="num_data=16"):
            omap.insert(key=b"16", value=b"v")
        assert omap.delete(key=b"3") == b"v"
        omap.insert(key=b"16", value=b"v")
        assert omap.search(key=b"16") == b"v"

        sent = len(recorder.messages)
        with pytest.raises(ContractError, match="exceed num_data=16"):
            omap_spec.make(client=client, num_data=16).init_server_storage([(_b(i), b"v") for i in range(17)])
        assert len(recorder.messages) == sent

    @pytest.mark.parametrize("storage", ["plain", "encrypted", "file"])
    def test_init_with_data(self, omap_spec, client, tmp_path, encryptor, storage):
        config: dict[str, Any] = {
            "plain": {},
            "encrypted": {"encryptor": encryptor},
            "file": {"encryptor": encryptor, "build_file": tmp_path / "build.tree"},
        }[storage]
        omap = omap_spec.make(client=client, num_data=256, **config)
        omap.init_server_storage(data=[(_b(i), _b(i * 3)) for i in range(128)])
        assert list(tmp_path.iterdir()) == []
        for i in range(128, 200):
            omap.insert(key=_b(i), value=_b(i * 3))
        assert [omap.search(key=_b(i)) for i in range(200)] == [_b(i * 3) for i in range(200)]

    def test_init_rejects_repeated_key(self, omap_spec, recorded_client):
        client, recorder = recorded_client()
        omap = omap_spec.make(client=client, num_data=64)
        with pytest.raises(ContractError, match=f"{omap_spec.name} '.*given twice"):
            omap.init_server_storage(data=[(b"1", b"a"), (b"2", b"b"), (b"1", b"c")])
        assert not recorder.messages
        omap.init_server_storage(data=[(b"1", b"a"), (b"2", b"b")])
        assert omap.search(key=b"1") == b"a"

    def test_mul_tree_init_keeps_each_tree_apart(self, ods_spec, client):
        omap = ods_spec.make(client=client, num_data=64)
        with pytest.raises(ContractError, match=f"{ods_spec.name} '.*given twice"):
            omap.init_mul_tree_server_storage(data_list=[[(b"1", b"a")], [(b"2", b"b"), (b"2", b"c")]])
        with pytest.raises(ContractError, match=f"{ods_spec.name} '.*given twice"):
            omap.init_mul_tree_server_storage(data_list=[[(b"1", b"a")], [(b"1", b"b")]])

        first, second = omap.init_mul_tree_server_storage(data_list=[[(b"1", b"a")], [(b"2", b"b")]])

        def search_both():
            return [omap.search(key=b"1"), omap.search(key=b"2")]

        assert omap.on_root(second, search_both)[1] == [None, b"b"]
        assert omap.on_root(first, search_both)[1] == [b"a", None]

    def test_operations_match_a_dict_model(self, omap_spec, client, duplicate_blocks):
        rng = random.Random(2024)
        omap = omap_spec.make(client=client, num_data=128)
        omap.init_server_storage()
        model: dict[bytes, bytes] = {}
        keyspace = [_b(i) for i in range(128)]
        k = omap_spec.batch_size
        for step in range(900):
            key, value, roll = rng.choice(keyspace), _b(step), rng.random()
            if roll < 0.35:
                if key not in model:
                    omap.insert(key=key, value=value)
                    model[key] = value
            elif roll < 0.5:
                assert omap.search(key=key) == model.get(key)
            elif roll < 0.6:
                assert omap.search(key=key, value=value) == model.get(key)
                if key in model:
                    model[key] = value
            elif roll < 0.8:
                assert omap.delete(key=key) == model.pop(key, None)
            elif k > 1 and roll < 0.9:
                fresh = [each for each in keyspace if each not in model]
                pairs = [(each, value) for each in rng.sample(fresh, min(len(fresh), rng.randint(1, k)))]
                omap.insert_batch(pairs)
                model.update(pairs)
            elif k > 1:
                lookups = [(rng.choice(keyspace), rng.choice([None, value])) for _ in range(rng.randint(1, k))]
                expected = []
                for each, new in lookups:
                    expected.append(model.get(each))
                    if new is not None and each in model:
                        model[each] = new
                assert omap.search_batch(lookups) == expected
            if omap_spec.ods:
                assert omap._owed <= omap._carry_limit
        if omap_spec.ods:
            rounds = client.metrics.rounds
            omap.drain()
            assert client.metrics.rounds - rounds == -(-omap._carry_limit // omap._widest_round)
            assert omap._owed == 0
        if omap_spec.invariant_checker is not None:
            omap_spec.invariant_checker(omap, model)
        assert [omap.search(key=key) for key in keyspace] == [model.get(key) for key in keyspace]
        assert not duplicate_blocks(omap)


def _budget_workload(pattern: str) -> list[tuple[str, bytes | None]]:
    keys = [b"%02d" % i for i in range(60)]
    if pattern == "random":
        rng = random.Random(64)
        ops = [(rng.choice(["search", "insert", "delete"]), rng.choice(keys)) for _ in range(150)]
        present: set[bytes] = set()
        workload: list[tuple[str, bytes | None]] = []
        for op, key in ops:
            if op == "insert" and key in present:
                op = "search"
            workload.append((op, key))
            if op == "insert":
                present.add(key)
            elif op == "delete":
                present.discard(key)
        return [*workload, ("search", None), ("insert", None), ("delete", None)]
    ordered = keys if pattern != "reverse" else keys[::-1]
    inserts = [("insert", key) for key in ordered]
    if pattern == "churn":
        deletes = [("delete", key) for key in ordered[::2] + ordered[1::2]]
    else:
        deletes = [("delete", key) for key in ordered]
    searches = [("search", key) for key in ordered[::7]] + [("search", b"zz")]
    return [*inserts, *searches, ("delete", b"zz"), *deletes, ("search", keys[0]), ("delete", keys[0])]


def _stash_workload(pattern: str, keys: list[bytes], ops: int) -> list[tuple[str, bytes]]:
    rng = random.Random(pattern)
    if pattern == "random":
        return [(rng.choice(["insert", "search", "delete"]), rng.choice(keys)) for _ in range(ops)]
    cycle: list[tuple[str, bytes]]
    if pattern in ("sorted", "reverse"):
        ordered = keys if pattern == "sorted" else keys[::-1]
        cycle = [("insert", k) for k in ordered] + [("search", k) for k in ordered[::3]]
        cycle += [("delete", k) for k in ordered]
    else:
        edges = [("delete", keys[0]), ("insert", keys[0]), ("delete", keys[-1]), ("insert", keys[-1])] * 4
        cycle = [("insert", k) for k in keys] + [("delete", k) for k in keys[::2]] + [("insert", k) for k in keys[::2]]
        cycle += edges + [("delete", k) for k in keys[::-1]]
    return (cycle * (ops // len(cycle) + 1))[:ops]


class TestOmapStash:
    @pytest.mark.parametrize("pattern", ["random", "sorted", "reverse", "adversarial"])
    @pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
    def test_every_block_returned_to_the_stash_gets_its_own_eviction(self, ods_spec, client, pattern, distinguishable):
        omap = ods_spec.make(client=client, num_data=16, key_size=4, data_size=4, distinguishable=distinguishable)
        omap.init_server_storage()
        model = {}
        for op, key in _stash_workload(pattern, [b"%04d" % i for i in range(16)], 2000):
            if op == "insert" and key not in model:
                omap.insert(key, key)
                model[key] = key
            elif op == "delete":
                assert omap.delete(key) == model.pop(key, None)
            else:
                assert omap.search(key) == model.get(key)
            padding = sum(_round_widths(ods_spec, omap, omap._budget)[omap._op_fetches :])
            assert omap._op_owed <= padding + omap._carry_limit
        assert omap._max_stash <= omap._stash_capacity


def _round_widths(spec, omap, rounds: int) -> list[int]:
    ppr, k, height = omap._config.paths_per_round, spec.batch_size, omap._max_height
    return [ppr * (min(spec.fanout**r, k) if r < height else k) for r in range(rounds)]


def _op_types(spec, omap, present: list[bytes], absent: list[bytes]) -> dict[str, Callable[[], Any]]:
    ops: dict[str, Callable[[], Any]] = {
        "search hit": lambda: omap.search(present[0]),
        "search miss": lambda: omap.search(absent[0]),
        "search update": lambda: omap.search(present[1], b"new"),
        "insert": lambda: omap.insert(absent[1], b"v"),
        "delete hit": lambda: omap.delete(present[2]),
        "delete miss": lambda: omap.delete(absent[2]),
    }
    if spec.ods:
        ops["search dummy"] = lambda: omap.search(None)
        ops["insert dummy"] = lambda: omap.insert(None)
        ops["delete dummy"] = lambda: omap.delete(None)
    if spec.supports_batch:
        k = spec.batch_size
        ops["search batch"] = lambda: omap.search_batch([(present[0], None), (absent[0], None), (present[3], b"w")])
        ops["insert batch"] = lambda: omap.insert_batch([(key, b"v") for key in absent[3 : 3 + k]])
    return ops


class TestOmapObliviousness:
    def test_every_op_reads_the_same_widths_whatever_the_data(self, omap_spec, recorded_client):
        shapes: dict[str, set[tuple[tuple[int | str, ...], ...]]] = {}
        for size in (4, 40):
            client, recorder = recorded_client()
            omap = omap_spec.make(client=client, num_data=64)
            present = [_b(i) for i in range(0, 2 * size, 2)]
            omap.init_server_storage([(key, key) for key in present])
            exact = omap_spec.exact_labels(omap)
            for name, op in _op_types(omap_spec, omap, present, [_b(i) for i in range(1, 20, 2)]).items():
                before = len(recorder.batches)
                op()
                shape = tuple(
                    tuple(len(read.leaves) if read.label in exact else "oram" for read in batch.reads)
                    for batch in recorder.batches[before:]
                )
                shapes.setdefault(name, set()).add(shape)
            assert omap.search(present[1]) == b"new"

        assert all(len(each) == 1 for each in shapes.values()), shapes
        if omap_spec.distinguishable:
            return
        assert len(set().union(*shapes.values())) == 1, shapes
        if omap_spec.ods:
            assert shapes["search hit"] == {tuple((width,) for width in _round_widths(omap_spec, omap, omap._budget))}

    def test_rounds_never_read_more_paths_than_the_tree_has_leaves(self, omap_spec, recorded_client):
        client, recorder = recorded_client()
        omap = omap_spec.make(client=client, num_data=max(2, omap_spec.min_num_data))
        omap.init_server_storage([(b"1", b"v")])
        exact = omap_spec.exact_labels(omap)
        before = len(recorder.batches)
        assert omap.search(b"1") == b"v"
        omap.insert(b"2", b"w")
        widths = [
            len(read.leaves) for batch in recorder.batches[before:] for read in batch.reads if read.label in exact
        ]
        leaves = {read.label: client.level_of(read.label) for batch in recorder.batches for read in batch.reads}
        assert widths
        assert all(width <= 1 << (min(leaves[label] for label in exact) - 1) for width in widths)

    @pytest.mark.parametrize("pattern", ["random", "sorted", "reverse", "churn"])
    @pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
    def test_every_op_spends_exactly_its_budget(
        self, ods_spec, counting_client, duplicate_blocks, pattern, distinguishable
    ):
        client, server = counting_client
        omap = ods_spec.make(client=client, num_data=64, distinguishable=distinguishable)
        omap.init_server_storage()
        model = {}
        for op, key in _budget_workload(pattern):
            expected = ods_spec.budgets[op if distinguishable else "hidden"]
            paths_before = server.paths_read()
            if op == "insert":
                spent, _ = server.read_rounds(omap.insert, key, key)
                if key is not None:
                    model[key] = key
            elif op == "delete":
                spent, deleted = server.read_rounds(omap.delete, key)
                assert deleted == model.pop(key, None)
            else:
                spent, found = server.read_rounds(omap.search, key)
                assert found == model.get(key)
            paths = server.paths_read() - paths_before
            widths = sum(_round_widths(ods_spec, omap, expected))
            assert (spent, paths) == (expected, widths), f"{op}({key!r}): {spent} rounds, {paths} paths"
        assert not duplicate_blocks(omap)

    def test_write_deferral_preserves_access_sequence(self, omap_spec, assert_deferral_preserves_access):
        def workload(client):
            omap = omap_spec.make(client=client, num_data=64)
            omap.init_server_storage(data=[(_b(i), _b(i)) for i in range(0, 40, 2)])
            for i in range(1, 20, 2):
                omap.insert(key=_b(i), value=_b(i))
            for i in range(0, 40, 3):
                omap.search(key=_b(i))
            omap.delete(key=b"4")

        rounds_off, rounds_on = assert_deferral_preserves_access(workload)
        assert rounds_on < rounds_off


_ROW_LAYOUT_CONFIG: dict[str, Any] = {"num_data": 64, "data_size": 16, "key_size": 8, "encryptor": AesGcm()}


@pytest.mark.parametrize(
    ("make", "expected_row"),
    [
        pytest.param(lambda **kw: AVLOmap(AvlOmapConfig(**kw)), 241, id="avl"),
        pytest.param(lambda **kw: BPlusOmap(BPlusOmapConfig(order=3, **kw)), 285, id="bplus_order_3"),
    ],
)
def test_row_size_is_the_declared_node_layout(make, expected_row, client):
    omap = make(client=client, **_ROW_LAYOUT_CONFIG)
    assert omap._cipher.row_bytes == expected_row
    pairs = [(b"%08d" % i, b"\xff" * 16) for i in range(32)]
    omap.init_server_storage(pairs)
    for key, value in pairs[:4]:
        assert omap.search(key, b"\x00" * 16) == value


def test_composed_map_slots_are_sized_from_their_declared_layouts(client):
    avl = AvlOmapConfig(client=client, **_ROW_LAYOUT_CONFIG)
    bplus = BPlusOmapConfig(order=3, client=client, **_ROW_LAYOUT_CONFIG)
    assert (OramOstOmap.oram_data_size(AVLOmap, avl), OramOstOmap.oram_data_size(BPlusOmap, bplus)) == (11, 10)

    config = GroupOmapConfig(client=client, **_ROW_LAYOUT_CONFIG)
    upper_size = GroupOmap.upper_oram_data_size(config)
    upper = PathOram(PathOramConfig(num_data=64, data_size=upper_size, client=client, encryptor=AesGcm()))
    group = GroupOmap(config, upper_oram=upper)
    assert (upper_size, upper._cipher.row_bytes, group._lower_oram._cipher.row_bytes) == (33, 173, 137)
    group.init_server_storage([(b"%08d" % i, b"\xff" * 16) for i in range(32)])
    assert group.search(b"%08d" % 3) == b"\xff" * 16
