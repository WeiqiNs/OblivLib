import math
import random
from collections.abc import Callable
from typing import Any

import pytest

from oblivlib.dependency import (
    AesGcm,
    AvlOmapConfig,
    BPlusOmapConfig,
    Client,
    ContractError,
    GroupOmapConfig,
    PathOramConfig,
)
from oblivlib.omap import AVLOmap, BPlusOmap, GroupOmap, OramOstOmap
from oblivlib.oram import PathOram


def _b(number: int) -> bytes:
    return b"%d" % number


class TestOmapBehavior:
    def test_round_trip(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=256)
        omap.init_server_storage()
        for i in range(200):
            omap.insert(key=_b(i), value=_b(i))
        for i in range(200):
            assert omap.search(key=_b(i)) == _b(i)

    def test_search_empty(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=64)
        omap.init_server_storage()
        assert omap.search(key=b"5") is None
        assert omap.delete(key=b"5") is None

    def test_dummy_ops(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=64)
        omap.init_server_storage()
        for i in range(20):
            omap.insert(key=_b(i), value=_b(i))
        assert omap.search(key=None) is None
        omap.insert(key=None)
        assert omap.delete(key=None) is None
        for i in range(20):
            assert omap.search(key=_b(i)) == _b(i)

    @pytest.mark.parametrize("num_data", [1, 2, 7])
    def test_edge_sizes(self, omap_spec, num_data):
        omap = omap_spec.make(client=Client.local(), num_data=num_data)
        omap.init_server_storage()
        for i in range(num_data):
            omap.insert(key=_b(i), value=_b(i * 10))
        for i in range(num_data):
            assert omap.search(key=_b(i)) == _b(i * 10)

    def test_update_and_missing(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=128)
        omap.init_server_storage()
        for i in range(50):
            omap.insert(key=_b(i), value=_b(i))
        assert omap.search(key=b"10", value=b"999") == b"10"
        assert omap.search(key=b"10") == b"999"
        assert omap.search(key=b"10000") is None

    def test_rejects_contract_violations(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=64)
        omap.init_server_storage()
        omap.insert(key=b"1", value=b"kept")
        rounds = client.metrics.rounds

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
        for operation, kwargs in cases:
            with pytest.raises(ContractError, match=f"{omap_spec.cls.__name__} '"):
                operation(**kwargs)

        assert client.metrics.rounds == rounds
        assert omap.search(key=b"1") == b"kept"

    def test_encryption_round_trip(self, omap_spec, client, encryptor):
        omap = omap_spec.make(client=client, num_data=128, encryptor=encryptor)
        omap.init_server_storage()
        for i in range(60):
            omap.insert(key=_b(i), value=_b(i * 3))
        for i in range(60):
            assert omap.search(key=_b(i)) == _b(i * 3)

    @pytest.mark.parametrize("use_build_file", [False, True])
    def test_init_with_data(self, omap_spec, client, test_file, encryptor, use_build_file):
        build = {"build_file": test_file, "encryptor": encryptor} if use_build_file else {}
        omap = omap_spec.make(client=client, num_data=256, **build)
        omap.init_server_storage(data=[(_b(i), _b(i)) for i in range(128)])
        for i in range(128, 200):
            omap.insert(key=_b(i), value=_b(i))
        for i in range(200):
            assert omap.search(key=_b(i)) == _b(i)

    def test_init_rejects_repeated_key(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=64)
        repeated = f"{omap_spec.cls.__name__} '.*given twice"
        with pytest.raises(ContractError, match=repeated):
            omap.init_server_storage(data=[(b"1", b"a"), (b"2", b"b"), (b"1", b"c")])
        with pytest.raises(ContractError, match=repeated):
            omap.init_mul_tree_server_storage(data_list=[[(b"1", b"a")], [(b"2", b"b"), (b"2", b"c")]])

        roots = omap.init_mul_tree_server_storage(data_list=[[(b"1", b"a")], [(b"1", b"b")]])
        omap.root = roots[1]
        assert omap.search(key=b"1") == b"b"

    def test_mul_tree_init(self, omap_spec, num_data, client):
        extra = 3
        size_group = math.floor(math.log2(num_data))
        num_group = num_data // size_group
        init_data = [
            [(_b(j), _b(j)) for j in range(i * 2 * size_group, (i * 2 + 1) * size_group)] for i in range(num_group)
        ]

        omap = omap_spec.make(client=client, num_data=num_data)
        roots = omap.init_mul_tree_server_storage(data_list=init_data)

        for i, root in enumerate(roots):
            omap.root = root
            for j in range(extra):
                omap.insert(key=_b((i * 2 + 1) * size_group + j), value=_b((i * 2 + 1) * size_group + j))
            roots[i] = omap.root

        for i, root in enumerate(roots):
            omap.root = root
            for j in range(i * 2 * size_group, (i * 2 + 1) * size_group + extra):
                assert omap.search(key=_b(j)) == _b(j)

    def test_model_oracle(self, omap_spec, duplicate_blocks):
        rng = random.Random(1234)
        omap = omap_spec.make(client=Client.local(), num_data=128)
        omap.init_server_storage()
        model, keyspace = {}, [_b(i) for i in range(128)]
        for _ in range(800):
            key = rng.choice(keyspace)
            if rng.random() < 0.5:
                if key not in model:
                    value = _b(rng.randint(0, 10**6))
                    omap.insert(key=key, value=value)
                    model[key] = value
            else:
                assert omap.search(key=key) == model.get(key)
        for key in keyspace:
            assert omap.search(key=key) == model.get(key)
        assert not duplicate_blocks(omap)

    def test_invariants(self, omap_spec):
        rng = random.Random(99)
        omap = omap_spec.make(client=Client.local(), num_data=128)
        omap.init_server_storage()
        model = {}
        for _ in range(300):
            key = _b(rng.randrange(128))
            if key not in model:
                omap.insert(key=key, value=key)
                model[key] = key
        omap_spec.invariant_checker(omap, model)

    def test_delete_oracle(self, omap_spec):
        rng = random.Random(2024)
        omap = omap_spec.make(client=Client.local(), num_data=128)
        omap.init_server_storage()
        model, keyspace = {}, [_b(i) for i in range(128)]
        for _ in range(900):
            key = rng.choice(keyspace)
            roll = rng.random()
            if roll < 0.45:
                if key not in model:
                    value = _b(rng.randint(0, 10**6))
                    omap.insert(key=key, value=value)
                    model[key] = value
            elif roll < 0.75:
                assert omap.search(key=key) == model.get(key)
            elif model:
                assert omap.delete(key=key) == model.get(key)
                model.pop(key, None)
        for key in keyspace:
            assert omap.search(key=key) == model.get(key)
        omap_spec.invariant_checker(omap, model)

    def test_delete_empties_tree(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=64)
        omap.init_server_storage()
        for i in range(40):
            omap.insert(key=_b(i), value=_b(i))
        for i in range(40):
            assert omap.delete(key=_b(i)) == _b(i)
        assert omap.root is None


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


class TestOmapObliviousness:
    def test_search_access_uniform(self, omap_spec, counting_client):
        client, server = counting_client
        omap = omap_spec.make(client=client, num_data=256)
        omap.init_server_storage()
        for i in range(200):
            omap.insert(key=_b(i), value=_b(i))
        hit = {server.rounds(omap.search, _b(k))[0] for k in random.Random(0).sample(range(200), 40)}
        miss = {server.rounds(omap.search, _b(10_000 + j))[0] for j in range(10)}
        assert len(hit) == 1, f"search rounds vary by key: {sorted(hit)}"
        assert miss == hit, f"missing-key search distinguishable from hit: miss={miss} hit={hit}"
        dummy = {server.rounds(omap.search, None)[0] for _ in range(5)}
        assert dummy == hit, f"search(None) distinguishable from a real search: {dummy} vs {hit}"

    def test_insert_access_uniform(self, omap_spec, counting_client):
        client, server = counting_client
        omap = omap_spec.make(client=client, num_data=256)
        omap.init_server_storage()
        rounds = {server.rounds(omap.insert, _b(i), _b(i))[0] for i in range(200)}
        assert len(rounds) == 1, f"insert rounds vary: {sorted(rounds)}"
        dummy = {server.rounds(omap.insert, None)[0] for _ in range(5)}
        assert dummy == rounds, f"insert(None) distinguishable from a real insert: {dummy} vs {rounds}"

    def test_delete_access_uniform(self, omap_spec, counting_client):
        client, server = counting_client
        omap = omap_spec.make(client=client, num_data=256)
        omap.init_server_storage()
        for i in range(200):
            omap.insert(key=_b(i), value=_b(i))
        existing = {server.rounds(omap.delete, _b(k))[0] for k in random.Random(3).sample(range(200), 30)}
        missing = {server.rounds(omap.delete, _b(10_000 + j))[0] for j in range(10)}
        dummy = {server.rounds(omap.delete, None)[0] for _ in range(5)}
        assert len(existing) == 1, f"delete(existing) rounds vary: {sorted(existing)}"
        assert missing == existing, f"delete(missing) distinguishable: miss={missing} hit={existing}"
        assert dummy == existing, f"delete(None) distinguishable: dummy={dummy} hit={existing}"

    @pytest.mark.parametrize("pattern", ["random", "sorted", "reverse", "churn"])
    @pytest.mark.parametrize("distinguishable", [False, True], ids=["hidden", "distinguishable"])
    def test_every_op_spends_exactly_its_budget(
        self, omap_spec, counting_client, duplicate_blocks, pattern, distinguishable
    ):
        client, server = counting_client
        omap = omap_spec.make(client=client, num_data=64, distinguishable=distinguishable)
        omap.init_server_storage()
        model = {}
        for op, key in _budget_workload(pattern):
            expected = omap_spec.budgets[op] if distinguishable else max(omap_spec.budgets.values())
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
            assert spent == expected, f"{op}({key!r}) spent {spent} rounds, budget {expected}"
            paths = server.paths_read() - paths_before
            assert paths == expected * omap._round_width, f"{op}({key!r}) read {paths} paths in {spent} rounds"
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
        assert rounds_on * 2 == rounds_off


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
    avl = AVLOmap(AvlOmapConfig(client=client, **_ROW_LAYOUT_CONFIG))
    bplus = BPlusOmap(BPlusOmapConfig(order=3, client=client, **_ROW_LAYOUT_CONFIG))
    assert (OramOstOmap.oram_data_size(avl), OramOstOmap.oram_data_size(bplus)) == (11, 10)

    upper_size = GroupOmap.upper_oram_data_size(num_data=64, key_size=8)
    upper = PathOram(PathOramConfig(num_data=64, data_size=upper_size, client=client, encryptor=AesGcm()))
    group = GroupOmap(GroupOmapConfig(client=client, **_ROW_LAYOUT_CONFIG), upper_oram=upper)
    assert (upper_size, upper._cipher.row_bytes, group._lower_oram._cipher.row_bytes) == (456, 1869, 137)
    group.init_server_storage([(b"%08d" % i, b"\xff" * 16) for i in range(32)])
    assert group.search(b"%08d" % 3) == b"\xff" * 16
