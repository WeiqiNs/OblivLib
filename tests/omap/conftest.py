from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from oblivlib.dependency import (
    AccessBudget,
    AvlOmapConfig,
    BPlusLeaf,
    BPlusOmapConfig,
    DaOramConfig,
    GroupOmapConfig,
    OramOstOmapConfig,
    PathOramConfig,
    RecursiveOramConfig,
)
from oblivlib.dependency.protocol import ReadPath
from oblivlib.omap import AVLOmap, BPlusOmap, GroupOmap, OramOstOmap
from oblivlib.oram import DAOram, PathOram, RecursivePathOram


class ReadCounter:
    def __init__(self, recorder):
        self._recorder = recorder

    def _leaves_read(self) -> int:
        return sum(len(op.leaves) for batch in self._recorder.batches for op in batch.reads if isinstance(op, ReadPath))

    def paths_read(self) -> int:
        return self._leaves_read()

    def _read_rounds(self) -> int:
        return sum(any(isinstance(op, ReadPath) for op in batch.reads) for batch in self._recorder.batches)

    def read_rounds(self, fn, *args, **kwargs):
        before = self._read_rounds()
        result = fn(*args, **kwargs)
        return self._read_rounds() - before, result

    def rounds(self, fn, *args, **kwargs):
        before = self._leaves_read()
        result = fn(*args, **kwargs)
        return self._leaves_read() - before, result


def _read_all_blocks(scheme):
    client = scheme._client
    client.add_read_path(scheme._name, range(scheme._leaf_range))
    path = scheme._cipher.open_path(client.execute().require(scheme._name))
    return [data for bucket in path.values() for data in bucket]


def _live_blocks(omap):
    held = _read_all_blocks(omap) + omap._stash + omap._local.to_list()
    blocks = {data.key: data for data in held}
    assert len(blocks) == len(held), "a node is held more than once"
    return blocks


def verify_avl_invariants(omap, model):
    nodes = _live_blocks(omap)
    assert set(nodes) == set(model), f"live keys mismatch: have {len(nodes)}, expected {len(model)}"
    if omap.root is None:
        assert not model
        return

    def walk(key, leaf, low, high):
        assert key in nodes, f"node {key} referenced by a pointer but not present"
        data = nodes[key]
        assert data.leaf == leaf, f"node {key}: stored leaf {data.leaf} != parent pointer {leaf}"
        assert (low is None or low < key) and (high is None or key < high), (
            f"BST order violated at {key} (bounds {low}, {high})"
        )
        l_height = walk(data.l_key, data.l_leaf, low, key) if data.l_key is not None else 0
        r_height = walk(data.r_key, data.r_leaf, key, high) if data.r_key is not None else 0
        assert data.l_height == l_height, f"node {key}: stored l_height {data.l_height} != {l_height}"
        assert data.r_height == r_height, f"node {key}: stored r_height {data.r_height} != {r_height}"
        assert abs(l_height - r_height) <= 1, f"node {key} unbalanced (l={l_height}, r={r_height})"
        return 1 + max(l_height, r_height)

    root_key, root_leaf = omap.root
    walk(root_key, root_leaf, None, None)


def verify_bplus_invariants(omap, model):
    if omap.root is None:
        assert not model
        return
    blocks = _live_blocks(omap)
    collected = {}
    leaf_depths = set()
    min_keys = (omap._order - 1) // 2

    def walk(block_id, leaf, low, high, depth):
        assert block_id in blocks, f"block {block_id} referenced by a pointer but not present"
        data = blocks[block_id]
        assert data.leaf == leaf, f"block {block_id}: stored leaf {data.leaf} != pointer {leaf}"
        value = data
        assert list(value.keys) == sorted(value.keys), f"keys unsorted in block {block_id}: {value.keys}"
        assert len(value.keys) < omap._order, f"block {block_id} overfull: {len(value.keys)} keys"
        if depth > 0:
            assert len(value.keys) >= min_keys, f"block {block_id} underfull: {len(value.keys)} keys"
        for k in value.keys:
            assert (low is None or low <= k) and (high is None or k < high), (
                f"key {k} outside routing bounds ({low}, {high}) in block {block_id}"
            )
        if isinstance(value, BPlusLeaf):
            leaf_depths.add(depth)
            for k, v in zip(value.keys, value.values, strict=True):
                assert k not in collected, f"duplicate key {k} across leaves"
                collected[k] = v
        else:
            assert len(value.children) == len(value.keys) + 1, f"internal arity off in block {block_id}"
            low_i = low
            for i, (child_id, child_leaf) in enumerate(value.children):
                high_i = value.keys[i] if i < len(value.keys) else high
                walk(child_id, child_leaf, low_i, high_i, depth + 1)
                low_i = value.keys[i] if i < len(value.keys) else low_i

    root_id, root_leaf = omap.root
    walk(root_id, root_leaf, None, None, 0)
    assert len(leaf_depths) == 1, f"leaves at several depths: {sorted(leaf_depths)}"
    assert collected == model, (
        f"leaves != model (missing {set(model) - set(collected)}, extra {set(collected) - set(model)})"
    )


ORAM_MAKERS = {
    "path": lambda **kw: PathOram(PathOramConfig(**kw)),
    "da": lambda **kw: DAOram(DaOramConfig(**kw)),
    "recursive": lambda **kw: RecursivePathOram(RecursiveOramConfig(**kw)),
}


@dataclass(frozen=True, kw_only=True)
class OmapSpec:
    name: str
    build: Callable[..., Any]
    ods: bool = False
    budgets: dict[str, int] | None = None
    invariant_checker: Callable[[Any, dict[bytes, bytes]], None] | None = None
    fanout: int = 2
    batch_size: int = 1
    distinguishable: bool = False
    min_num_data: int = 1

    @property
    def supports_batch(self) -> bool:
        return self.batch_size > 1

    def make(self, *, client, num_data, key_size=16, data_size=16, **kw):
        if self.distinguishable:
            kw.setdefault("distinguishable", True)
        return self.build(client=client, num_data=num_data, key_size=key_size, data_size=data_size, **kw)

    @staticmethod
    def exact_labels(omap) -> set[str]:
        if isinstance(omap, OramOstOmap):
            return {omap._ost._name}
        if isinstance(omap, GroupOmap):
            return {omap._lower_oram._name}
        return {omap._name}


def _ods_builder(cls, config_cls, extra):
    def build(**kw):
        return cls(config_cls(**extra, **kw))

    return build


def _upper_build_file(build_file):
    return None if build_file is None else Path(f"{build_file}.upper")


def _ost_builder(ods_cls, ods_config_cls, extra, oram_kind):
    def build(*, client, num_data, encryptor=None, build_file=None, **kw):
        ost = ods_cls(
            ods_config_cls(num_data=num_data, client=client, encryptor=encryptor, build_file=build_file, **extra, **kw)
        )
        oram = ORAM_MAKERS[oram_kind](
            num_data=num_data,
            data_size=OramOstOmap.oram_data_size(ost),
            client=client,
            encryptor=encryptor,
            build_file=_upper_build_file(build_file),
        )
        return OramOstOmap(OramOstOmapConfig(num_data=num_data), ost=ost, oram=oram)

    return build


def _group_builder(oram_kind):
    def build(*, client, num_data, encryptor=None, build_file=None, **kw):
        upper = ORAM_MAKERS[oram_kind](
            num_data=num_data,
            data_size=GroupOmap.upper_oram_data_size(num_data),
            client=client,
            encryptor=encryptor,
            build_file=_upper_build_file(build_file),
        )
        config = GroupOmapConfig(num_data=num_data, client=client, encryptor=encryptor, build_file=build_file, **kw)
        return GroupOmap(config, upper_oram=upper)

    return build


def _avl(budgets, **extra):
    return OmapSpec(
        name="AVLOmap",
        build=_ods_builder(AVLOmap, AvlOmapConfig, extra),
        ods=True,
        budgets=budgets,
        invariant_checker=verify_avl_invariants,
        batch_size=extra.get("batch_size", 1),
        min_num_data=extra.get("batch_size", 1),
    )


def _bplus(budgets, **extra):
    return OmapSpec(
        name="BPlusOmap",
        build=_ods_builder(BPlusOmap, BPlusOmapConfig, extra),
        ods=True,
        budgets=budgets,
        invariant_checker=verify_bplus_invariants,
        fanout=extra["order"],
        batch_size=extra.get("batch_size", 1),
        min_num_data=extra.get("batch_size", 1),
    )


def _ost(ods_kind, oram_kind):
    ods = {"avl": (AVLOmap, AvlOmapConfig, {}), "bplus": (BPlusOmap, BPlusOmapConfig, {"order": 5})}[ods_kind]
    spec = OmapSpec(
        name="OramOstOmap",
        build=_ost_builder(*ods, oram_kind),
        min_num_data=1 if oram_kind == "path" else 11,
    )
    return f"ost_{ods_kind}_{oram_kind}", spec


def _group(oram_kind, *, distinguishable):
    spec = OmapSpec(
        name="GroupOmap",
        build=_group_builder(oram_kind),
        distinguishable=distinguishable,
        min_num_data=1 if oram_kind == "path" else 11,
    )
    return f"group_{oram_kind}" + ("_distinguishable" if distinguishable else ""), spec


_SPECS: list[tuple[str, OmapSpec]] = [
    ("avl", _avl({"search": 8, "insert": 16, "delete": 16, "hidden": 16})),
    ("avl_cached", _avl({"search": 8, "insert": 9, "delete": 11, "hidden": 11}, cached=True)),
    ("avl_batched", _avl({"search": 8, "insert": 16, "delete": 16, "hidden": 16}, batch_size=4)),
    ("avl_batched_cached", _avl({"search": 8, "insert": 9, "delete": 11, "hidden": 11}, batch_size=4, cached=True)),
    (
        "avl_raised_budgets",
        _avl(
            {"search": 25, "insert": 24, "delete": 26, "hidden": 26},
            search_budget=AccessBudget(3, 1),
            insert_budget=AccessBudget(3, 0),
            delete_budget=AccessBudget(3, 2),
        ),
    ),
    ("bplus", _bplus({"search": 4, "insert": 7, "delete": 7, "hidden": 7}, order=5)),
    ("bplus_one_path", _bplus({"search": 4, "insert": 10, "delete": 13, "hidden": 13}, order=5, paths_per_round=1)),
    ("bplus_cached", _bplus({"search": 4, "insert": 4, "delete": 4, "hidden": 4}, order=5, cached=True)),
    (
        "bplus_one_path_cached",
        _bplus({"search": 4, "insert": 7, "delete": 7, "hidden": 7}, order=5, paths_per_round=1, cached=True),
    ),
    ("bplus_batched", _bplus({"search": 6, "insert": 11, "delete": 11, "hidden": 11}, order=3, batch_size=4)),
    (
        "bplus_batched_cached",
        _bplus({"search": 6, "insert": 6, "delete": 6, "hidden": 6}, order=3, batch_size=4, cached=True),
    ),
    (
        "bplus_batched_one_path_cached",
        _bplus(
            {"search": 6, "insert": 16, "delete": 11, "hidden": 16},
            order=3,
            batch_size=5,
            paths_per_round=1,
            cached=True,
        ),
    ),
    (
        "bplus_one_path_raised_budgets",
        _bplus(
            {"search": 8, "insert": 12, "delete": 13, "hidden": 13},
            order=5,
            paths_per_round=1,
            search_budget=AccessBudget(2, 0),
            insert_budget=AccessBudget(3, 0),
        ),
    ),
    *(_ost(ods_kind, oram_kind) for ods_kind in ("avl", "bplus") for oram_kind in ("path", "da", "recursive")),
    *(_group(oram_kind, distinguishable=flag) for oram_kind in ("path", "da", "recursive") for flag in (False, True)),
]


@pytest.fixture(params=[pytest.param(spec, id=spec_id) for spec_id, spec in _SPECS])
def omap_spec(request):
    return request.param


@pytest.fixture(params=[pytest.param(spec, id=spec_id) for spec_id, spec in _SPECS if spec.ods])
def ods_spec(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param((spec, num_data), id=f"{spec_id}-{num_data}")
        for spec_id, spec in _SPECS
        for num_data in sorted({1, 2, 7, spec.min_num_data, spec.min_num_data + 1})
        if num_data >= spec.min_num_data
    ]
)
def omap_edge_size(request):
    return request.param


@pytest.fixture
def counting_client(recorded_client):
    client, recorder = recorded_client()
    return client, ReadCounter(recorder)


@pytest.fixture
def check_bplus_invariants():
    return verify_bplus_invariants
