import pytest

from oblivlib.dependency import AvlOmapCachedConfig, AvlOmapConfig, BPlusOmapCachedConfig, BPlusOmapConfig
from oblivlib.dependency.protocol import ReadPath
from oblivlib.omap import AVLOmap, AVLOmapCached, BPlusOmap, BPlusOmapCached


class ReadCounter:
    def __init__(self, recorder):
        self._recorder = recorder

    def _leaves_read(self) -> int:
        return sum(len(op.leaves) for batch in self._recorder.batches for op in batch.reads if isinstance(op, ReadPath))

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
        value = data.value
        l_height = walk(value.l_key, value.l_leaf, low, key) if value.l_key is not None else 0
        r_height = walk(value.r_key, value.r_leaf, key, high) if value.r_key is not None else 0
        assert value.l_height == l_height, f"node {key}: stored l_height {value.l_height} != {l_height}"
        assert value.r_height == r_height, f"node {key}: stored r_height {value.r_height} != {r_height}"
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

    def is_leaf(value):
        return len(value.keys) == len(value.values)

    def walk(block_id, leaf, low, high, depth):
        assert block_id in blocks, f"block {block_id} referenced by a pointer but not present"
        data = blocks[block_id]
        assert data.leaf == leaf, f"block {block_id}: stored leaf {data.leaf} != pointer {leaf}"
        value = data.value
        assert list(value.keys) == sorted(value.keys), f"keys unsorted in block {block_id}: {value.keys}"
        assert len(value.keys) < omap._order, f"block {block_id} overfull: {len(value.keys)} keys"
        if depth > 0:
            assert len(value.keys) >= min_keys, f"block {block_id} underfull: {len(value.keys)} keys"
        for k in value.keys:
            assert (low is None or low <= k) and (high is None or k < high), (
                f"key {k} outside routing bounds ({low}, {high}) in block {block_id}"
            )
        if is_leaf(value):
            leaf_depths.add(depth)
            for k, v in zip(value.keys, value.values, strict=True):
                assert k not in collected, f"duplicate key {k} across leaves"
                collected[k] = v
        else:
            assert len(value.values) == len(value.keys) + 1, f"internal arity off in block {block_id}"
            low_i = low
            for i, (child_id, child_leaf) in enumerate(value.values):
                high_i = value.keys[i] if i < len(value.keys) else high
                walk(child_id, child_leaf, low_i, high_i, depth + 1)
                low_i = value.keys[i] if i < len(value.keys) else low_i

    root_id, root_leaf = omap.root
    walk(root_id, root_leaf, None, None, 0)
    assert len(leaf_depths) == 1, f"leaves at several depths: {sorted(leaf_depths)}"
    assert collected == model, (
        f"leaves != model (missing {set(model) - set(collected)}, extra {set(collected) - set(model)})"
    )


class OmapSpec:
    def __init__(
        self,
        cls,
        config_cls,
        *,
        budgets,
        invariant_checker,
        extra=None,
    ):
        self.cls = cls
        self.config_cls = config_cls
        self.budgets = budgets
        self.invariant_checker = invariant_checker
        self.extra = extra or {}

    def make(self, *, client, num_data, key_size=16, data_size=16, **kw):
        config_kw = {**self.extra, **kw}
        return self.cls(
            self.config_cls(num_data=num_data, key_size=key_size, data_size=data_size, client=client, **config_kw)
        )


OMAP_SPECS = [
    pytest.param(
        OmapSpec(
            AVLOmap,
            AvlOmapConfig,
            budgets={"search": 8, "insert": 8, "delete": 22},
            invariant_checker=verify_avl_invariants,
        ),
        id="avl",
    ),
    pytest.param(
        OmapSpec(
            AVLOmapCached,
            AvlOmapCachedConfig,
            budgets={"search": 8, "insert": 8, "delete": 22},
            invariant_checker=verify_avl_invariants,
        ),
        id="avl_cached",
    ),
    pytest.param(
        OmapSpec(
            BPlusOmap,
            BPlusOmapConfig,
            budgets={"search": 4, "insert": 4, "delete": 7},
            invariant_checker=verify_bplus_invariants,
            extra={"order": 5},
        ),
        id="bplus",
    ),
    pytest.param(
        OmapSpec(
            BPlusOmapCached,
            BPlusOmapCachedConfig,
            budgets={"search": 4, "insert": 4, "delete": 4},
            invariant_checker=verify_bplus_invariants,
            extra={"order": 5},
        ),
        id="bplus_cached",
    ),
]


@pytest.fixture(params=OMAP_SPECS)
def omap_spec(request):
    return request.param


@pytest.fixture
def counting_client(recorded_client):
    client, recorder = recorded_client()
    return client, ReadCounter(recorder)


@pytest.fixture
def check_bplus_invariants():
    return verify_bplus_invariants
