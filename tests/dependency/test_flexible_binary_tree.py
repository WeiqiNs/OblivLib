import random

import pytest

from oblivlib.dependency import AesGcm, Client, Data, PathData, ScaleDownError
from oblivlib.dependency.flexible_binary_tree import FlexibleBinaryTree
from oblivlib.dependency.layout import Bytes, UInt, block_layout
from oblivlib.dependency.path_cipher import make_path_cipher

_BLOCK = block_layout(UInt(16), UInt(16), Bytes(64))


class _TreeFactory:
    def __init__(self, mode: str):
        self._mode = mode

    def create(self, num_data: int, *, blocks: list[Data]) -> tuple[FlexibleBinaryTree, list[Data]]:
        cipher = make_path_cipher(_BLOCK, None if self._mode == "memory-plain" else AesGcm(), bucket_size=2)
        return FlexibleBinaryTree.create(
            client=Client.local(), label="flex", num_data=num_data, bucket_size=2, cipher=cipher, blocks=blocks
        )

    def __call__(self, num_data: int, *, blocks: list[Data] | None = None) -> FlexibleBinaryTree:
        return self.create(num_data, blocks=blocks or [])[0]


@pytest.fixture(params=["memory-plain", "memory-enc"])
def make_tree(request):
    return _TreeFactory(request.param)


def _naive_path(leaf: int, level: int) -> list[int]:
    index, path = leaf + (1 << (level - 1)) - 1, []
    while index >= 0:
        path.append(index)
        index = (index - 1) // 2
    return path


def _block(key: int, leaf: int = 0) -> Data:
    return Data(key=key, leaf=leaf, value=b"v%d" % key)


class TestFlexibleBinaryTree:
    def test_create_all_nodes_present(self, make_tree):
        tree = make_tree(4)
        assert tree.level == 3
        assert tree.read_path(list(range(4))) == {index: [] for index in range(7)}

    def test_create_places_blocks_and_returns_overflow(self, make_tree):
        tree, overflow = make_tree.create(4, blocks=[_block(key, leaf=1) for key in range(7)])
        assert overflow == [_block(6, 1)]
        assert tree.read_path([1]) == {
            0: [_block(4, 1), _block(5, 1)],
            1: [_block(2, 1), _block(3, 1)],
            4: [_block(0, 1), _block(1, 1)],
        }

    def test_read_path_root_down_skips_absent(self, make_tree):
        tree = make_tree(4)
        tree.write_path([0], {0: [_block(0)], 3: [_block(3)]})

        path = tree.read_path([0, 1])
        assert list(path) == [0, 3, 4]
        assert path == {0: [_block(0)], 3: [_block(3)], 4: []}

    def test_write_path_rejects_off_path_index(self, make_tree):
        tree = make_tree(4)
        with pytest.raises(ValueError, match=r"\[4\]"):
            tree.write_path([0], {0: [_block(0)], 4: [_block(4)]})
        assert tree.read_path([0]) == {0: [], 1: [], 3: []}

    def test_scale_down_at_level_one_raises(self, make_tree):
        tree = make_tree(1)
        tree.write_path([0], {})
        with pytest.raises(ScaleDownError):
            tree.scale_down()
        assert tree.level == 1

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_ops_match_model(self, make_tree, seed):
        rng = random.Random(seed)
        tree = make_tree(4)
        level = 3
        model: PathData = {index: [] for index in range(7)}

        for step in range(300):
            all_leaves = list(range(1 << (level - 1)))
            bottom = range((1 << (level - 1)) - 1, (1 << level) - 1)
            leaves = rng.sample(all_leaves, rng.randint(1, min(3, len(all_leaves))))
            nodes = sorted({index for leaf in leaves for index in _naive_path(leaf, level)})
            roll = rng.random()
            if roll < 0.35:
                data: PathData = {index: [_block(step * 100 + index)] for index in nodes if rng.random() < 0.5}
                tree.write_path(leaves, data)
                for index in nodes:
                    model.pop(index, None)
                model.update(data)
            elif roll < 0.65:
                assert tree.read_path(leaves) == {index: model[index] for index in nodes if index in model}
            elif roll < 0.75:
                kept = {index: bucket for index, bucket in model.items() if index not in bottom}
                tree.write_path(all_leaves, kept)
                model = kept
            elif roll < 0.85 and level < 6:
                tree.scale_up()
                level += 1
            elif level == 1 or any(index in model for index in bottom):
                with pytest.raises(ScaleDownError):
                    tree.scale_down()
            else:
                tree.scale_down()
                level -= 1
            assert tree.level == level
