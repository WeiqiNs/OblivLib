import random

from oblivlib.dependency import AVLTree, KVPair


def _invariants(node, low=None, high=None) -> int:
    if node is None:
        return 0
    assert low is None or node.key > low
    assert high is None or node.key < high
    left = _invariants(node.left_node, low, node.key)
    right = _invariants(node.right_node, node.key, high)
    assert abs(left - right) <= 1
    assert node.height == 1 + max(left, right)
    return node.height


def _shape(node):
    if node is None:
        return None
    return node.key, node.value, node.height, _shape(node.left_node), _shape(node.right_node)


class TestAVLTree:
    def test_recursive_insert_matches_iterative(self):
        tree = AVLTree(leaf_range=1000)
        root_it = root_rec = None
        for k in random.Random(5).sample(range(10000), 500):
            root_it = tree.insert(root=root_it, kv_pair=KVPair(key=k, value=k))
            root_rec = tree.recursive_insert(root=root_rec, kv_pair=KVPair(key=k, value=k))
        _invariants(root_it)
        assert _shape(root_it) == _shape(root_rec)

    def test_recursive_delete_matches_iterative(self):
        tree = AVLTree(leaf_range=1000)
        root_it = root_rec = None
        keys = list(range(300))
        for k in keys:
            root_it = tree.insert(root=root_it, kv_pair=KVPair(key=k, value=k))
            root_rec = tree.insert(root=root_rec, kv_pair=KVPair(key=k, value=k))

        random.Random(0).shuffle(keys)
        for k in [-1, *keys, -1]:
            root_it = tree.delete(root=root_it, key=k)
            root_rec = tree.recursive_delete(root=root_rec, key=k)
            _invariants(root_it)
            assert _shape(root_it) == _shape(root_rec)
        assert root_it is None and root_rec is None

    def test_get_data_list_layout(self):
        tree = AVLTree(leaf_range=10)
        root = None
        for i in range(10):
            root = tree.insert(root=root, kv_pair=KVPair(key=i, value=i))
        assert root is not None

        data_list = tree.get_data_list(root=root)
        assert len(data_list) == 10
        assert root.left_node is not None and root.right_node is not None
        assert data_list[0].key == root.key
        assert data_list[0].value.l_key == root.left_node.key
        assert data_list[0].value.r_key == root.right_node.key
