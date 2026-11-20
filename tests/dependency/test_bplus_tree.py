import random

from oblivlib.dependency import BPlusTree, BPlusTreeNode, KVPair


def _invariants(node) -> None:

    def check(n, depth):
        assert n.keys == sorted(n.keys)
        if n.is_leaf:
            return {depth}
        assert len(n.values) == len(n.keys) + 1
        return set().union(*(check(child, depth + 1) for child in n.values))

    assert len(check(node, 0)) == 1


def _shape(node):
    if node is None:
        return None
    if node.is_leaf:
        return True, list(node.keys), list(node.values)
    return False, list(node.keys), [_shape(child) for child in node.values]


class TestBPlusTree:
    def test_recursive_insert_matches_iterative(self):
        for order in (3, 4, 5):
            tree = BPlusTree(order=order, leaf_range=1000)
            root_it, root_rec = BPlusTreeNode(), BPlusTreeNode()
            for k in range(200):
                root_it = tree.insert(root=root_it, kv_pair=KVPair(key=k, value=k))
                root_rec = tree.recursive_insert(root=root_rec, kv_pair=KVPair(key=k, value=k))
            _invariants(root_it)
            assert _shape(root_it) == _shape(root_rec)

    def test_recursive_delete_matches_iterative(self):
        for order in (3, 4, 5):
            tree = BPlusTree(order=order, leaf_range=1000)
            root_it, root_rec = BPlusTreeNode(), BPlusTreeNode()
            keys = list(range(200))
            for k in keys:
                root_it = tree.insert(root=root_it, kv_pair=KVPair(key=k, value=k))
                root_rec = tree.insert(root=root_rec, kv_pair=KVPair(key=k, value=k))

            random.Random(order).shuffle(keys)
            for k in [-1, *keys, -1]:
                root_it = tree.delete(root=root_it, key=k)
                root_rec = tree.recursive_delete(root=root_rec, key=k)
                if root_it is not None:
                    _invariants(root_it)
                assert _shape(root_it) == _shape(root_rec)
            assert root_it is None and root_rec is None

    def test_get_data_list_layout(self):
        tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()
        for i in range(21):
            root = tree.insert(root=root, kv_pair=KVPair(key=i, value=i))

        data_list = tree.get_data_list(root=root)
        assert data_list[0].value.keys == root.keys
        assert len(data_list[0].value.values) == len(root.values)
