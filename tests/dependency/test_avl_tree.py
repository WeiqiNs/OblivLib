from oblivlib.dependency import AVLTree, KVPair


class TestAVLTree:
    def test_get_data_list_layout(self):
        tree = AVLTree(leaf_range=10)
        root = None
        for i in range(10):
            root = tree.recursive_insert(root=root, kv_pair=KVPair(key=i, value=i))
        assert root is not None

        data_list = tree.get_data_list(root=root)
        assert len(data_list) == 10
        assert root.left_node is not None and root.right_node is not None
        assert data_list[0].key == root.key
        assert data_list[0].value.l_key == root.left_node.key
        assert data_list[0].value.r_key == root.right_node.key
