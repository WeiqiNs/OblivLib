import itertools
from bisect import bisect_right
from collections.abc import Callable, Sequence
from typing import override

from oblivlib.dependency import BPlusInternal, BPlusLeaf, BPlusNode
from oblivlib.dependency.config import BPlusOmapConfig
from oblivlib.dependency.layout import Buffer, Bytes, Field, Pair, Seq, UInt
from oblivlib.omap.base_omap import KvList
from oblivlib.omap.ost_base_omap import LocalNodes, OstBaseOmap, Root

_BLOCK_ID_BITS = 64
_LEAF_TAG = 0
_INTERNAL_TAG = 1


def bplus_max_levels(num_keys: int, order: int) -> int:
    fanout = -(-order // 2)
    levels, split_keys, subtree_keys = 1, order, fanout - 1
    while split_keys <= num_keys:
        levels += 1
        split_keys += (order - 1) * subtree_keys
        subtree_keys *= fanout
    return levels


def _even_chunks(count: int, capacity: int) -> list[tuple[int, int]]:
    chunks = -(-count // capacity)
    bounds = [count * index // chunks for index in range(chunks + 1)]
    return list(itertools.pairwise(bounds))


class _BPlusNodeLayout(Field[BPlusNode]):
    def __init__(self, *, key: Field[int], leaf: Field[int], order: int, key_size: int, data_size: int):
        self._key: Field[int] = key
        self._leaf: Field[int] = leaf
        self._keys: Seq[bytes] = Seq(Bytes(key_size), order - 1)
        self._values: Seq[bytes] = Seq(Bytes(data_size), order - 1)
        self._children: Seq[tuple[int, int]] = Seq(Pair(key, leaf), order)
        self._width: int = self._keys.size + max(self._values.size, self._children.size)

    @property
    @override
    def size(self) -> int:
        return self._key.size + self._leaf.size + 1 + self._width

    @override
    def write(self, value: BPlusNode, out: bytearray) -> None:
        self._key.write(value.key, out)
        self._leaf.write(value.leaf, out)
        start = len(out) + 1
        if isinstance(value, BPlusLeaf):
            out.append(_LEAF_TAG)
            self._keys.write(value.keys, out)
            self._values.write(value.values, out)
        else:
            out.append(_INTERNAL_TAG)
            self._keys.write(value.keys, out)
            self._children.write(value.children, out)
        out += bytes(self._width - (len(out) - start))

    @override
    def read(self, data: Buffer, offset: int) -> tuple[BPlusNode, int]:
        key, offset = self._key.read(data, offset)
        leaf, offset = self._leaf.read(data, offset)
        tag, end = data[offset], offset + 1 + self._width
        keys, offset = self._keys.read(data, offset + 1)
        if tag == _LEAF_TAG:
            values, _ = self._values.read(data, offset)
            return BPlusLeaf(key=key, leaf=leaf, keys=keys, values=values), end
        children, _ = self._children.read(data, offset)
        return BPlusInternal(key=key, leaf=leaf, keys=keys, children=children), end


class BPlusOmap(OstBaseOmap[BPlusOmapConfig, int, BPlusNode]):
    def __init__(self, config: BPlusOmapConfig, *, height_keys: int | None = None):
        super().__init__(config, height_keys=height_keys)

        self._local_nodes: LocalNodes[int, BPlusNode] = LocalNodes()
        self._mid: int = config.order // 2
        self._block_id: int = 0

    @property
    @override
    def _local(self) -> LocalNodes[int, BPlusNode]:
        return self._local_nodes

    @property
    @override
    def _fanout(self) -> int:
        return self._order

    @override
    def _height_bound(self, num_nodes: int) -> int:
        return bplus_max_levels(num_nodes, self._order)

    @property
    def _order(self) -> int:
        return self._config.order

    @property
    def _min_keys(self) -> int:
        return (self._order - 1) // 2

    @classmethod
    @override
    def _key_field(cls, config: BPlusOmapConfig) -> Field[int]:
        return UInt(_BLOCK_ID_BITS)

    @property
    @override
    def _block_layout(self) -> Field[BPlusNode]:
        return _BPlusNodeLayout(
            key=self._key_field(self._config),
            leaf=self._leaf_field,
            order=self._order,
            key_size=self.key_size,
            data_size=self.data_size,
        )

    def _next_block(self) -> tuple[int, int]:
        block = self._block_id, self._get_new_leaf()
        self._block_id += 1
        return block

    def _new_leaf_node(self, keys: list[bytes], values: list[bytes]) -> BPlusLeaf:
        key, leaf = self._next_block()
        return BPlusLeaf(key=key, leaf=leaf, keys=keys, values=values)

    def _new_internal_node(self, keys: list[bytes], children: list[tuple[int, int]]) -> BPlusInternal:
        key, leaf = self._next_block()
        return BPlusInternal(key=key, leaf=leaf, keys=keys, children=children)

    @override
    def _build_ods_blocks(self, data: KvList) -> tuple[list[BPlusNode], Root[int]]:
        pairs = sorted(data, key=lambda pair: pair[0])
        level: list[tuple[bytes, BPlusNode]] = []
        for start, end in _even_chunks(len(pairs), self._order - 1):
            chunk = pairs[start:end]
            leaf = self._new_leaf_node(keys=[key for key, _ in chunk], values=[value for _, value in chunk])
            level.append((chunk[0][0], leaf))
        blocks = [node for _, node in level]
        while len(level) > 1:
            parents: list[tuple[bytes, BPlusNode]] = []
            for start, end in _even_chunks(len(level), self._order):
                group = level[start:end]
                parent = self._new_internal_node(
                    keys=[low for low, _ in group[1:]], children=[(node.key, node.leaf) for _, node in group]
                )
                blocks.append(parent)
                parents.append((group[0][0], parent))
            level = parents
        root = level[0][1]
        return blocks, (root.key, root.leaf)

    def _require_internal(self, key: int) -> BPlusInternal:
        node = self._local.require(key)
        assert isinstance(node, BPlusInternal)
        return node

    def _require_leaf(self, key: int) -> BPlusLeaf:
        node = self._local.require(key)
        assert isinstance(node, BPlusLeaf)
        return node

    @override
    def _descend(
        self, keys: Sequence[bytes], *, hold: bool, on_stop: Callable[[int, BPlusNode], None] | None = None
    ) -> None:
        root_key = self._fetch_root().key
        at = [root_key] * len(keys)
        level = [root_key]
        while isinstance(self._local.require(at[0]), BPlusInternal):
            requests: dict[int, tuple[int, int]] = {}
            for lane, node_key in enumerate(at):
                node = self._require_internal(node_key)
                child_index = bisect_right(node.keys, keys[lane])
                child_key, child_leaf = node.children[child_index]
                if child_key not in requests:
                    new_leaf = self._get_new_leaf()
                    node.children[child_index] = (child_key, new_leaf)
                    requests[child_key] = (child_leaf, new_leaf)
                at[lane] = child_key
            if not hold:
                self._return_local(level)
            self._fetch_rehomed(requests)
            level = list(requests)
        if on_stop is not None:
            for lane, node_key in enumerate(at):
                on_stop(lane, self._local.require(node_key))

    def _path_to(self, key: bytes) -> None:
        assert self._root is not None
        path: list[int] = []
        node_key = self._root[0]
        while True:
            path.append(node_key)
            node = self._local.require(node_key)
            if isinstance(node, BPlusLeaf):
                break
            node_key = node.children[bisect_right(node.keys, key)][0]
        self._local.path = path

    def _split_node(self, node: BPlusNode) -> tuple[int, int]:
        right_node: BPlusNode
        if isinstance(node, BPlusLeaf):
            right_node = self._new_leaf_node(keys=node.keys[self._mid :], values=node.values[self._mid :])
            node.keys = node.keys[: self._mid]
            node.values = node.values[: self._mid]
        else:
            right_node = self._new_internal_node(
                keys=node.keys[self._mid + 1 :], children=node.children[self._mid + 1 :]
            )
            node.keys = node.keys[: self._mid]
            node.children = node.children[: self._mid + 1]

        self._local.keep(right_node.key, right_node)
        return right_node.key, right_node.leaf

    def _perform_insertion(self) -> None:
        path = self._local.path
        for index in range(len(path) - 1, -1, -1):
            node = self._local.require(path[index])
            if len(node.keys) < self._order:
                return
            insert_key = node.keys[self._mid]
            right_node = self._split_node(node=node)
            if index == 0:
                parent = self._new_internal_node(keys=[insert_key], children=[(node.key, node.leaf), right_node])
                self._local.keep(parent.key, parent)
                self._root: Root[int] | None = (parent.key, parent.leaf)
            else:
                parent_node = self._require_internal(path[index - 1])
                position = bisect_right(parent_node.keys, insert_key)
                parent_node.keys.insert(position, insert_key)
                parent_node.children.insert(position + 1, right_node)

    def _find_path_with_siblings(
        self, key: bytes
    ) -> tuple[dict[int, BPlusNode], list[int], dict[int, BPlusNode], dict[int, int]]:
        path_nodes: dict[int, BPlusNode] = {}
        child_indices: list[int] = []
        siblings: dict[int, BPlusNode] = {}
        sibling_indices: dict[int, int] = {}
        level = 0

        node = self._fetch_root()
        self._local.remove(node.key)
        path_nodes[level] = node

        while isinstance(node, BPlusInternal):
            child_index = bisect_right(node.keys, key)
            child_indices.append(child_index)
            sibling_index = child_index - 1 if child_index > 0 else child_index + 1

            child_node, sibling_node = self._fetch_children(node, [child_index, sibling_index])
            siblings[level + 1] = sibling_node
            sibling_indices[level + 1] = sibling_index
            level += 1
            if len(child_node.keys) > self._min_keys:
                self._return_to_stash([*path_nodes.values(), *siblings.values()])
                path_nodes.clear()
                siblings.clear()
            path_nodes[level] = child_node
            node = child_node

        return path_nodes, child_indices, siblings, sibling_indices

    def _fetch_children(self, parent: BPlusInternal, indices: list[int]) -> list[BPlusNode]:
        requests = [parent.children[index] for index in indices]
        paired = self._round_width(self._op_rounds) >= len(requests)
        for group in [requests] if paired else [[request] for request in requests]:
            self._fetch_nodes(group)
        children: list[BPlusNode] = []
        for index, (child_key, _) in zip(indices, requests, strict=True):
            child = self._local.pop(child_key)
            child.leaf = self._get_new_leaf()
            parent.children[index] = (child_key, child.leaf)
            children.append(child)
        return children

    def _borrow(
        self, parent: BPlusInternal, child_index: int, node: BPlusNode, sibling: BPlusNode, *, from_left: bool
    ) -> None:
        if isinstance(node, BPlusLeaf):
            assert isinstance(sibling, BPlusLeaf)
            if from_left:
                node.keys.insert(0, sibling.keys.pop())
                node.values.insert(0, sibling.values.pop())
                parent.keys[child_index - 1] = node.keys[0]
            else:
                node.keys.append(sibling.keys.pop(0))
                node.values.append(sibling.values.pop(0))
                parent.keys[child_index] = sibling.keys[0]
        else:
            assert isinstance(sibling, BPlusInternal)
            if from_left:
                node.keys.insert(0, parent.keys[child_index - 1])
                node.children.insert(0, sibling.children.pop())
                parent.keys[child_index - 1] = sibling.keys.pop()
            else:
                node.keys.append(parent.keys[child_index])
                node.children.append(sibling.children.pop(0))
                parent.keys[child_index] = sibling.keys.pop(0)

    @staticmethod
    def _merge_into(left: BPlusNode, right: BPlusNode, separator: bytes) -> None:
        if isinstance(left, BPlusLeaf):
            assert isinstance(right, BPlusLeaf)
            left.keys.extend(right.keys)
            left.values.extend(right.values)
        else:
            assert isinstance(right, BPlusInternal)
            left.keys.append(separator)
            left.keys.extend(right.keys)
            left.children.extend(right.children)

    def _apply_delete(
        self,
        key: bytes,
        path_nodes: dict[int, BPlusNode],
        child_indices: list[int],
        siblings: dict[int, BPlusNode],
        sibling_indices: dict[int, int],
    ) -> bytes | None:
        leaf_level = len(child_indices)
        leaf = path_nodes[leaf_level]
        assert isinstance(leaf, BPlusLeaf)
        key_index = next((index for index, each_key in enumerate(leaf.keys) if each_key == key), None)
        if key_index is None:
            return None
        del leaf.keys[key_index]
        deleted_value = leaf.values.pop(key_index)

        if not child_indices:
            if not leaf.keys:
                self._root = None
                del path_nodes[leaf_level]
            return deleted_value

        node_level = leaf_level
        node: BPlusNode = leaf
        for level in range(len(child_indices) - 1, -1, -1):
            if len(node.keys) >= self._min_keys:
                break
            parent = path_nodes[level]
            assert isinstance(parent, BPlusInternal)
            child_index = child_indices[level]
            child_level = level + 1
            sibling = siblings[child_level]
            is_left_sibling = sibling_indices[child_level] < child_index

            if len(sibling.keys) > self._min_keys:
                self._borrow(parent, child_index, node, sibling, from_left=is_left_sibling)
                break

            if is_left_sibling:
                self._merge_into(sibling, node, parent.keys.pop(child_index - 1))
                del parent.children[child_index]
                del path_nodes[node_level]
            else:
                self._merge_into(node, sibling, parent.keys.pop(child_index))
                del parent.children[child_index + 1]
                del siblings[child_level]

            node_level = level
            node = parent

        root_node = path_nodes.get(0)
        if isinstance(root_node, BPlusInternal) and not root_node.keys:
            self._root = root_node.children[0]
            del path_nodes[0]
        return deleted_value

    @override
    def _swap_value(self, node: BPlusNode, key: bytes, value: bytes | None) -> bytes | None:
        assert isinstance(node, BPlusLeaf)
        if key not in node.keys:
            return None
        position = node.keys.index(key)
        found = node.values[position]
        if value is not None:
            node.values[position] = value
        return found

    @override
    def _insert_many(self, pairs: Sequence[tuple[bytes, bytes]]) -> None:
        self._begin_op("insert")
        if not pairs:
            self._finish_op()
            return

        if self._root is None:
            (key, value), *pairs = pairs
            first = self._new_leaf_node(keys=[key], values=[value])
            self._local.add(first.key, first)
            self._root = (first.key, first.leaf)
        else:
            self._descend([key for key, _ in pairs], hold=True)

        for key, value in pairs:
            self._path_to(key)
            leaf = self._require_leaf(self._local.path[-1])
            index = bisect_right(leaf.keys, key)
            leaf.keys.insert(index, key)
            leaf.values.insert(index, value)
            self._perform_insertion()

        self._flush_local_to_stash()
        self._finish_op()

    @override
    def _delete_key(self, key: bytes | None) -> bytes | None:
        self._begin_op("delete")
        if key is None or self._root is None:
            self._finish_op()
            return None

        path_nodes, child_indices, siblings, sibling_indices = self._find_path_with_siblings(key=key)
        deleted_value = self._apply_delete(
            key=key,
            path_nodes=path_nodes,
            child_indices=child_indices,
            siblings=siblings,
            sibling_indices=sibling_indices,
        )

        self._return_to_stash([*path_nodes.values(), *siblings.values()])
        self._finish_op()
        return deleted_value
