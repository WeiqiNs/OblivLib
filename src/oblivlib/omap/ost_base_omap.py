from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cached_property
from typing import override

from oblivlib.dependency.config import ODS_OPS, OdsOp, TreeOmapConfig
from oblivlib.dependency.contract import (
    require_batch,
    require_omap_init,
    require_omap_key,
    require_omap_pair,
    require_room,
)
from oblivlib.dependency.layout import Field, Optional, Pair
from oblivlib.dependency.tree_storage_base import TreeStorageBase, leaf_field
from oblivlib.dependency.types import Block
from oblivlib.omap.base_omap import BaseOmap, KvList

type Root[K] = tuple[K, int]


class LocalNodes[KeyT, NodeT]:
    def __init__(self) -> None:
        self.nodes: dict[KeyT, NodeT] = {}
        self.path: list[KeyT] = []

    def __len__(self) -> int:
        return len(self.nodes)

    def add(self, key: KeyT, node: NodeT) -> None:
        self.nodes[key] = node
        self.path.append(key)

    def keep(self, key: KeyT, node: NodeT) -> None:
        self.nodes[key] = node

    def get(self, key: KeyT) -> NodeT | None:
        return self.nodes.get(key)

    def require(self, key: KeyT) -> NodeT:
        return self.nodes[key]

    def to_list(self) -> list[NodeT]:
        return list(self.nodes.values())

    def remove(self, key: KeyT) -> None:
        del self.nodes[key]
        self.path.remove(key)

    def pop(self, key: KeyT) -> NodeT:
        node = self.require(key)
        self.remove(key)
        return node

    def clear(self) -> None:
        self.nodes.clear()
        self.path.clear()


class OstBaseOmap[OmapConfigT: TreeOmapConfig, KeyT, NodeT: Block[object]](
    TreeStorageBase[OmapConfigT, NodeT], BaseOmap, ABC
):
    def __init__(self, config: OmapConfigT, *, height_keys: int | None = None):
        super().__init__(config)

        self._root: Root[KeyT] | None = None
        self._max_height: int = self._height_bound(self.num_data if height_keys is None else height_keys)
        self._budget: int = 0
        self._op_rounds: int = 0
        self._op_fetches: int = 0
        self._owed: int = 0
        self._op_owed: int = 0
        self._size: int = 0

    @property
    @abstractmethod
    def _local(self) -> LocalNodes[KeyT, NodeT]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _key_field(cls, config: OmapConfigT) -> Field[KeyT]:
        raise NotImplementedError

    @classmethod
    def root_layout(cls, config: OmapConfigT) -> Optional[Root[KeyT]]:
        return Optional(Pair(cls._key_field(config), leaf_field(config.num_data)))

    @property
    def root(self) -> Root[KeyT] | None:
        return self._root

    def on_root[T](self, root: Root[KeyT] | None, op: Callable[[], T]) -> tuple[Root[KeyT] | None, T]:
        self._root = root
        result = op()
        return self._root, result

    @property
    @abstractmethod
    def _fanout(self) -> int:
        raise NotImplementedError

    @property
    def key_size(self) -> int:
        return self._config.key_size

    @property
    def _distinguishable(self) -> bool:
        return self._config.distinguishable

    @abstractmethod
    def _height_bound(self, num_nodes: int) -> int:
        raise NotImplementedError

    @property
    def _batch_size(self) -> int:
        return self._config.batch_size

    @property
    def _widest_round(self) -> int:
        return min(self._config.paths_per_round * self._batch_size, self.leaf_range)

    @property
    def _carry_limit(self) -> int:
        if not self._config.cached:
            return 0
        height, cached, uncached = (
            self._max_height,
            self._config.minimums(cached=True),
            self._config.minimums(cached=False),
        )
        return max(
            sum(self._round_width(index) for index in range(cached[op].rounds(height), uncached[op].rounds(height)))
            for op in ODS_OPS
        )

    def drain(self) -> None:
        for _ in range(-(-self._carry_limit // self._widest_round)):
            self._read_paths(self.padded_leaves([], self._widest_round))
            self._write_paths()

    @property
    @override
    def _stash_limit(self) -> int:
        return self._stash_capacity * self._widest_round + self._owed

    def _return_to_stash(self, blocks: Sequence[NodeT]) -> None:
        self._stash.extend(blocks)
        self._owed += len(blocks)
        self._check_stash()

    @override
    def _stage_write_back(self) -> None:
        paths = len(self._unwritten_leaves)
        super()._stage_write_back()
        self._owed = max(0, self._owed - paths)

    def _flush_local_to_stash(self) -> None:
        self._return_to_stash(self._local.to_list())
        self._local.clear()

    def _return_local(self, keys: Iterable[KeyT]) -> None:
        self._return_to_stash([self._local.pop(key) for key in list(keys)])

    @cached_property
    def _descent_keys(self) -> list[int]:
        keys = [1]
        for _ in range(1, self._max_height):
            keys.append(min(keys[-1] * self._fanout, self._batch_size))
        return keys

    def _round_width(self, index: int) -> int:
        keys = self._descent_keys[index] if index < self._max_height else self._batch_size
        return min(self._config.paths_per_round * keys, self.leaf_range)

    def _start_round(self, leaves: Iterable[int]) -> None:
        assert self._op_rounds < self._budget, "an op needed more rounds than its proven budget"
        self._read_paths(self.padded_leaves(leaves, self._round_width(self._op_rounds)))
        self._op_rounds += 1

    def _fetch_nodes(self, requests: Sequence[tuple[KeyT, int]]) -> None:
        self._start_round([leaf for _, leaf in requests])
        for key, _ in requests:
            node = self._require_in_stash(key)
            self._stash.remove(node)
            self._local.add(key, node)

    def _fetch_node(self, key: KeyT, leaf: int) -> None:
        self._fetch_nodes([(key, leaf)])

    def _fetch_root(self) -> NodeT:
        assert self._root is not None
        key, leaf = self._root
        self._fetch_node(key=key, leaf=leaf)
        node = self._local.require(key)
        node.leaf = self._get_new_leaf()
        self._root = (key, node.leaf)
        return node

    def _fetch_rehomed(self, requests: Mapping[KeyT, tuple[int, int]]) -> None:
        self._fetch_nodes([(child_key, leaf) for child_key, (leaf, _) in requests.items()])
        for child_key, (_, new_leaf) in requests.items():
            self._local.require(child_key).leaf = new_leaf

    def _begin_op(self, op: OdsOp) -> None:
        config, height = self._config, self._max_height
        if self._distinguishable:
            self._budget = config.budget(op).rounds(height)
        else:
            self._budget = max(config.budget(each).rounds(height) for each in ODS_OPS)
        self._op_rounds = 0

    def _finish_op(self) -> None:
        pending = bool(self._unwritten_leaves)
        self._stage_write_back()
        padding = range(self._op_rounds, self._budget)
        capacity = sum(self._round_width(index) for index in padding)
        self._op_fetches, self._op_owed = self._op_rounds, self._owed
        assert self._owed - capacity <= self._carry_limit, "an op left more evictions owed than its budget allows"
        for _ in padding:
            self._start_round([])
            self._write_paths()
        if pending and not padding:
            self._client.send_writes()

    @abstractmethod
    def _build_ods_blocks(self, data: KvList) -> tuple[list[NodeT], Root[KeyT]]:
        raise NotImplementedError

    @override
    def init_server_storage(self, data: KvList | None = None) -> None:
        self._root = self.init_mul_tree_server_storage([data or []])[0]

    def init_mul_tree_server_storage(self, data_list: list[KvList] | None = None) -> list[Root[KeyT] | None]:
        blocks: list[NodeT] = []
        root_list: list[Root[KeyT] | None] = []
        every_pair = [pair for data in data_list or [] for pair in data]
        require_omap_init(self._identity, every_pair, key_size=self.key_size, data_size=self.data_size)
        size = len(every_pair)
        require_room(self._identity, held=0, fresh=size, capacity=self.num_data)
        for data in data_list or []:
            root: Root[KeyT] | None = None
            if data:
                tree_blocks, root = self._build_ods_blocks(data)
                blocks.extend(tree_blocks)
            root_list.append(root)
        self._host_tree(self._build_tree(blocks))
        self._size = size
        return root_list

    def _require_pair(self, key: bytes | None, value: bytes | None, *, value_required: bool) -> None:
        if key is not None:
            require_omap_pair(
                self._identity,
                key,
                value,
                key_size=self.key_size,
                data_size=self.data_size,
                value_required=value_required,
            )

    @override
    def search(self, key: bytes | None, value: bytes | None = None) -> bytes | None:
        self._require_pair(key, value, value_required=False)
        return self._search_many([(key, value)])[0]

    @override
    def insert(self, key: bytes | None, value: bytes | None = None) -> None:
        self._require_pair(key, value, value_required=True)
        self._insert_counted([] if key is None or value is None else [(key, value)])

    def search_batch(self, pairs: Sequence[tuple[bytes, bytes | None]]) -> list[bytes | None]:
        require_batch(self._identity, [key for key, _ in pairs], self._batch_size, distinct=False)
        for key, value in pairs:
            self._require_pair(key, value, value_required=False)
        return self._search_many(pairs)

    def insert_batch(self, pairs: Sequence[tuple[bytes, bytes]]) -> None:
        require_batch(self._identity, [key for key, _ in pairs], self._batch_size, distinct=True)
        for key, value in pairs:
            self._require_pair(key, value, value_required=True)
        self._insert_counted(pairs)

    def _search_many(self, pairs: Sequence[tuple[bytes | None, bytes | None]]) -> list[bytes | None]:
        self._begin_op("search")
        results: list[bytes | None] = [None] * len(pairs)
        lanes = [(lane, key, value) for lane, (key, value) in enumerate(pairs) if key is not None]
        if not lanes or self._root is None:
            self._finish_op()
            return results

        def on_stop(index: int, node: NodeT) -> None:
            lane, key, value = lanes[index]
            results[lane] = self._swap_value(node, key, value)

        self._descend([key for _, key, _ in lanes], hold=False, on_stop=on_stop)
        self._flush_local_to_stash()
        self._finish_op()
        return results

    @abstractmethod
    def _descend(
        self, keys: Sequence[bytes], *, hold: bool, on_stop: Callable[[int, NodeT], None] | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _swap_value(self, node: NodeT, key: bytes, value: bytes | None) -> bytes | None:
        raise NotImplementedError

    def require_room(self, fresh: int) -> None:
        require_room(self._identity, held=self._size, fresh=fresh, capacity=self.num_data)

    def _insert_counted(self, pairs: Sequence[tuple[bytes, bytes]]) -> None:
        self.require_room(len(pairs))
        self._insert_many(pairs)
        self._size += len(pairs)

    @abstractmethod
    def _insert_many(self, pairs: Sequence[tuple[bytes, bytes]]) -> None:
        raise NotImplementedError

    @override
    def delete(self, key: bytes | None) -> bytes | None:
        if key is not None:
            require_omap_key(self._identity, key, self.key_size)
        deleted = self._delete_key(key)
        if deleted is not None:
            self._size -= 1
        return deleted

    @abstractmethod
    def _delete_key(self, key: bytes | None) -> bytes | None:
        raise NotImplementedError
