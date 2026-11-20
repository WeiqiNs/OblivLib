from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import Any, cast, override

from oblivlib.dependency import Data, KVPair
from oblivlib.dependency.config import ODS_OPS, OdsOp, TreeOmapConfig
from oblivlib.dependency.contract import require_omap_init, require_omap_key, require_value
from oblivlib.dependency.layout import Optional, Record
from oblivlib.dependency.tree_builder import TreeImage
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.omap.base_omap import BaseOmap

ROOT = tuple[Any, int]
KV_LIST = list[tuple[Any, Any]]


class LocalNodesBase[NodeT]:
    def __init__(self) -> None:
        self.nodes: dict[Any, Data] = {}
        self.parent_of: dict[Any, Any] = {}
        self.root_key: Any = None
        self.path: list[Any] = []

    def __len__(self) -> int:
        return len(self.nodes)

    def __bool__(self) -> bool:
        return len(self.nodes) > 0

    def add(self, node: Data, parent_key: Any = None) -> None:
        self.nodes[node.key] = node
        self.parent_of[node.key] = parent_key
        self.path.append(node.key)
        if parent_key is None:
            self.root_key = node.key

    def get(self, key: Any) -> NodeT | None:
        return cast(NodeT | None, self.nodes.get(key))

    def require(self, key: Any) -> NodeT:
        node = self.nodes.get(key)
        if node is None:
            raise KeyError(f"Node {key} is not loaded in local.")
        return cast(NodeT, node)

    def get_parent(self, key: Any) -> NodeT | None:
        parent_key = self.parent_of.get(key)
        return cast(NodeT | None, self.nodes.get(parent_key)) if parent_key is not None else None

    def get_parent_key(self, key: Any) -> Any:
        return self.parent_of.get(key)

    def get_root(self) -> NodeT | None:
        return cast(NodeT | None, self.nodes.get(self.root_key)) if self.root_key is not None else None

    def require_root(self) -> NodeT:
        node = self.nodes.get(self.root_key) if self.root_key is not None else None
        if node is None:
            raise KeyError("The local has no root node.")
        return cast(NodeT, node)

    def to_list(self) -> list[Data]:
        return list(self.nodes.values())

    def remove(self, key: Any) -> Data | None:
        node = self.nodes.pop(key, None)
        if node:
            self.parent_of.pop(key, None)
            if key in self.path:
                self.path.remove(key)
            if self.root_key == key:
                self.root_key = None
        return node

    def pop(self, key: Any) -> NodeT:
        node = self.remove(key)
        if node is None:
            raise KeyError(f"Node {key} is not loaded in local.")
        return cast(NodeT, node)

    def clear(self) -> None:
        self.nodes.clear()
        self.parent_of.clear()
        self.root_key = None
        self.path.clear()


class OstBaseOmap[OmapConfigT: TreeOmapConfig, LocalT: LocalNodesBase[Any]](
    TreeStorageBase[OmapConfigT], BaseOmap, ABC
):
    def __init__(self, config: OmapConfigT):
        super().__init__(config)

        self._root: ROOT | None = None
        self._local: LocalT = self._new_local()
        self._max_height: int = 1
        self._budget: int = 0
        self._op_accesses: int = 0
        self._op_fetches: int = 0
        self._owed: int = 0
        self._op_owed: int = 0

    @abstractmethod
    def _new_local(self) -> LocalT:
        raise NotImplementedError

    @property
    def _key_size(self) -> int:
        return self._config.key_size

    @property
    def _distinguishable(self) -> bool:
        return self._config.distinguishable

    @property
    def root(self) -> ROOT | None:
        return self._root

    @root.setter
    def root(self, root: ROOT | None) -> None:
        self._root = root

    @abstractmethod
    def update_mul_tree_height(self, num_tree: int) -> None:
        raise NotImplementedError

    @cached_property
    def root_layout(self) -> Optional[ROOT]:
        return Optional(Record((self._key_field, self._leaf_field)))

    @property
    def _carry_limit(self) -> int:
        return self._max_height if self._config.cached else 0

    def drain(self) -> None:
        for _ in range(self._carry_limit):
            self._read_paths([self._get_new_leaf()])
            self._write_paths()

    @property
    @override
    def _stash_limit(self) -> int:
        return self._stash_capacity + self._owed

    def _return_to_stash(self, blocks: Sequence[Any]) -> None:
        self._stash += blocks
        self._owed += len(blocks)
        self._check_stash()

    @override
    def _stage_write_back(self) -> None:
        paths = len(set(self._unwritten_leaves))
        super()._stage_write_back()
        self._owed = max(0, self._owed - paths)

    def _flush_local_to_stash(self) -> None:
        self._return_to_stash(self._local.to_list())
        self._local.clear()

    def _return_local(self, keys: Iterable[Any]) -> None:
        self._return_to_stash([self._local.pop(key) for key in list(keys)])

    def _fetch_node(self, key: Any, leaf: int | None, parent_key: Any = None) -> None:
        if self._op_accesses >= self._budget:
            raise ValueError(f"{self._identity}: the op needs more than its {self._budget} rounds.")
        assert leaf is not None
        self._op_accesses += 1
        self._read_paths([leaf])
        node = self._require_in_stash(key)
        self._stash.remove(node)
        self._local.add(node=node, parent_key=parent_key)

    def _short_circuit_read(self, key: Any) -> bool:
        if key is None or self.root is None:
            self._finish_op()
            return True
        return False

    def _begin_op(self, op: OdsOp) -> None:
        config, height = self._config, self._max_height
        if self._distinguishable:
            self._budget = config.budget(op).rounds(height)
        else:
            self._budget = max(config.budget(each).rounds(height) for each in ODS_OPS)
        self._op_accesses = 0
        if not self._carry_limit:
            self._owed = 0

    def _finish_op(self) -> None:
        pending = bool(self._unwritten_leaves)
        self._stage_write_back()
        padding = self._budget - self._op_accesses
        self._op_fetches, self._op_owed = self._op_accesses, self._owed
        if self._owed - padding > self._carry_limit:
            raise ValueError(
                f"{self._identity}: {self._owed} blocks returned to the stash still need an eviction of "
                + f"their own, but only {padding} of the {self._budget} rounds remain and "
                + f"{self._carry_limit} may carry over."
            )
        for _ in range(padding):
            self._op_accesses += 1
            self._read_paths([self._get_new_leaf()])
            self._write_paths()
        if pending and not padding:
            self._client.execute()

    @abstractmethod
    def _build_ods_blocks(self, data: list[KVPair]) -> tuple[list[Data], ROOT]:
        raise NotImplementedError

    @staticmethod
    def _normalize_pairs(data: KV_LIST) -> list[KVPair]:
        return [KVPair(key=pair[0], value=pair[1]) for pair in data]

    def _init_ods_storage(self, data: KV_LIST | None) -> TreeImage:
        blocks: list[Data] = []
        if data:
            blocks, self.root = self._build_ods_blocks(data=self._normalize_pairs(data))
        return self._build_tree(blocks)

    @override
    def init_server_storage(self, data: KV_LIST | None = None) -> None:
        require_omap_init(self._identity, data or [], key_size=self._key_size, data_size=self._data_size)
        self._host_tree(self._init_ods_storage(data=data))

    def _init_mul_tree_ods_storage(self, data_list: list[KV_LIST] | None) -> tuple[TreeImage, list[ROOT | None]]:
        blocks: list[Data] = []
        root_list: list[ROOT | None] = []
        for data in data_list or []:
            if data:
                tree_blocks, root = self._build_ods_blocks(data=self._normalize_pairs(data))
                blocks.extend(tree_blocks)
                root_list.append(root)
            else:
                root_list.append(None)

        return self._build_tree(blocks), root_list

    def init_mul_tree_server_storage(self, data_list: list[KV_LIST] | None = None) -> list[ROOT | None]:
        for data in data_list or []:
            require_omap_init(self._identity, data, key_size=self._key_size, data_size=self._data_size)
        tree, root_list = self._init_mul_tree_ods_storage(data_list=data_list)
        self._host_tree(tree)
        return root_list

    @override
    def search(self, key: bytes | None, value: bytes | None = None) -> bytes | None:
        if key is not None:
            require_omap_key(self._identity, key, self._key_size)
            if value is not None:
                require_value(self._identity, value, self._data_size)
        return self._search(key, value)

    @override
    def insert(self, key: bytes | None, value: bytes | None = None) -> None:
        if key is not None:
            require_omap_key(self._identity, key, self._key_size)
            require_value(self._identity, value, self._data_size)
        self._insert(key, value)

    def delete(self, key: bytes | None) -> bytes | None:
        if key is not None:
            require_omap_key(self._identity, key, self._key_size)
        return self._delete(key)

    @abstractmethod
    def _search(self, key: Any, value: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _insert(self, key: Any, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def _delete(self, key: Any) -> Any:
        raise NotImplementedError
