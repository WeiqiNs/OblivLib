import math
from collections.abc import Callable, Iterable
from decimal import Decimal, localcontext
from typing import override

from oblivlib.dependency.config import GroveConfig
from oblivlib.dependency.keyed_store import KeyedStore
from oblivlib.dependency.types import KeyedBlock, PathData


def overflow_probability(*, insertions: int, group: int, bucket_size: int, level: int, precision: int = 80) -> Decimal:
    with localcontext() as context:
        context.prec = precision
        total = Decimal(0)
        for depth in range(max(1, math.ceil(math.log2(bucket_size + 1))), level):
            arrivals = 1 << depth
            stay = Decimal(1) / (Decimal(2) ** (depth + 1))
            within = sum(
                (Decimal(math.comb(arrivals, kept)) * stay**kept * (1 - stay) ** (arrivals - kept))
                for kept in range(bucket_size // group + 1)
            )
            intervals = Decimal(math.ceil(insertions / arrivals))
            total += Decimal(arrivals) * intervals * (1 - within)
        return total


def meta_bucket_size(*, insertions: int, group: int, level: int, security: int = 128) -> int:
    bound = Decimal(2) ** (-security)
    bucket_size = group
    while overflow_probability(insertions=insertions, group=group, bucket_size=bucket_size, level=level) > bound:
        bucket_size += 1
    return bucket_size


class DupStore(KeyedStore[GroveConfig]):
    def __init__(self, config: GroveConfig, *, key_size: int):
        super().__init__(config, key_size=key_size)
        self._counter: int = 0

    def rl_leaves(self, count: int) -> list[int]:
        bits = self._level - 1
        leaves: list[int] = []
        for _ in range(count):
            leaves.append(int(format(self._counter, f"0{bits}b")[::-1], 2))
            self._counter = (self._counter + 1) % self.leaf_range
        return leaves

    def read_rl(self, count: int) -> None:
        self._read_paths(self.rl_leaves(count))

    @override
    def add(self, blocks: Iterable[KeyedBlock]) -> None:
        fresh = list(blocks)
        keys = {block.key for block in fresh}
        self._stash[:] = [*fresh, *(block for block in self._stash if block.key not in keys)]
        self._check_stash()

    @override
    def _absorb(self, path: PathData[KeyedBlock]) -> None:
        held = {block.key for block in self._stash}
        for index in sorted(path):
            for block in path[index]:
                if block.key not in held:
                    held.add(block.key)
                    self._stash.append(block)
        self._check_stash()

    def take_at(self, leaf: int, wanted: Callable[[bytes], bool]) -> list[KeyedBlock]:
        taken = [block for block in self._stash if block.leaf == leaf and wanted(block.key)]
        if taken:
            gone = {id(block) for block in taken}
            self._stash[:] = [block for block in self._stash if id(block) not in gone]
        return taken
