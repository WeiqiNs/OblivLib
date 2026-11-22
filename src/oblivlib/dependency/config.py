"""Construction-parameter configs for the ORAM and OMAP schemes.

Defaults live here — one place per scheme — instead of being repeated across every constructor
signature. The config hierarchy mirrors the scheme hierarchy: shared defaults are inherited and
scheme-specific defaults override them. All configs are frozen and keyword-only; derive a variant
with `dataclasses.replace(cfg, num_data=...)`.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, get_args, override

from oblivlib.dependency.heap_index import leaf_count

if TYPE_CHECKING:
    from oblivlib.dependency.client import Client
    from oblivlib.dependency.crypto import Encryptor

ResetMethod = Literal["prob", "hard"]
OdsOp = Literal["search", "insert", "delete"]
ODS_OPS: tuple[OdsOp, ...] = ("search", "insert", "delete")

MIN_GC_LENGTH = 64


def unique_name(prefix: str) -> Callable[[], str]:
    return lambda: f"{prefix}-{secrets.randbelow(1 << 32):08x}"


@dataclass(frozen=True, kw_only=True)
class OramConfig:
    """Shared construction parameters for every tree-based ORAM.

    - ``num_data``: number of logical blocks, keys ``0 .. num_data - 1``. The tree gets the smallest
      power-of-two leaf count that covers it.
    - ``data_size``: the most bytes a value may hold; a longer value raises ``ContractError``. Sealed rows
      are sized for a full bucket of blocks with values this long.
    - ``client``: the ``Client`` handle to server storage.
    - ``name``: this scheme's storage label on ``client``. Defaults to the scheme's prefix plus a random
      suffix, so schemes sharing a client never collide; pass one only to address storage by a known label.
    - ``build_file``: build the initial tree on the client in this file instead of memory, for trees
      larger than RAM. Requires an ``encryptor`` (the file holds only sealed rows); it is handed to the
      server and gone from this path once hosted, so position-map children reuse it in turn.
    - ``bucket_size``: blocks per tree node.
    - ``stash_scale``: the stash holds at most ``stash_scale * max(1, level - 1)`` blocks; exceeding it
      raises ``StashOverflowError``.
    - ``encryptor``: seals each bucket into one ciphertext; ``None`` stores plaintext.
    """

    num_data: int
    data_size: int
    client: Client
    name: str = field(default_factory=unique_name("oram"))
    build_file: str | Path | None = None
    bucket_size: int = 4
    stash_scale: int = 7
    encryptor: Encryptor | None = None

    def __post_init__(self) -> None:
        if self.num_data < 1:
            raise ValueError(f"num_data must be >= 1, got {self.num_data}.")
        if self.data_size < 1:
            raise ValueError(f"data_size must be >= 1, got {self.data_size}.")
        if self.bucket_size < 1:
            raise ValueError(f"bucket_size must be >= 1, got {self.bucket_size}.")
        if self.stash_scale < 1:
            raise ValueError(f"stash_scale must be >= 1, got {self.stash_scale}.")
        if self.build_file is not None and self.encryptor is None:
            raise ValueError("build_file requires an encryptor; a build file never holds plaintext.")


@dataclass(frozen=True, kw_only=True)
class PathOramConfig(OramConfig):
    name: str = field(default_factory=unique_name("po"))


@dataclass(frozen=True, kw_only=True)
class StaticOramConfig(OramConfig):
    """``prf_key``: key of the PRF that fixes each key's leaf; random when ``None``."""

    name: str = field(default_factory=unique_name("so"))
    prf_key: bytes | None = None


@dataclass(frozen=True, kw_only=True)
class MulPathOramConfig(OramConfig):
    name: str = field(default_factory=unique_name("mul_path_oram"))


@dataclass(frozen=True, kw_only=True)
class PosMapOramConfig(OramConfig):
    """Shared structure for the schemes with a recursive position map.

    ``on_chip_mem``: target size of the position map kept on the client; ``num_data`` must exceed it.
    """

    on_chip_mem: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_chip_mem < 1:
            raise ValueError(f"on_chip_mem must be >= 1, got {self.on_chip_mem}.")


@dataclass(frozen=True, kw_only=True)
class RecursiveOramConfig(PosMapOramConfig):
    """``compression_ratio``: child leaves stored per position-map block."""

    name: str = field(default_factory=unique_name("rc"))
    compression_ratio: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.compression_ratio < 2:
            raise ValueError(f"compression_ratio must be >= 2, got {self.compression_ratio}.")


@dataclass(frozen=True, kw_only=True)
class CounterOramConfig(PosMapOramConfig):
    """Shared structure for the counter-based recursive schemes (DA, Freecursive).

    ``num_ic``: children per position-map block. ``ic_length`` / ``gc_length``: bit widths of the
    individual and group counters. ``prf_key``: key of the leaf-deriving PRF; random when ``None``.
    """

    num_ic: int = 64
    ic_length: int = 6
    gc_length: int = 64
    prf_key: bytes | None = None

    @property
    def counter_widths(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def count_length(self) -> int:
        return sum(self.counter_widths)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_ic < 2:
            raise ValueError(f"num_ic must be >= 2, got {self.num_ic}.")
        if self.ic_length < 1:
            raise ValueError(f"ic_length must be >= 1, got {self.ic_length}.")
        if self.gc_length < MIN_GC_LENGTH:
            raise ValueError(f"gc_length must be >= {MIN_GC_LENGTH}, got {self.gc_length}.")


@dataclass(frozen=True, kw_only=True)
class DaOramConfig(CounterOramConfig):
    name: str = field(default_factory=unique_name("da"))

    @property
    @override
    def counter_widths(self) -> tuple[int, ...]:
        return (self.gc_length, *[self.ic_length] * self.num_ic, *[1] * self.num_ic)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_ic > 2**self.ic_length:
            raise ValueError(
                f"num_ic ({self.num_ic}) must be at most 2**ic_length ({2**self.ic_length}) so every pending reset "
                + "clears before the block can overflow again."
            )


@dataclass(frozen=True, kw_only=True)
class FreecursiveOramConfig(CounterOramConfig):
    """``reset_method``: ``"prob"`` (secure) resets a block with probability ``reset_prob`` (default
    ``1 / num_ic``) and treats a counter overflow as an error; ``"hard"`` (the insecure original scheme)
    resets only on overflow."""

    name: str = field(default_factory=unique_name("fc"))
    num_ic: int = 48
    ic_length: int = 10
    reset_method: ResetMethod = "prob"
    reset_prob: float | None = None

    @property
    @override
    def counter_widths(self) -> tuple[int, ...]:
        return (self.gc_length, *[self.ic_length] * self.num_ic)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reset_method not in get_args(ResetMethod):
            raise ValueError(f"reset_method must be one of {get_args(ResetMethod)}, got {self.reset_method!r}.")
        if self.reset_prob is not None and not 0.0 < self.reset_prob <= 1.0:
            raise ValueError(f"reset_prob must be in (0, 1], got {self.reset_prob}.")


@dataclass(frozen=True, kw_only=True)
class OmapConfig(OramConfig):
    """Base config for oblivious maps.

    ``key_size``: byte length that bounds a key, used to size node blocks like ``data_size``.
    ``distinguishable``: when ``True`` each op pads to its own round bound, so the op type leaks;
    the default ``False`` pads every op to the largest bound.
    """

    name: str = field(default_factory=unique_name("omap"))
    key_size: int
    distinguishable: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.key_size < 1:
            raise ValueError(f"key_size must be >= 1, got {self.key_size}.")


@dataclass(frozen=True)
class AccessBudget:
    a: Fraction | int
    b: Fraction | int

    def rounds(self, height: int) -> int:
        return math.floor(self.a * height + self.b)


@dataclass(frozen=True, kw_only=True)
class TreeOmapConfig(OmapConfig):
    MIN_BUDGETS: ClassVar[dict[OdsOp, AccessBudget]] = {}
    MIN_CACHED_BUDGETS: ClassVar[dict[OdsOp, AccessBudget]] = {}
    PATHS_PER_ROUND: ClassVar[tuple[int, ...]] = (1,)

    cached: bool = False
    batch_size: int = 1
    paths_per_round: int = 1
    search_budget: AccessBudget | None = None
    insert_budget: AccessBudget | None = None
    delete_budget: AccessBudget | None = None

    def minimums(self, *, cached: bool) -> dict[OdsOp, AccessBudget]:
        return self.MIN_CACHED_BUDGETS if cached else self.MIN_BUDGETS

    def budget(self, op: OdsOp) -> AccessBudget:
        match op:
            case "search":
                given = self.search_budget
            case "insert":
                given = self.insert_budget
            case "delete":
                given = self.delete_budget
        return given or self.minimums(cached=self.cached)[op]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.paths_per_round not in self.PATHS_PER_ROUND:
            allowed = " or ".join(str(width) for width in self.PATHS_PER_ROUND)
            raise ValueError(f"paths_per_round must be {allowed}, got {self.paths_per_round}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        leaf_range = leaf_count(self.num_data)
        if self.batch_size > leaf_range:
            raise ValueError(f"batch_size {self.batch_size} exceeds the {leaf_range} leaves of the tree.")
        for op in ODS_OPS:
            given, least = self.budget(op), self.minimums(cached=self.cached)[op]
            if given.a < least.a or given.a + given.b < least.a + least.b:
                raise ValueError(
                    f"{op}_budget {given} is below the proven minimum {least}: a must be >= {least.a} and "
                    + f"a + b >= {least.a + least.b}."
                )


@dataclass(frozen=True, kw_only=True)
class AvlOmapConfig(TreeOmapConfig):
    MIN_BUDGETS: ClassVar[dict[OdsOp, AccessBudget]] = {
        "search": AccessBudget(1, 0),
        "insert": AccessBudget(2, 0),
        "delete": AccessBudget(2, 0),
    }
    MIN_CACHED_BUDGETS: ClassVar[dict[OdsOp, AccessBudget]] = {
        "search": AccessBudget(1, 0),
        "insert": AccessBudget(1, 1),
        "delete": AccessBudget(Fraction(3, 2), Fraction(-1, 2)),
    }

    name: str = field(default_factory=unique_name("avl"))


@dataclass(frozen=True, kw_only=True)
class BPlusOmapConfig(TreeOmapConfig):
    """``order``: maximum children per internal node (at least 3)."""

    MIN_BUDGETS: ClassVar[dict[OdsOp, AccessBudget]] = {
        "search": AccessBudget(1, 0),
        "insert": AccessBudget(3, -2),
        "delete": AccessBudget(4, -3),
    }
    MIN_TWO_PATH_BUDGETS: ClassVar[dict[OdsOp, AccessBudget]] = {
        "search": AccessBudget(1, 0),
        "insert": AccessBudget(2, -1),
        "delete": AccessBudget(2, -1),
    }
    MIN_CACHED_BUDGETS: ClassVar[dict[OdsOp, AccessBudget]] = {
        "search": AccessBudget(1, 0),
        "insert": AccessBudget(2, -1),
        "delete": AccessBudget(2, -1),
    }
    MIN_CACHED_TWO_PATH_BUDGETS: ClassVar[dict[OdsOp, AccessBudget]] = {
        "search": AccessBudget(1, 0),
        "insert": AccessBudget(1, 0),
        "delete": AccessBudget(1, 0),
    }
    PATHS_PER_ROUND: ClassVar[tuple[int, ...]] = (1, 2)

    name: str = field(default_factory=unique_name("bplus"))
    order: int
    paths_per_round: int = 2

    @override
    def minimums(self, *, cached: bool) -> dict[OdsOp, AccessBudget]:
        two_path = self.paths_per_round == 2
        uncached = self.MIN_TWO_PATH_BUDGETS if two_path else self.MIN_BUDGETS
        if not cached:
            return uncached
        cached_budgets = self.MIN_CACHED_TWO_PATH_BUDGETS if two_path else self.MIN_CACHED_BUDGETS
        if self.batch_size == 1 or (two_path and self.batch_size <= self.order + 1):
            return cached_budgets
        return {**uncached, "delete": cached_budgets["delete"]}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.order < 3:
            raise ValueError(f"order must be >= 3, got {self.order}.")


@dataclass(frozen=True, kw_only=True)
class GroupOmapConfig(OmapConfig):
    name: str = field(default_factory=unique_name("group_omap"))
