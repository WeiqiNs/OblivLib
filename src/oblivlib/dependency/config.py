"""Construction-parameter configs for the ORAM and OMAP schemes.

Defaults live here — one place per scheme — instead of being repeated across every constructor
signature. The config hierarchy mirrors the scheme hierarchy: shared defaults are inherited and
scheme-specific defaults override them. All configs are frozen and keyword-only; derive a variant
with `dataclasses.replace(cfg, num_data=...)`.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from oblivlib.dependency.client import Client
    from oblivlib.dependency.crypto import Encryptor

ResetMethod = Literal["prob", "hard"]


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
    name: str = field(default_factory=unique_name("so"))


@dataclass(frozen=True, kw_only=True)
class MulPathOramConfig(OramConfig):
    """``stash_scale_multiplier`` multiplies ``stash_scale`` to absorb a batch of paths per access."""

    name: str = field(default_factory=unique_name("mul_path_oram"))
    stash_scale_multiplier: int = 1


@dataclass(frozen=True, kw_only=True)
class RecursiveOramConfig(OramConfig):
    """``on_chip_mem``: target size of the position map kept on the client; ``num_data`` must exceed it.
    ``compression_ratio``: child leaves stored per position-map block."""

    name: str = field(default_factory=unique_name("rc"))
    on_chip_mem: int = 10
    compression_ratio: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_chip_mem < 1:
            raise ValueError(f"on_chip_mem must be >= 1, got {self.on_chip_mem}.")
        if self.compression_ratio < 2:
            raise ValueError(f"compression_ratio must be >= 2, got {self.compression_ratio}.")


@dataclass(frozen=True, kw_only=True)
class CounterOramConfig(OramConfig):
    """Shared structure for the counter-based recursive schemes (DA, Freecursive).

    ``num_ic``: children per position-map block. ``ic_length`` / ``gc_length``: bit widths of the
    individual and group counters. ``prf_key``: key of the leaf-deriving PRF; random when ``None``.
    """

    num_ic: int = 64
    ic_length: int = 6
    gc_length: int = 64
    prf_key: bytes | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_ic < 1:
            raise ValueError(f"num_ic must be >= 1, got {self.num_ic}.")
        if self.ic_length < 1:
            raise ValueError(f"ic_length must be >= 1, got {self.ic_length}.")
        if self.gc_length < 1:
            raise ValueError(f"gc_length must be >= 1, got {self.gc_length}.")


@dataclass(frozen=True, kw_only=True)
class DaOramConfig(CounterOramConfig):
    """``on_chip_mem``: target size of the position map kept on the client; ``num_data`` must exceed it."""

    name: str = field(default_factory=unique_name("da"))
    on_chip_mem: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_chip_mem < 1:
            raise ValueError(f"on_chip_mem must be >= 1, got {self.on_chip_mem}.")


@dataclass(frozen=True, kw_only=True)
class FreecursiveOramConfig(CounterOramConfig):
    """``on_chip_size``: target size of the position map kept on the client; ``num_data`` must exceed it.
    ``reset_method``: ``"prob"`` (secure) resets a block with probability ``reset_prob`` (default
    ``1 / num_ic``) and treats a counter overflow as an error; ``"hard"`` (the insecure original scheme)
    resets only on overflow."""

    name: str = field(default_factory=unique_name("fc"))
    num_ic: int = 48
    ic_length: int = 10
    gc_length: int = 32
    on_chip_size: int = 10
    reset_method: ResetMethod = "prob"
    reset_prob: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_chip_size < 1:
            raise ValueError(f"on_chip_size must be >= 1, got {self.on_chip_size}.")
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


@dataclass(frozen=True, kw_only=True)
class AvlOmapConfig(OmapConfig):
    name: str = field(default_factory=unique_name("avl"))


@dataclass(frozen=True, kw_only=True)
class AvlOmapCachedConfig(AvlOmapConfig):
    name: str = field(default_factory=unique_name("avl_opt"))


@dataclass(frozen=True, kw_only=True)
class BPlusOmapConfig(OmapConfig):
    """``order``: maximum children per internal node (at least 3)."""

    name: str = field(default_factory=unique_name("bplus"))
    order: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.order < 3:
            raise ValueError(f"order must be >= 3, got {self.order}.")


@dataclass(frozen=True, kw_only=True)
class BPlusOmapCachedConfig(BPlusOmapConfig):
    name: str = field(default_factory=unique_name("bplus_opt"))


@dataclass(frozen=True, kw_only=True)
class GroupOmapConfig(OmapConfig):
    name: str = field(default_factory=unique_name("group_omap"))


@dataclass(frozen=True, kw_only=True)
class OramOstOmapConfig:
    """Construction parameters for OramOstOmap (the ODS-over-ORAM framework). The two sub-schemes (the
    ODS and the ORAM, each with its own config) are objects passed to the constructor; only the data
    count lives here -- hence this is a standalone config rather than an OmapConfig subclass."""

    num_data: int

    def __post_init__(self) -> None:
        if self.num_data < 1:
            raise ValueError(f"num_data must be >= 1, got {self.num_data}.")
