# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Overview

`oblivlib` implements oblivious RAM (ORAM) and oblivious maps (OMAP) for a client/server setting: the client is trusted, the server stores encrypted rows and answers batched path reads and writes. It backs the maintainers' papers (notably "Towards Practical Oblivious Map", VLDB 2025); `README.md` maps each paper to its file.

## Commands

- Install: `pip install -e ".[dev]"` (pytest, basedpyright, ruff; everything is in `pyproject.toml`).
- Test: `pytest`, or one file / one test by path. `NUM_DATA=256 pytest ...` shrinks the `num_data` fixture (default `2**12`, `tests/conftest.py`).
- Types: `basedpyright` (`pyrightconfig.json`, `recommended` mode with the dynamic-typing rules off because internal blocks are loosely typed). Point it at the env with the deps (`--pythonpath`), or every import fails.
- Lint: `ruff check`, `ruff format --check`.
- CI (`.github/workflows/ci.yml`) runs all four on Python 3.12 and 3.x.

## Architecture

**Client/server.** Schemes do all I/O through a `Client` (`dependency/client.py`) passed in their config. They stage `add_read_path` / `add_write_path` per storage `label` and call `execute()`, which sends one `Batch` whose writes apply before its reads. With `defer_writes` (default) a write-only `execute()` sends nothing and its writes ride on the next batch; `flush()` / `close()` send them. `client.metrics` counts rounds and bytes. Below the client: a `Backend` (`LocalBackend` in process, or `TransportBackend` over a `Transport` such as `ZmqTransport`) carries `protocol.py` messages to a `StorageServer`, which keeps each tree in a fixed-row store (memory or file) or, for plaintext debugging, a variable-row memory store (`server_stores.py`). `Client.local()` and `Client.connect(endpoint)` build the two usual stacks; `demo/` runs both ends over ZeroMQ.

**Storage.** The public contract is bytes: ORAM keys are ints in `[0, num_data)`, OMAP keys are bytes of at most `key_size`, values bytes of at most `data_size`; `contract.py` checks them at every public entry point before any state changes. Every stored record has a fixed width built from `layout.py` fields, so every row has exactly the declared size (`CapacityError` if a value does not fit); `PathCipher` encodes and, with an `encryptor`, seals each bucket into one row. Nothing is pickled. `TreeStorageBase` (`tree_storage_base.py`) is shared by every tree: config accessors, leaf math, stash, path read/write, and building the initial tree (`tree_builder.py`) that `Client.host_tree` hands to the server. `_write_paths()` writes back exactly the paths last read, and a pending write-back is staged before the next read, so no block is ever held twice.

**ORAMs** (`oram/`). Each scheme takes one frozen, keyword-only config (`dependency/config.py`, validated in `__post_init__`). `TreeBaseOram` gives one access template: a scheme implements `_operate_on_key_without_eviction`; `operate_on_key(key, value=UNSET)` returns the current value and writes `value` if given, and `update(key, change)` reads the value, runs `change(value) -> (new_value, result)` (which may do other work, such as a whole ODS op), stores `new_value`, then evicts; the composed maps build on `update`. `MulPathOram.operate_on_keys` does one access for many keys. Schemes: `PathOram`, `RecursivePathOram`, `StaticOram`, `MulPathOram`, `DAOram`, `FreecursiveOram` (secure with `reset_method="prob"`, insecure with `"hard"`). DA and Freecursive share their PRF-derived leaves, position-map levels and on-chip counters through `CounterOram` (`counter_oram.py`).

**OMAPs** (`omap/`). `AVLOmap` and `BPlusOmap` are oblivious data structures on `OstBaseOmap`: `search(key, value=None)` (a value updates in place), `insert(key, value)` (the key must be new), `delete(key)`, and `search_batch` / `insert_batch`. Composed maps: `OramOstOmap` (VLDB 2025) hashes keys into any ORAM whose slots hold the roots of small trees of any ODS map. `GroupOmap` hashes each key into one of `num_data` buckets; an upper ORAM stores each bucket's `(seed, count)`, and the bucket's blocks sit in a lower tree (a plain `TreeStorageBase`, no position map) on paths `PRF(seed || slot)` for slots `1..count`, each block carrying its own key. Search and delete read the bucket's slot paths padded with random ones to `upper_bound` paths, then re-home the bucket under a fresh seed (delete renumbers the remaining slots). Insert writes slot `count + 1` through one random path when `distinguishable=True`; otherwise it takes the search shape, so every op reads the same `upper_bound` lower paths (capped at the leaf count). `balls_into_bins.max_bin_load` bounds how many keys one slot or bucket receives (`upper_bound`). Every map refuses an insert that would exceed `num_data` keys before touching storage, since every height and bucket bound assumes it.

**Crypto** (`crypto.py`): `Encryptor` / `AesGcm` (exact `ciphertext_length`), `Blake2Prf`, and the hashing helpers the composed maps use. `encryptor=None` stores plaintext rows (memory servers only). The model is honest-but-curious; GCM is defense in depth.

## ODS budgets

Every ODS op is oblivious on its own: it runs exactly its budget of rounds, `floor(a*h + b)` (`AccessBudget`, `a`/`b` may be fractions), where `h` bounds the tree height (`avl_max_height`, `bplus_max_levels`). With `distinguishable=False` (default) every op pads to the largest budget, so op types look alike. A round reads a fixed number of distinct random-looking paths and writes their union back. Round `r` reads `paths_per_round * min(fanout^r, k)` paths for `r < h` and `paths_per_round * k` after (`_round_width`; fanout 2 for AVL, `order` for B+; `k = batch_size`; capped at the leaf count). The width depends only on the round index, never on data. Config knobs: `cached`, `batch_size`, `paths_per_round` (B+ 1 or 2, default 2; AVL 1), and per-op budget overrides, which may only raise the minimums. `textbook()` reproduces the original implementation's counts at one path per round.

**Ledger.** Path ORAM's stash bound needs one eviction per block put back into the stash, and evicting into a union of `w` paths counts as `w`. `_return_to_stash` adds each returned block to `_owed`; a write-back of `w` paths pays `min(_owed, w)`; `_finish_op` pads to the budget and raises if more than the carry limit is still owed. The stash may hold `stash_capacity * widest round + _owed`. Every written-back path either carries a returned block or is wasted, so an op's cost is (blocks returned) + (wasted write-backs); each bound below bounds that sum.

**Minimum budgets** (rounds; batches of `k` keys use the same counts):

| map | mode | search | insert | delete |
|---|---|---|---|---|
| AVL | uncached | h | 2h | 2h |
| AVL | cached | h | h + 1 | floor((3h - 1)/2) |
| B+, 1 path/round | uncached | h | 3h - 2 | 4h - 3 |
| B+, 1 path/round | cached | h | 2h - 1 | 2h - 1 |
| B+, 2 paths/round | uncached | h | 2h - 1 | 2h - 1 |
| B+, 2 paths/round | cached | h | h | h |

Cached batched B+ gets the cached insert budget only at 2 paths/round with `k <= order + 1`; otherwise insert keeps its uncached budget (`BPlusOmapConfig.minimums`).

Why they hold (worst case: every fetched node held to the end; the last fetch round's write-back is staged after the op's final flush):
- **Search:** each level goes back just before the next read, so nothing is wasted.
- **AVL insert:** the `h` path nodes plus the new node; `h - 1` descent write-backs are wasted.
- **AVL delete:** `p` descent fetches plus at most two fetches per rotating level. Rotation fetches pay for blocks returned by lower levels. A rotating ancestor is two taller than its path child, so rotations are at most `h - p` and the total is at most `2h` for every `p`. Cached, the cost is the fetch count, `p + 2 min(h - p, p - 1)` (`avl_delete_fetches`).
- **B+ insert:** the held path plus one sibling per split (a root split needs `L <= h - 1`), so at most `2h - 1` blocks.
- **B+ delete:** the root plus a child and sibling per level, `2L - 1` nodes; at 2 paths/round each pair shares a round.
- **Cached:** an op's cost is `max(fetch rounds, ceil(returned / width))`, because owed blocks carried in fill the write-backs that would be wasted.
- **Carry limit** (`_carry_limit`, derived, not configured): the total width of the rounds between an op's cached and uncached budgets. With its uncached budget an op ends owing nothing, so with the cached one it is short by at most those rounds. Starting with more owed only removes waste, so the debt between ops never exceeds this. `drain()` always runs a fixed `ceil(C / widest round)` rounds and leaves nothing owed.
- **Batches:** all keys descend level by level, at most `min(fanout^r, k)` distinct nodes at depth `r`. Search releases each level. Insert holds the fetched union and inserts the keys one at a time locally (AVL `path_to`, B+ `_path_to`; B+ keeps new split nodes local until the flush). An unfetched subtree's key range holds no batch key, and rotations and splits never widen it, so no insert needs an unfetched node. In B+ a node splits only when a key arrives, so each level splits at most `k` times.

Pinned by: `test_each_avl_bound_is_reached`, `test_each_cached_avl_bound_is_reached`, `test_sorted_inserts_reach_every_{single_path,two_path,cached}_bplus_bound` (worst-case witnesses), `test_every_op_spends_exactly_its_budget` and `test_every_op_reads_the_same_widths_whatever_the_data` (identical round shapes), `TestOmapStash`, and the config table in `tests/dependency/test_config.py`.

## Failures

Errors the client can foresee (contract violations, the `num_data` limit) are raised before any I/O, so nothing changes and the call can be retried. Any other exception during an op (transport, stash overflow, a full bucket) leaves the scheme's state undefined: rebuild from a fresh client and storage. Schemes carry no recovery code. One ORAM runs one access at a time: an access begun inside another (for example from `update`'s `change`) is refused before any I/O. Two exceptions are unavoidable: `update` can only check the value its `change` returns after the read, and inserting a key that is already present is a caller error the maps cannot see before descending, so it is undefined.

## Conventions

- **No comments** in source or tests (tool pragmas excepted), and no docstrings that restate a signature. Rationale the code cannot carry (security arguments, invariants, budget derivations) goes here or in the commit/PR text.
- Shared methods on base classes use a single underscore so subclasses can reach them. Another object's underscore members are off limits in `src` (basedpyright `reportPrivateUsage`; tests are exempt so they can inspect counters): a composed scheme uses its parts' public surface (`name`, `num_data`, `data_size`, `leaf_range`, `key_size`, the ops, a position-map level's `access` / `build_and_host`, `_BucketStore`'s bucket ops).
- Validation: anything that depends only on one config is checked in its `__post_init__`; anything across two objects in the composed scheme's constructor; at run time only the per-call contract and the `num_data` limit. Internal invariants are `assert`s. Position-map levels are built from a copy of the scheme's config with a smaller `num_data`, so a rule about the whole scheme (such as `num_data` above the on-chip size) lives in the scheme constructor, not the config.
- Public names come from `oblivlib.dependency` (`__all__` is the list). Internal modules (`protocol`, `contract`, `heap_index`, `layout`, `balls_into_bins`, `path_cipher`, `server_stores`, `tree_storage_base`) are imported from their submodule.
- Invariants and their tests: no block held twice across trees, stashes and ODS `local` (`duplicate_blocks`, asserted by the model tests); row size equals the declared layout (`test_row_size_is_the_declared_block_layout_of_every_tree`, `test_row_size_is_the_declared_node_layout`); oversized fields raise `CapacityError` (`tests/dependency/test_layout.py`).
- Tests mirror source order: add a test next to the test of its source function's sibling. A trivial helper covered by a higher-level test needs no test of its own.
