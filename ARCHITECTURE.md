# oblivlib — Design Knowledge Base

How the library is put together, for anyone (human or agent) about to change it. The source carries no
comments by convention: any rationale the code cannot express lives here. Paper-to-file mapping is in
`README.md`; commands and conventions are in `CLAUDE.md`.

---

## 1. Scope and threat model

`oblivlib` implements oblivious data structures (ORAM and OMAP) for a
client/server deployment:

- The **client** is trusted, holds all secret state (position map, stash, keys), and may compute
  non-obliviously.
- The **server** is **honest-but-curious**: it follows the protocol, stores only encrypted blocks, and
  answers batched read/write requests. It must not learn the access pattern.

Because the server is not malicious, the protocol does not defend against hostile responses beyond
decoding every message as plain data (§2): a peer can lie, but it cannot run code.

---

## 2. Client, protocol, and server (the wire spec)

Every storage access goes through a `Client` (`src/oblivlib/dependency/client.py`). Schemes receive it
as `client=` in their config: it is the client-side handle *to* the server, not the server. The layers
under it, each owning one concern:

| Layer | Owner | Role |
|---|---|---|
| Scheme-facing API | `Client` | Stages queries, defers writes, hosts trees, keeps metrics. |
| Messages | `Backend` (`LocalBackend`, `TransportBackend`) | Delivers one request and returns its reply. |
| Bytes | `Transport` (`transport.py`) | Moves an encoded request; `ZmqTransport`, `LoopbackTransport`, `SimulatedNetwork`. |
| Engine | `StorageServer` (`storage_server.py`) | Answers requests; no transport of its own. |
| Stores | `server_stores.py` | Fixed-row trees (memory or file), variable-row trees (memory), lists. |

`Client.local(storage_dir=None)` wires a `LocalBackend` straight to a `StorageServer`: rows are
immutable `bytes`, so nothing is copied or encoded in process. `Client.connect(endpoint)` wires a
`TransportBackend` over a `ZmqTransport`; the server side runs `serve(ZmqListener(endpoint), server)`
until it has answered a `Shutdown`. A client is not thread-safe: use one per thread.

### Framing and decoding

Every message is a frozen, slotted dataclass in `protocol.py`, framed as the msgpack array
`[ClassName, *fields in declaration order]`; nested ops are framed the same way. Fields hold only ints,
str, bytes, None, lists, and nested ops. Both ends are this library (research code, trusted peers), so
decoding (`decode_request`, `decode_reply`) only rebuilds the dataclasses: a list whose first element
is a string is a tagged message, an unknown tag or a wrong field count raises, and field types are not
checked. No field is a list of strings, which keeps that rule unambiguous. `decode_request` also
rejects anything that is not a `Request` (a reply, a nested op, a bare value), and `handle_wire` turns
any decode failure into an `ErrorReply`, so only requests reach `_dispatch`, which answers `Ok()` for
every request it does not otherwise answer. Nothing is ever unpickled. The message set is the
`Request`, `Reply`, `WriteOp`, and `ReadOp` unions in `protocol.py`; read those for the current members.
`PROTOCOL_VERSION` is checked by `Hello`.

### Sessions and store lifecycle

Every request except `Hello` and `Shutdown` carries its session first. The client sends `Hello` lazily,
on its first request; the reply carries the session id (`secrets.token_hex(16)`). Labels are namespaced
per session. `CreateTree(level, row_bytes)` makes a tree of `2**level - 1` rows: fixed-size rows when
`row_bytes` is an int, variable-size (plaintext) rows when it is `None`, which a disk-backed server
refuses. `Resize` grows or shrinks a tree in place, refusing to
drop a present row (`ScaleDownError`); `Client.resize` refuses while reads of that label are staged,
since they name leaves of the old level. `WriteRange(start, rows)` writes consecutive rows (bulk
hosting). `Close` releases the session and deletes its files; `StorageServer.close()` closes every
session. A disk server keeps a tree at `storage_dir/<session>/<label as hex>.tree` and reads and writes
it with `os.pread`/`os.pwrite`.

### Batches

`Batch(session, writes, reads)` puts "writes first, then reads" in the message's structure.

- **Client side.** Path writes merge per label, a later write to a node winning; list ops append in
  staging order; path reads union per label (leaves sorted); a whole-list read absorbs index reads.
  `add_write_path` checks that the rows cover exactly the paths to their leaves, and staging an unknown
  label raises `UnknownLabelError` at once.
- **Canonical row order.** A path read or write carries its leaves, and its rows follow ascending heap
  index (root first) over the deduplicated union of their paths: `heap_index.path_indices`.
- **Absent rows.** `b""` on the wire is an absent node. A fixed-row store keeps it as an all-zero row
  and reads an all-zero row back as `b""`; a sealed row is never all zeros.
- **All or nothing.** The server validates the whole batch (`_validate_batch`) before applying any of
  it: every label exists with the right kind, every leaf is in range, each write carries one row per
  path node, every row fits its store, list ops are simulated against the running length, and list
  read indices are checked against the post-write length. Then it applies the writes in order and
  answers the reads, aligned with `Batch.reads`.
- **An empty `execute()` is still a round trip**, so the round count never depends on what was staged.

### Write deferral

`Client(defer_writes=True)`, the default, does not send an `execute()` that staged only writes: its
writes stay pending and ride on the next batch, which applies them before its own reads. Every read
therefore sees the state it would have seen undeferred, and the batch sequence is exactly the
undeferred one with each run of read-less batches folded into the next batch (leaves unioned per
label); `assert_deferral_preserves_access` in `tests/conftest.py` checks this for every scheme family.
Every scheme's operations alternate a read round and a write-back round, Freecursive's reset chunks
included, so the round count halves, the same for every key and value. `flush()` sends pending writes on their own,
every lifecycle call flushes first, and `close()` flushes. An error from a deferred write surfaces on
whichever call sends it. `defer_writes=False` sends every `execute()`.

### Errors

A failure comes back as `ErrorReply(kind, message)`; `handle` never raises. `ERROR_KINDS` in
`protocol.py` maps each kind to its class in `errors.py`; `error_reply` maps an `OSError` to
`StorageError` and anything unexpected to `ServerError`; `expect_reply` raises the mapped class and
turns an unknown kind or the wrong reply type into `ProtocolError`. `Client.execute()` raises;
`ExecuteResult` holds only results, and `require(label)` raises `MissingResultError` for a label with
no result. A request with no reply within `DEFAULT_TIMEOUT_MS` (`transport.py`) raises
`TransportError`; the `REQ` socket is then unusable (any later request, including `close()`'s once a session
exists, raises `TransportError` at once) and nothing is resent, so start a new client.

### Limits and metrics

There is no message-size limit. The client streams a tree in `WriteRange` chunks of at most
`WRITE_RANGE_CHUNK_BYTES` (`client.py`) so no single message holds the whole tree. `client.metrics`
counts `rounds` (batches sent, empty ones included), `payload_bytes` (row and list-value bytes written
and read, `WriteRange` included), and `wire_bytes` (encoded request and reply bytes, counted by
`TransportBackend`; zero in process). Local and loopback clients report identical rounds and payload.
`SimulatedNetwork` wraps a transport and accumulates modeled time, a round-trip time plus bytes over
bandwidth per request, without sleeping.

### Hosting a built tree

`Client.host_tree(label, image)` hands over a tree built on the client (§3). A `MemoryImage` is
created and streamed with `WriteRange`. A `FileImage` already has the server's on-disk format, so it
can move without re-encryption: placed out of band by the client's `ship_file` (which returns its name
inside the server's storage directory) and then attached, adopted in place when the backend exposes the
server's `storage_dir` (the file is moved there and attached), or else streamed row by row. A
successful hand-over leaves no local build file. `AttachTree` accepts only a plain file name directly in
`storage_dir` (no separators, no `..`, not absolute), requires the exact size, and moves the file into
the session's directory; an in-memory server refuses it. A failed hand-over or attach leaves the file
where it was; it holds only sealed rows, and a file shipped into `storage_dir` can be attached again
by the name `ship_file` returned; an adopted file is moved back to its build path. A stream that fails
partway leaves the label created with the rows sent so far, so that label cannot be hosted again, and a
streamed `FileImage` is not checked against its expected size.

---

## 3. Data and storage layer

### The data contract

ORAM keys are int addresses in `[0, num_data)`; OMAP keys are `bytes` of at most `key_size`, ordered by
plain bytes comparison (OMAPs only search, insert, and delete exact keys); values are `bytes` of at most
`data_size`. Users serialize their own data. `contract.py` checks keys and values at every public entry
point, before any state changes, and raises `ContractError` (a `ValueError`) prefixed with the scheme's
identity. Initial data gives each key at most once, in every ORAM and OMAP (and per tree in
`init_mul_tree_server_storage`); a repeated key raises `ContractError` before anything is hosted.
OMAP init checks all pairs up front (`contract.require_omap_init`); ORAM init checks while streaming
(`TreeBaseOram._initial_blocks`), so a repeat stops the build partway. An ORAM address never written
reads `b""`; `None` is not a value. Internal structured values (position-map blocks, composed-map roots
and metadata, ODS nodes) are fixed-layout records (below); the library never pickles.

### Blocks and codecs

`Data(key, leaf, value)` (`types.py`) is a block. `Data`, `AVLData`, and `BPlusData` share the
`FieldTuple` mixin: `to_fields()` lists the dataclass's fields in declaration order and `from_fields`
rebuilds the instance, so reordering, adding, or removing a field changes the stored format.

Every stored record has a declared layout of fixed-width fields (`layout.py`): `UInt(bits)`,
`Bytes(max_size)` (length plus zero padding), `FixedBytes`, `Optional`, `Array`, `Seq(field, max_count)`
(count plus padded slots), `Record`, and `Variant`, each with a fixed `size`; a value that does not fit
its field raises `CapacityError` (a `ValueError`) while it is encoded. A scheme declares its block as
`block_layout(_key_field, _leaf_field, _value_field)` (`TreeStorageBase` defaults: `UInt` for
`num_data - 1`, `UInt` for `leaf_range - 1`, `Bytes(data_size)`); AVL, B+, Recursive and counter levels,
and the GroupOmap lower ORAM override fields. B+ block ids are `UInt(64)`, so the growing id counter
never outruns its field. Counter blocks are one `UInt` word read and written through `BitFields`.

### Path ciphers and rows

A `PathCipher` (`path_cipher.py`) turns a bucket into a server row and back, and is the one path both
initial building and per-op eviction go through (`TreeStorageBase._cipher`). A bucket is
`Seq(block, bucket_size)`, so it always encodes to the same length.

- `PlainPathCipher`: the row is the encoded bucket (memory servers only). An empty bucket encodes to a
  non-empty row, so a present empty node stays distinct from an absent one.
- `SealedPathCipher`: the row is `enc(encoded bucket)`, so `row_bytes` is
  `ciphertext_length(Seq(block, bucket_size).size)`, computed from the layout. A bucket with too many
  blocks or a field that does not fit raises `CapacityError` instead of producing a longer row.

### Heap-index math (`heap_index.py`)

Trees are complete binary trees flattened into heap order: the root is index 0 and node `i`'s parent is
`(i - 1) // 2`. All tree math is integer-only module functions in `heap_index.py` (imported from the
submodule, not re-exported): `compute_level(num_data) = (num_data - 1).bit_length() + 1` (the smallest
level whose `2**(level-1)` leaves cover `num_data`), `tree_size`, `leaf_index`, `parent`, `path_to_root`
(node first), `path_indices` (the canonical row order: deduplicated, root first), `empty_path`, and
`fill_data_to_path`.

- `leaf_lca(leaf_a, leaf_b, level)` is O(1). Two leaves at the same depth have 1-based heap indices
  `leaf + 2**(level-1)` of equal bit length, and an ancestor's 1-based index is a binary prefix of its
  descendant's. The highest set bit of `a ^ b` marks the first level where the two root-to-leaf paths
  diverge, so `a >> (a ^ b).bit_length()` is their longest common prefix, the lowest common ancestor
  (minus 1 for the 0-based index). This replaces an O(depth) parent walk in the inner loop of stash
  eviction. It only holds for nodes at equal depth, which is all eviction needs.
- `fill_data_to_path` places a block at the deepest bucket on both its own path and the target path
  set, bubbling up toward the root if full.

### Building the initial tree (`tree_builder.py`)

`build_tree(blocks, level, bucket_size, cipher, build_file)` places each block in the deepest non-full
bucket on its leaf's path and seals every bucket. Without a `build_file` the result is a `MemoryImage`
(all `2**level - 1` buckets in memory: never build one at large `num_data` just to read its level; use
`heap_index.compute_level`). With one it is a `FileImage` in the server's format: the file is truncated,
every row pre-sealed empty, and each placement reads, opens, appends, seals, and writes its row in
place, so the file never holds plaintext. Blocks whose whole path is full come back as `overflow`.
A failed build leaves its partial `build_file`, which holds only sealed rows.

`TreeStorageBase._build_tree` is the one construction path: it builds with the scheme's cipher and
`build_file`, puts the overflow in the stash (as in Path ORAM), and checks the stash. Blocks stream in:
`TreeBaseOram._initial_blocks` yields the supplied `(key, value)` pairs (a mapping or a one-shot
iterator), tracking supplied keys in a `bytearray(num_data)` so a duplicate raises, then `b""` for every
other address; `_initial_leaf` gives each address's leaf (`StaticOram` overrides it). The recursive
schemes host the main tree first, then each position-map level builds and hosts its own tree
(`_build_and_host`, through the level's `_build_tree` and `_host_tree`), largest level first and one at a
time, so the levels reuse the main ORAM's `build_file` path in turn. A Recursive level draws its own
leaves, then its padding leaves, as it builds, so the seeded draw order is per level, largest first. The
ODS init paths collect their blocks and build once.

### Server stores (`server_stores.py`)

A `TreeStore` holds `2**level - 1` rows. `FixedRowStore` keeps rows of exactly `row_bytes` in one
region, a `bytearray` or a file, and rejects any other non-empty length (`RowSizeError`);
`VariableRowStore` keeps rows of any length in memory. `resize` truncates or extends in place, so the heap
prefix and its rows survive, and refuses to drop a present row. `ListStore` holds a list of `bytes`.

### Construction configs (`config.py`)

Every ORAM/OMAP is built from a frozen, keyword-only config whose hierarchy mirrors the scheme
hierarchy and holds each scheme's defaults:

- `OramConfig` (`num_data, data_size, client, name, build_file, bucket_size, stash_scale, encryptor`)
  → `PathOramConfig`, `StaticOramConfig`, `MulPathOramConfig`, `RecursiveOramConfig`,
  `CounterOramConfig` → `DaOramConfig`/`FreecursiveOramConfig`. `build_file` requires an `encryptor`.
- `OmapConfig(OramConfig)` adds `key_size` and `distinguishable` → AVL/B+ configs (and their cached
  variants) and `GroupOmapConfig`. `OramOstOmapConfig` stands alone (only `num_data`).

`client` is required. `name` defaults to `unique_name(prefix)`, a per-scheme prefix plus a random suffix,
so only a caller that needs a known storage label passes one. Configs validate in `__post_init__`. Dataclasses don't chain `__post_init__`, so every subclass that
adds a constraint calls `super().__post_init__()` first.

The config is the single source of truth. `TreeStorageBase` exposes each field as a read-only private
property (`self._num_data`, `self._client`, …). Derived values (`_level`, `_leaf_range`) are computed
once in `__init__` because they are hot. Objects that aren't configuration (`StaticOram`'s
`prf`, `GroupOmap`'s `upper_oram`, `OramOstOmap`'s `ost`/`oram`) are separate constructor arguments.
Derive variants with `dataclasses.replace`.

**Position-map levels are not ORAMs.** A recursive scheme's position map is a chain of level objects,
`RecursivePosMapLevel`, `DaPosMapLevel`, and `FreecursivePosMapLevel`, each a `TreeStorageBase` rather
than a `TreeBaseOram`. The main ORAM makes them in its constructor, each with
`replace(self._config, num_data=…, name=f"{name}_pos_map_{i}", …)`; a counter level also takes the
previous (larger) tree's size and level as required keyword ints (`last_oram_data`, `last_oram_level`),
so the recursion plumbing stays out of the config. `_levels[0]` is the smallest level (`i = 0`, the
outermost map), and levels are hosted largest first; both the labels and the hosting order show in the
server's access trace. A level shares the main ORAM's client and owns its name, which is also its
storage label. Each level drives its own per-level I/O (`_access_pos_map_level`); the main ORAM only
samples leaves and threads them down the chain.

---

## 4. ORAM layer (`src/oblivlib/oram/`)

### `TreeStorageBase`

`src/oblivlib/dependency/tree_storage_base.py`, generic over its config type. It is the shared core of
`TreeBaseOram` and the ODS base `OstBaseOmap`: config accessors, level/leaf-range/stash-size math,
the block layout (`_block_layout`), `_get_new_leaf`, the stash with its capacity check (`_check_stash`,
which records the peak stash size in `_max_stash` and then raises `StashOverflowError`, also a
`MemoryError`, on overflow), the stash helpers (`_require_in_stash`, `_update_stash_leaf`), tree
construction and hosting (`_build_tree`, `_host_tree`), the path rounds every scheme goes through, and
the scheme's `PathCipher` (`_cipher`, built from its block layout; §3, §6). It lives in `dependency/` so
both higher layers depend downward on it rather than on each other.

**A path read and its write-back are one unit.** `_open_paths(leaves)` reads paths and returns their
opened buckets; `_read_paths(leaves)` also moves every block into the stash, and the ODS fetches take
the opened buckets to split nodes between `local` and the stash. Reading moves the blocks to the
client, but the server keeps its copies until those paths are rewritten, so the storage base records
the read leaves (`_unwritten_leaves`) and `_write_paths()` evicts the stash onto exactly those paths,
block-major (`_stage_write_back`). Callers never name the paths they write back. If an access raises
between the two, the leaves stay recorded, and the next `_open_paths` stages their write-back ahead of
its own read; the batch applies writes before reads (§2), so no read ever sees a server copy of a block
the client already holds. The stash gives up a block only once its sealed row is staged, so an
eviction that raises while sealing loses nothing. Without this, a read overlapping an unwritten path
absorbs a second copy of a block; eviction then writes both back and the copies diverge as one of them
is updated.

### `TreeBaseOram` and the access protocol

Adds eviction and one access template. The position map belongs to the schemes that keep one
(`PathOram` and its subclasses other than `StaticOram`, whose `_new_pos_map` is empty, and
`RecursivePathOram`'s on-chip map); DA and Freecursive derive leaves
from a PRF. Each scheme implements a single hook, `_operate_on_key_without_eviction(key, value)`: it
finds the key's leaves (and does any position-map level I/O), then calls `_read_block`, which reads those
paths in one batch, pulls their real blocks into the stash, reads the key (optionally overwriting it),
remaps it, and marks the access pending (`_access_pending`). `_write_back` writes the read paths back
(`_write_paths`) and clears the mark. The public methods check the data contract and
compose the two:

- `operate_on_key(key, value=UNSET)` is the hook then `_write_back`, and returns the value *before*
  any write. With write deferral (§2) the write-back rides on the next operation's read batch, so an
  access costs one round instead of two.
- `operate_on_key_without_eviction` is the hook alone, and `eviction_with_update_stash(key, value)` sets
  the key's value in the stash, then calls `_write_back`. The split lets a caller (`OramOstOmap`,
  `GroupOmap`) compute the new value between the read and the write-back; the server sees exactly the
  fused access. An ORAM holds one pending access, so nothing else may run on the same instance between
  the two halves (another access could evict the pending block, or drop the pending paths' write-back).
  Starting another access while one is pending raises a `ValueError` before any I/O, and
  `OramOstOmap` and `GroupOmap` finish their outer ORAM access in a `finally`, so an inner op that raises
  (a full GroupOmap bucket's `MemoryError`, a failed tree op or lower access) still writes the slot back:
  the current ODS root for `OramOstOmap`; for `GroupOmap`, the new seed once the lower read has remapped
  the bucket, and on a failed insert the new key only while its block is still in the lower stash. A lower
  eviction that raises still leaves the lower ORAM's own access pending.
  `eviction_with_update_stash` with no access pending raises a
  `ValueError` before touching the stash.

Freecursive extends the template for its resets: chunks before the key's are read and written back in
the hook (`_pass_reset_chunk`), the key's chunk is the pending read, and the chunks after it wait in
`_pending_reset_chunks` until its `_write_back` override passes them after the key's own write-back.
A counter update always returns reset chunks: a plain increment is one single-entry chunk for the
child, so the level and the main ORAM share one walk (`_split_at_key`) with no separate no-reset branch.
`MulPathOram`'s batch API uses the same pending access: `_operate_on_keys_without_eviction` reads every
key's path and records them, and `_eviction_for_mul_keys` applies updates and calls `_write_back`.

`TreeBaseOram` is generic over its position-map level type. `_levels` lists a recursive scheme's levels,
smallest first; the non-recursive schemes pass `Never` and keep it empty. `max_stash` is the main ORAM's
peak stash plus every level's peak, an upper bound since the peaks need not coincide.

**Eviction is block-major and that is deliberate.** Each stash block goes to its deepest legal bucket,
bubbling up if full. This matches the textbook bucket-major sweep's overflow exactly (both are maximal
placements on a laminar matroid) and is ~15% faster at realistic stash sizes. Do not switch.

### Schemes

| Scheme | Idea |
|---|---|
| `PathOram` | Random leaf reassignment on every access. |
| `MulPathOram` | Reads/evicts several paths per batch (`operate_on_keys*`, `eviction_for_mul_keys`, each checking the contract, including every leaf in `key_path_map`/`new_path_map`, before its underscore twin; `eviction_for_mul_keys` raises `KeyError` for an update whose key is not in the stash, before changing any value). `stash_scale_multiplier` is baked into `stash_scale` at construction. A batch given explicit read paths (`key_path_map`) leaves the position map alone: the caller owns those keys' positions, so it should also pass `new_path_map`, or the random new leaves are recorded nowhere (`GroupOmap` passes both). |
| `StaticOram` | Leaves fixed by `PRF(key)`: the "new" leaf equals the current one. |
| `RecursivePathOram` | Position map compressed into a chain of smaller ORAMs, each block holding `compression_ratio` child leaves; only the smallest map (≤ `on_chip_mem`) stays on chip. A short final block is padded with filler entries that no key ever reads. |
| `DAOram`, `FreecursiveOram` | Leaves are `PRF(num_data ‖ key ‖ GC ‖ IC) mod leaf_range`, with fixed-width big-endian fields and `num_data` the size of the tree whose leaf it is; position-map blocks hold counters instead of leaves (below). |

### Counter blocks (DA / Freecursive)

**PRF domain separation.** Every tree in the chain shares one PRF key, so the message names its tree:
`num_data` strictly decreases along the chain (the compression ratio is at least 2), and every field has
a fixed width. Without the tag, a position-map block and the data block with the same number, at the
same counters, hashed the same message, and their leaves agreed modulo the smaller range, which linked
accesses across trees.

A position-map block covers `num_ic` children and stores a group counter (GC) plus one individual
counter (IC) per child. An access bumps the child's IC and recomputes its leaf from the PRF. When ICs
would overflow, the block **resets**: GC increments, ICs zero, and every child's leaf is recomputed.

- **Freecursive** layout is `GC ‖ IC×num_ic`. `reset_method="prob"` resets with probability
  `reset_prob` (default `1/num_ic`, drawn from the OS CSPRNG) and treats an overflow as an error.
  `"hard"` resets only on overflow, which is the insecure original scheme. A reset fans out into chunks
  of two child paths read together. Slots past the end of the data read a random path so the data
  boundary stays hidden. A split access finishes the reset chunks after the key's own chunk inside
  `eviction_with_update_stash`, because later chunks depend on the earlier write-backs. Eviction always
  executes: write deferral (§2) already holds back a write-only batch, so there is no flag to stage it.
- **DAOram** layout is `GC ‖ IC×num_ic ‖ indicator×num_ic`. Instead of resetting all children at once,
  it de-amortizes: an indicator bit marks a child whose backup count is still pending, and each access
  carries at most one pending reset alongside it. A consumed backup clears only that indicator. An
  overflow bumps GC and sets every other child's indicator. **Every access reads two paths**, the data
  path and a reset path (random when none is due), so a reset is indistinguishable from no reset.

A counter update is a pure function of the counter word, shared by the on-chip counters and the stored
blocks: DA's `_step` and `_reset` (`DaCounterMixin`), Freecursive's `_advance`. A group counter that
would outgrow its field raises `CapacityError` from `BitFields.set`; `gc_length` is at least
`MIN_GC_LENGTH` (64, `config.py`), since GC grows at most once per access.

Everything else lives once in `oram/counter_oram.py`. `CounterMixin` is a `Protocol` holding the
counter widths, the counter layout (`_counters`), and the PRF leaves (`_prf_leaf`, `_get_previous_leaf_from_prf`); the
layout's total width is the config's `count_length`, and `on_chip` names the config's on-chip size
(`on_chip_mem` / `on_chip_size`). `CounterPosMapLevel` (a `TreeStorageBase`) is the level constructor
and `_build_and_host`. `CounterOram` (a `TreeBaseOram`) is the main ORAM's constructor, on-chip
storage, `_initial_leaf`, `_make_levels` (through the scheme's `_new_level` factory), and
`init_server_storage`. Freecursive adds `FreecursiveCounterMixin` for `_reset_prob`,
`_get_reset_leaves`, and `_single_entry`. The mixins are `Protocol`s because they only declare the
attributes they read (`_config`, `_prf`, `_last_oram_data`, `_last_oram_level`): a plain mixin with
declared-but-unassigned attributes trips `reportUninitializedInstanceVariable`, and a shared base
deriving from `TreeStorageBase` under both roles would form a diamond with `TreeBaseOram`, which
`reportUnsafeMultipleInheritance` rejects. Reset state moves through `NamedTuple` records (`ResetLeaf`,
`DaStep`, `ResetEntry`). `ResetLeaf`'s field is `offset` because a NamedTuple field cannot shadow
`tuple.index`.

---

## 5. OMAP layer (`src/oblivlib/omap/`)

### Families

- **ODS maps.** `OstBaseOmap` (a `TreeStorageBase`) with `AVLOmap`, `BPlusOmap`, and the cached
  `AVLOmapCached`/`BPlusOmapCached`. Each holds a `root` pointer `(key, leaf)`, a per-op `local` set of
  downloaded nodes (`LocalNodesBase` plus scheme `LocalNodes`), and the stash, and exposes `search`,
  `insert`, and `delete`, which check the contract and call each scheme's `_search`/`_insert`/`_delete`.
  `search(key, value)` returns the old value (`None` when absent) and writes `value` when it is not
  `None`; `key=None` is a dummy op. Insert assumes the key is absent: duplicate keys are undefined
  behaviour.
- **Composed maps.** `OramOstOmap` (VLDB 2025) hashes each key with a PRF into an ORAM slot that stores
  the root of that slot's ODS tree, encoded with the ODS's `root_layout`. An op fetches
  the root, runs the ODS op, and writes the root back. It works with any `TreeBaseOram` × `OstBaseOmap`,
  and calls `update_mul_tree_height` so the ODS sizes its budgets for one small tree per slot. `search`,
  `insert`, and `delete` share that fetch-run-write-back sequence (`_on_slot`). The ORAM must be at
  least `OramOstOmap.oram_data_size(ost)` wide; construction checks this.
- **`GroupOmap`** hashes keys into buckets. An upper ORAM stores per-bucket metadata
  a fixed-layout `(seed, keys)` record, and a lower `MulPathOram` stores each value in a block **named by its key**,
  on the path `PRF(seed ‖ key) mod leaf_range` of the lower tree (not `num_data`, so search paths cover
  the same range as an insert's uniformly random path). A search reads every block of the key's bucket in one batch and
  reshuffles them under a new seed. An insert stashes a new block on its path through
  `MulPathOram._insert_block`, which reads and evicts one uniformly random path, so the server sees the
  same single-path access as before. GroupOmap drives the lower ORAM through its underscore
  (contract-free) methods, since the lower ORAM is internal and keyed by bytes rather than addresses. Its
  block key field is `Bytes(key_size)`. Buckets hold at most `load_bound.max_bucket_load(num_data)` keys, and the upper ORAM must be
  at least `GroupOmap.upper_oram_data_size(num_data, key_size)` wide; construction checks this.

### Obliviousness model

Every op pads to a fixed round budget: real fetches are counted in `_op_rounds`, then `_pad_to(budget)`
adds dummy read+evict rounds. `_short_circuit_read` routes dummy (`key is None`) and empty-tree ops
through the same padding, so a hit, a miss, a dummy, and an empty map all look identical. Budgets come
from `_op_round_bounds` as functions of the height bound `h`. With the default
`distinguishable=False`, every op pads to the largest bound so the op type is hidden; `True` lets each
op use its own bound.

Budgets count rounds: `_op_rounds` counts real fetch rounds (a stash hit spends none), and each padding
unit is one dummy read plus its eviction, so every op spends exactly its budget, hit or miss, cached or
not, and eviction runs on every round. A round reads `_round_width` paths (1; 2 for `BPlusOmapCached`,
whose delete reads a level's child and sibling together), topped up with distinct random leaves. With
`h` the height bound below: AVL search `h`, insert `h`, delete `3h - 2` (one root-to-replacement path,
then at most two off-path fetches per remaining path node); B+ search `h`, insert `h`, delete `2h - 1`
(the root, then child and sibling per level); `BPlusOmapCached` delete `h`. A negative pad raises: it
means a bound is wrong. The cached maps serve stash hits from the client, but pad them, so their round
counts equal the plain maps' (B+ delete aside).

`search` streams: it re-homes and evicts each node before reading the next, so `local` stays O(1).

**Height bounds** come from the fill invariants (the tallest tree they allow; B+ inserts within capacity may stay a level lower): AVL `h` is the largest height whose minimal tree (`N(1) = 1`, `N(2) = 2`,
`N(h) = N(h-1) + N(h-2) + 1`) has at most `n` nodes (`avl_max_height`); B+ `h` is the largest level
count whose minimal tree (an internal root with two children, `ceil(order/2)` children per internal node,
`ceil(order/2) - 1` keys per leaf) holds at most `n` keys (`bplus_max_levels`). In multi-tree mode
`update_mul_tree_height` applies them to the per-slot bound from the Lambert-W max-load formula of
eprint 2021/1280 at a 2⁻¹²⁸ overflow probability (`load_bound.max_bucket_load`, which `GroupOmap` also
uses; `lambert_w` solves `w·eʷ = x` by Halley's method).

### Plaintext reference trees (`dependency/avl_tree.py`, `bplus_tree.py`)

`AVLTree`/`BPlusTree` build the initial ODS storage (`get_data_list`) and are the standard the oblivious
ports mirror. Each keeps a recursive twin of `insert`/`delete` that tests cross-check structurally. The
non-recursive `delete`s are the templates for the oblivious deletes: they record the visited path in a
`local` list in clear phases (locate, remove, repair bottom-up). `AVLTreeNode`/`BPlusTreeNode` are the
in-memory pointer trees used only for this initial build; `AVLData`/`BPlusData` are the stored node
values, with child keys and leaves in place of pointers, that the oblivious maps operate on.

- An ORAM block's value is one node: `AVLData` holds the value plus each child's `(key, leaf, height)`;
  `BPlusData` holds parallel `keys`/`values` lists, where an internal node's values are child
  `(id, leaf)` pairs and a leaf's are the stored values.
- `AVLTree` routes equal-or-larger keys right (the oblivious insert routes equal keys left; only
  duplicate keys, which are undefined behaviour, can tell). A two-children delete replaces the node with the in-order
  predecessor when the left subtree is taller, else the successor; `delete` and `recursive_delete` make
  the same choice, so they produce structurally identical trees.
- B+ leaf splits keep the median in the right half and copy it up as the separator. Internal splits
  move the median up and carry the extra child pointer. The median is read before the split mutates the
  node. Underflow repair (`_fix_underflow`) uses a single sibling (left preferred): borrow if it can
  spare a key, else merge with the left node absorbing the right. Considering exactly one sibling rather
  than the better of two is what the oblivious map follows: it prefetches just that sibling, so the work
  per level is fixed and independent of the borrow/merge outcome. Both deletes route through
  `_fix_underflow`, which is what makes them cross-checkable.

---

## 6. Crypto (`src/oblivlib/dependency/crypto.py`)

- `AesGcm`: `nonce(12) ‖ ciphertext ‖ tag(16)`. GCM adds no padding, so length is plaintext + 28.
- `Blake2Prf`: keyed BLAKE2b, with `digest_mod_n` for leaf derivation. `hash_data_to_leaf` and
  `hash_data_to_map` hash `bytes` keys with it for the composed maps.
- `FeistelPrp`: a 4-round balanced Feistel (the Luby–Rackoff bound for a strong PRP) over an even bit
  width. The SHA-256 round function clones a key-seeded hash per round, which is equivalent to hashing
  `key ‖ round ‖ value`. Non-power-of-2 domains use cycle-walking, which stays a bijection. A round's SHA-256 output mixes
  at most 256 bits per half, so domains above `2**512` are rejected.

**Encryption is per bucket.** Building the initial tree and evicting a path both seal through the
scheme's one `SealedPathCipher` (§3): the bucket is encoded in its fixed layout and encrypted once, so there is one nonce+tag per bucket and
every row has the same length whatever it holds. Only the client holds the encryptor; the server stores
and returns sealed rows. GCM authentication is defense in depth under the honest-but-curious model, not
malicious-server integrity: it has no replay or freshness protection. Random 96-bit nonces cap safe use
at ~2³² encryptions per key.

`encryptor=None` stores each bucket as its plaintext encoded row on a memory server only; the tests use this for speed. A `build_file` always needs an encryptor, so plaintext never reaches
disk.

---

## 7. Invariants and contracts

- A block mapped to leaf `x` is on the root→`x` path or in the stash, and is held exactly once: no
  read sees a server copy of a block the client already holds (§4, "A path read and its write-back are
  one unit"). The model tests check this across every tree and stash of a scheme
  (`duplicate_blocks` in `tests/conftest.py`).
- Recursive schemes require `num_data` > on-chip size (`on_chip_mem` for DA/Recursive, `on_chip_size`
  for Freecursive); the constructor raises otherwise. The number of position-map levels is exact: the
  fewest `L` with `on_chip * ratio**L >= num_data`, in integer arithmetic (`pos_map_level_count` in
  `tree_base_oram.py`; DA and Freecursive keep the last level on chip, so they use one fewer). A ratio
  below 2 cannot compress, so the configs reject it (`compression_ratio`, `num_ic`).
- `max_stash` is the main ORAM's peak stash plus every position-map level's peak; each peak is recorded
  in `_check_stash`, so it includes build overflow.
- A `MulPathOram` batch with explicit read paths (`key_path_map`) leaves the position map unchanged
  (§4).
- `FreecursiveOramConfig.reset_method` is one of `ResetMethod` (`config.py`); anything else raises
  `ValueError` at config construction.
- Keys and values obey the data contract (§3); a violation raises `ContractError` before any client
  state changes or any round is spent.
- For the per-op-oblivious schemes (every ORAM and every ODS map, cached ones included; not `GroupOmap`,
  §8), the accessed-path sequence and the round count depend only on the sequence of calls, never on
  keys, values, or cache state; write deferral folds read-less batches into the next batch the same way
  for every input (§2).
- A sealed row is exactly `row_bytes` long whatever its occupancy or value lengths, and building and
  per-op eviction seal through the same `PathCipher`, so an initial row is indistinguishable from an
  evicted one.
- Every scheme on a shared client has a distinct `name` (its storage label). Configs default it to
  `<prefix>-<8 hex digits>` drawn from `secrets.randbelow`, and the server rejects a reused label.
- ORAM: `operate_on_key(key)` reads, `operate_on_key(key, value)` writes `value` (`bytes`), an address
  never written reads `b""`, and the return value is always the pre-write value.
- `FlexibleBinaryTree` absent nodes are empty rows and are skipped by `read_path`, so the server sees
  which nodes are empty versus present. That is the same information the write lengths already reveal,
  and the adopting scheme controls both.
- A real op that exceeds its round budget fails loudly: `_perform_dummy_operation` raises on a negative
  pad count. The check runs after the op's fetches reached the server and its changes took effect, so the
  extra rounds have already leaked; it reports a broken bound, it does not prevent the leak.

---

## 8. Known issues

- **`FlexibleBinaryTree`** (`flexible_binary_tree.py`) has no callers yet; it is the resizable tree a
  future scheme (e.g. SORAM) is meant to adopt. It is a client-side handle over a hosted tree
  (`FlexibleBinaryTree.create` builds and hosts it). Its model is client-driven: leaves are int labels at
  the tree's current `level` (the caller keeps them in range), and every node is either present (a
  bucket, possibly empty) or absent (an empty row); all nodes start present. `read_path(leaves)` returns
  the present nodes on those paths, root first. `write_path(leaves, data)` rejects an index off those
  paths, writes each bucket in `data`, and writes every other node on the paths as absent. `scale_up()`
  adds an absent bottom layer and `scale_down()` drops the bottom layer, both through the server's
  `Resize`, which raises `ScaleDownError` at level 1 or while any bottom node is present. Neither moves
  data: evacuating the bottom layer before shrinking is the scheme's job, done through ordinary
  `write_path` calls.
- **Type checking:** basedpyright `recommended` is clean across `dependency`/`oram`/`omap` with no
  `type: ignore`. `OstBaseOmap` is generic over its `LocalNodes` container rather than narrowing a base
  attribute, and `TreeBaseOram` is generic over its position-map level type the same way (a mutable
  `_levels` attribute is invariant, so a subclass could not narrow a base `list[TreeStorageBase]`).
  msgpack ships no stubs, so `typings/msgpack/__init__.pyi` declares the calls the library
  makes.
- **GroupOmap leaks bucket occupancy:** a search reads one lower path per key in the bucket, so the
  number of lower paths depends on how full the bucket is.
- **A split Freecursive access shows where the key's reset chunk sits:** the reset chunks before the
  key's run in `operate_on_key_without_eviction` and the ones after it in `eviction_with_update_stash`, so
  traffic a caller interleaves between the halves (as `OramOstOmap` does) reveals `(key % num_ic) // 2`
  on an access that resets. The fused `operate_on_key` is unaffected.
- **Counter overflows are not recoverable:** a probabilistic-reset IC overflow, or a group counter
  overflow, at a position-map level raises `CapacityError` after that level's read and leaves the
  instance inconsistent.
- **An access that raises partway through a multi-tree scheme is not rolled back.** No block is
  duplicated (§4), but a position-map level, an ODS parent, or a counter may already name a leaf the
  data block was never moved to, so a later access of that key raises `KeyError`. Single-tree accesses
  recover: a failed `_write_back` leaves the access pending for `eviction_with_update_stash` to finish,
  and a failed `MulPathOram._insert_block` keeps its block in the stash.
- **An ODS op that raises mid-way leaves the map unusable:** fetched nodes stay in `local` and `root`
  may name a stale leaf; later ops raise or lose nodes. After a `StashOverflowError`, later ops raise
  it again until evictions bring the stash back within capacity, and may then hit the stale pointers
  (`KeyError`). `num_data` bounds sizing but is never enforced, so overfilling a map ends this way
  rather than with a `ContractError`.
- **`GroupOmap` gaps:** inserting a key already present stores a second block (undefined behaviour,
  as for the ODS maps); `GroupOmapConfig.distinguishable` is ignored; the lower `MulPathOram` builds a
  `num_data` position map that `GroupOmap` never reads.
- **Composed-map capacity errors** (a full GroupOmap bucket) raise a bare `MemoryError` once the upper
  access completes; no lower access is made, so the server sees the failed insert as a shorter op.
- **Not yet built:** a round driver and handle API for schemes, retrying or resending a lost batch, and
  persistence (reattaching a server's trees after a restart). `GroupOmap` ignores `build_file`, and a
  path or list read result is untyped (`ExecuteResult.require` returns `Any`).

---

## 9. Tests (`tests/`)

- `tests/conftest.py`: the `num_data` fixture (default `2**12`, overridable via the `NUM_DATA` env var,
  which avoids needing a rootdir `pytest_addoption`), `client` (`Client.local()`), `remote_client` (a
  `Client` over `TransportBackend(LoopbackTransport(StorageServer()))`, the full wire path in process),
  `recorded_client` (a factory for a client over a `RecordingBackend` that records every message),
  `seeded_access` (runs a workload under a fixed seed on a recording client and returns its batches'
  access shapes and rounds), `assert_deferral_preserves_access` (built on `seeded_access`: runs a
  workload with deferral off and on and compares the batches), `read_all_blocks` (every block a tree
  holds), `duplicate_blocks` (keys held more than once across a scheme's trees, stashes, and ODS
  `local`, composed maps included; the model tests assert it is empty), `fail_once` (makes one call of
  a method raise), `encryptor`, and `test_file`.
- `tests/dependency/`: `make_search_tree` parametrizes one behavioural suite
  (`test_search_tree_common.py`) over AVL and B+. Per-tree files hold the structural invariant validators
  and recursive/iterative cross-checks. The protocol, server, client, transport,
  cipher, builder, and configs each have their own file (`test_config.py` holds one table covering every
  config check); `FlexibleBinaryTree` runs over memory-plain and memory-encrypted trees, since it has no
  storage-specific code. Test order mirrors source order. Trivial helpers covered transitively get no
  dedicated test. Code whose public behavior the ORAM/OMAP layers own (`contract.py`,
  `TreeStorageBase`, the composed maps' key hashing, `max_bucket_load`) is tested through those layers; `test_tree_storage_base.py` and `test_load_bound.py` add only the stash bound, `_require_in_stash`, default names, and `lambert_w`.
- `tests/oram/`: `make_oram` parametrizes over `ORAM_SPECS` (Freecursive twice, prob/hard). Adding one
  `ORAM_SPECS` line enrols a scheme in the whole common suite, which checks every scheme against a dict
  model and checks that a split access (`operate_on_key_without_eviction` + `eviction_with_update_stash`)
  reaches the server exactly as the fused one does; DA and Freecursive repeat that check through counter
  resets, with write deferral on and off. The recursive schemes also check `max_stash` against
  their levels' peaks, and the edge cases pin the exact position-map level count at float-boundary
  sizes. `MulPathOram`'s batch API, edge cases, and the remote protocol and handover modes have their own
  files.
- `tests/omap/`: `omap_spec` parametrizes over `OMAP_SPECS`, each entry declaring its guarantees
  (its round `budgets` at `num_data=64`). The obliviousness tests run for every map, count path reads
  and read rounds through a recording client, and check every op spends exactly its budget.
  Composed maps have their own files.
