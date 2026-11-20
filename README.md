# OblivLib

OblivLib (`oblivlib`) implements classic oblivious algorithms (oblivious RAM and oblivious maps) for
client/server deployments where the client is trusted and may compute non-obliviously, and the server only
stores encrypted data.

Related publications by the maintainers:

- [Towards Practical Oblivious Map](https://dl.acm.org/doi/10.14778/3712221.3712235)

## ORAM

- [Path ORAM](https://dl.acm.org/doi/10.1145/2508859.2516660):
  [`path_oram.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/oram/path_oram.py)
- [Recursive Path ORAM](https://link.springer.com/chapter/10.1007/978-3-642-25385-0_11):
  [`recursive_path_oram.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/oram/recursive_path_oram.py)
- [Freecursive ORAM](https://people.csail.mit.edu/devadas/pubs/freecursive.pdf), insecure
  original: [`freecursive_oram.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/oram/freecursive_oram.py) with `reset_method="hard"`
- [Freecursive ORAM with probabilistic resets](https://eprint.iacr.org/2016/1084):
  [`freecursive_oram.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/oram/freecursive_oram.py) with `reset_method="prob"` (default)
- [DAORAM, de-amortized fixed resets](https://dl.acm.org/doi/10.14778/3712221.3712235):
  [`da_oram.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/oram/da_oram.py)
- Multi-path and static (PRF-positioned) Path ORAM variants:
  [`mul_path_oram.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/oram/mul_path_oram.py), [`static_oram.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/oram/static_oram.py)

## OMAP

- [AVL-tree OMAP](https://dl.acm.org/doi/10.1145/2660267.2660314) with the
  [streaming search](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf): [`avl_omap.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/omap/avl_omap.py)
- [B+-tree OMAP](https://people.eecs.berkeley.edu/~matei/papers/2020/vldb_oblidb.pdf):
  [`bplus_omap.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/omap/bplus_omap.py)
- [ORAM + search-tree framework](https://dl.acm.org/doi/10.14778/3712221.3712235), composing
  any ORAM with any tree OMAP: [`oram_ost_omap.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/omap/oram_ost_omap.py)
- Group-by-hash OMAP: [`group_omap.py`](https://github.com/WeiqiNs/OblivLib/blob/main/src/oblivlib/omap/group_omap.py)

Tree OMAPs hide the operation type by default (`distinguishable=False`); set it to `True` to let each
operation pad only to its own bound. Every search, insert, and delete pads to a fixed number of
rounds, `floor(a * h + b)` for a tree of height bound `h`. The defaults are the proven minimums; raise one
with `AvlOmapConfig(..., insert_budget=AccessBudget(3, 0))`, or use `AvlOmapConfig.textbook(...)` /
`BPlusOmapConfig.textbook(...)` for the budgets of the original held-node implementation. A B+ round reads
two paths by default (`paths_per_round=2`), which lowers insert and delete from `3h - 2` and `4h - 3` to
`2h - 1` rounds; `paths_per_round=1` keeps one path per round.
`AvlOmapConfig(..., cached=True)` lets an operation leave up to `max(k(h - 1), h)` evictions to the next ones (`h` for `k = 1`), which
lowers insert to `h + 1` and delete to `floor((3h - 1) / 2)` rounds; call `drain()` before shutdown or
handover to settle them in a fixed number of rounds.
`AvlOmapConfig(..., batch_size=k)` adds `search_batch` / `insert_batch` for up to `k` keys in the
same number of rounds as one key, each round reading up to `k` paths.

## Installation

```bash
pip install oblivlib    # or: uv add oblivlib
```

For development, clone the repository and run `pip install -e ".[dev]"`, which adds pytest, basedpyright,
and ruff.

## Usage

ORAM keys are int addresses in `[0, num_data)` and OMAP keys are `bytes` of at most `key_size`; every
value is `bytes` of at most `data_size`, so serialize your own data before storing it. A key or value
outside that contract raises `ContractError` before anything touches the server.

### Local and simulated runs

`Client.local()` keeps the storage server in the same process, the quickest way to try a scheme or run
experiments:

```python
from oblivlib.dependency import AesGcm, Client, PathOramConfig
from oblivlib.oram import PathOram

with Client.local() as client:
    oram = PathOram(PathOramConfig(num_data=1024, data_size=16, client=client, encryptor=AesGcm()))
    oram.init_server_storage()
    oram.operate_on_key(3, b"hello")
    assert oram.operate_on_key(3) == b"hello"
    print(client.metrics)
```

To estimate network cost without a real network, put `SimulatedNetwork` in front of an in-process server.
It models a round-trip time and a bandwidth for every request without sleeping:

```python
from oblivlib.dependency import Client, LoopbackTransport, SimulatedNetwork, StorageServer, TransportBackend

network = SimulatedNetwork(LoopbackTransport(StorageServer()), rtt_s=0.02, bytes_per_s=100e6)
with Client(TransportBackend(network)) as client:
    ...  # build and use schemes exactly as above
print(f"{network.request_count} requests, {network.modeled_seconds:.2f}s of modeled network time")
```

### Client and server

In a deployment the server only stores sealed rows and answers batched requests over ZeroMQ, while the
client holds the keys and runs the algorithm. On the server:

```python
from oblivlib.dependency import StorageServer, ZmqListener, serve

serve(ZmqListener("tcp://*:5555"), StorageServer(storage_dir="/var/lib/oblivlib"))
```

On the client only the connection changes; schemes are built and used as in the local example:

```python
from oblivlib.dependency import AesGcm, AvlOmapConfig, Client
from oblivlib.omap import AVLOmap

with Client.connect("tcp://server-host:5555") as client:
    omap = AVLOmap(AvlOmapConfig(num_data=1024, key_size=16, data_size=16, client=client, encryptor=AesGcm()))
    omap.init_server_storage()
    omap.insert(b"alice", b"42")
    assert omap.search(b"alice") == b"42"
```

Runnable versions live in [`demo/`](https://github.com/WeiqiNs/OblivLib/tree/main/demo): start `python demo/server.py`, then run `python demo/client.py`.
Every scheme gets a unique storage label by default, so several can share one client; pass `name=` in a
config only to address storage by a known label. The shared test suites
([`tests/oram/test_oram_common.py`](https://github.com/WeiqiNs/OblivLib/blob/main/tests/oram/test_oram_common.py),
[`tests/omap/test_omap_common.py`](https://github.com/WeiqiNs/OblivLib/blob/main/tests/omap/test_omap_common.py)) show every scheme in use.
