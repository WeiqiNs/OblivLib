# OblivLib

[![codecov](https://codecov.io/gh/WeiqiNs/OblivLib/graph/badge.svg)](https://codecov.io/gh/WeiqiNs/OblivLib)

OblivLib (`oblivlib`) implements oblivious algorithms (oblivious RAM, oblivious maps and oblivious graphs) for
client/server deployments where the client is trusted and may compute non-obliviously, and the server only
stores encrypted data.

Related publications by the maintainers:

- [Towards Practical Oblivious Map](https://doi.org/10.14778/3712221.3712235)
- [Enabling Index-free Adjacency in Oblivious Graph Processing with Delayed Duplications](https://www.vldb.org/pvldb/vol19/p1826-cao.pdf)

## ORAM

- [Path ORAM](https://doi.org/10.1145/2508859.2516660): `oblivlib.oram.PathOram`
- [Recursive Path ORAM](https://doi.org/10.1007/978-3-642-25385-0_11): `oblivlib.oram.RecursivePathOram`
- [Freecursive ORAM](https://doi.org/10.1145/2694344.2694353): `oblivlib.oram.FreecursiveOram` with
  `reset_method="hard"` (insecure as published)
- Freecursive ORAM with [probabilistic resets](https://eprint.iacr.org/2016/1084):
  `oblivlib.oram.FreecursiveOram` with `reset_method="prob"` (default)
- [DAORAM, de-amortized fixed resets](https://doi.org/10.14778/3712221.3712235): `oblivlib.oram.DAOram`
- Multi-path and static (PRF-positioned) Path ORAM variants: `oblivlib.oram.MulPathOram`,
  `oblivlib.oram.StaticOram`

## OMAP

- [AVL-tree OMAP](https://doi.org/10.1145/2660267.2660314) with the
  [streaming search](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf): `oblivlib.omap.AVLOmap`
- [B+-tree OMAP](https://doi.org/10.14778/3364324.3364331): `oblivlib.omap.BPlusOmap`
- [ORAM + search-tree framework](https://doi.org/10.14778/3712221.3712235), composing any ORAM with any
  tree OMAP: `oblivlib.omap.OramOstOmap`
- Group-by-hash OMAP: `oblivlib.omap.GroupOmap`

Tree OMAPs hide the operation type by default (`distinguishable=False`); set it to `True` to let each
operation pad only to its own bound. Every search, insert, and delete pads to a fixed number of
rounds, `floor(a * h + b)` for a tree of height bound `h`. The defaults are the proven minimums; raise one
with `AvlOmapConfig(..., insert_budget=AccessBudget(3, 0))`. A B+ round reads
two paths by default (`paths_per_round=2`), which lowers insert and delete from `3h - 2` and `4h - 3` to
`2h - 1` rounds; `paths_per_round=1` keeps one path per round.
`cached=True` (AVL and B+; at two paths per round it brings single-key B+ operations to `h` rounds) lets an operation leave a bounded number of evictions (at most about `k * h`) to the next ones, which
lowers insert to `h + 1` and delete to `floor((3h - 1) / 2)` rounds; call `drain()` before shutdown or
handover to settle them in a fixed number of rounds.
`batch_size=k` (AVL and B+) adds `search_batch` / `insert_batch` for up to `k` keys in the
same number of rounds as one key, each round reading up to `k` paths.

## Graph

- [GraphOS](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf), over a batched AVL OMAP and a keyed value
  tree: `oblivlib.graph.GraphOS`
- [Grove, index-free adjacency with delayed duplications](https://www.vldb.org/pvldb/vol19/p1826-cao.pdf),
  over an AVL OMAP index, a graph tree and two meta trees: `oblivlib.graph.Grove`

Each scheme is built from its config in `oblivlib.dependency` (`PathOramConfig`, `AvlOmapConfig`,
`GraphOsConfig`, ...).

## Installation

OblivLib needs Python 3.12 or later and is not on PyPI yet; install it from GitHub:

```bash
pip install git+https://github.com/WeiqiNs/OblivLib    # or: uv add git+https://github.com/WeiqiNs/OblivLib
```

For development, clone the repository and run `pip install -e ".[dev]"`, which adds pytest, pytest-cov,
basedpyright and ruff. `pytest --cov` runs the suite and fails below 100% line and branch coverage.

## Usage

ORAM keys are int addresses in `[0, num_data)` and OMAP keys are `bytes` of at most `key_size`; every
value is `bytes` of at most `data_size`, so serialize your own data before storing it. Graph vertices are
int ids in `[0, 2**64)`, at most `num_data` of them, each with at most `max_degree` neighbors. A key or
value outside that contract raises `ContractError` before anything touches the server.

### Local and simulated runs

`Client.local()` keeps the storage server in the same process, the quickest way to try a scheme or run
experiments:

```python
from oblivlib.dependency import AesGcm, Client, PathOramConfig
from oblivlib.oram import PathOram

with Client.local() as client:
    oram = PathOram(PathOramConfig(num_data=1024, data_size=16, client=client, encryptor=AesGcm()))
    oram.init_server_storage()
    with client.measure() as measurement:
        oram.operate_on_key(3, b"hello")
        assert oram.operate_on_key(3) == b"hello"
        client.flush()
    print(measurement.result.rounds, measurement.elapsed_ns, measurement.client_ns)
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

### Metrics

Every `Client` keeps one cumulative account of its work, and `client.metrics` is a frozen `Metrics`
snapshot of it: rounds and requests, row bytes written and read, wire bytes sent and received, time spent
in the client's own message encoding, in the transport and on the server (`network_ns` is the transport
time the server did not account for), and the rows, bytes and time of bucket encoding, encryption,
decryption and decoding. Counters only grow; `with client.measure() as m:` gives the difference over a
span in `m.result`, plus its wall time `m.elapsed_ns` and the time none of the counters cover,
`m.client_ns` (scheme logic, stash work, PRFs and hashing).

- **One client, one account.** Every scheme on a client feeds it, so the parts of a composed scheme must
  share one client to be counted together.
- **A span's I/O is what crossed the wire during it.** A deferred write is sealed, and timed, in the
  operation that staged it, but its bytes count in the span that sends it. A span over a workload is exact
  up to one pending batch at each edge; call `client.flush()` at the edges for tight numbers.
  `measure()` never flushes, since that would change the round count.
- **In process, wire bytes are 0**; row bytes are the in-process measure.
- **Cipher counters include building the initial tree**, which with a `build_file` reads and writes
  sealed rows locally, so `opened_bytes == row_bytes_read` only holds for spans after hosting.

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

Graphs are used the same way:

```python
from oblivlib.dependency import AesGcm, Client, GroveConfig
from oblivlib.graph import Grove

with Client.local() as client:
    grove = Grove(GroveConfig(num_data=64, max_degree=4, data_size=16, client=client, encryptor=AesGcm()))
    grove.init_server_storage({0: [1], 1: [0, 2], 2: [1]}, {0: b"a", 1: b"b", 2: b"c"})
    assert grove.neighbor_values([1]) == {0: b"a", 2: b"c"}
    assert grove.hop_neighborhood(0, 2) == {0: b"a", 1: b"b", 2: b"c"}
```

Runnable versions live in [`demo/`](https://github.com/WeiqiNs/OblivLib/tree/main/demo): start `python demo/server.py`, then run `python demo/client.py`.
Every scheme gets a unique storage label by default, so several can share one client; pass `name=` in a
config only to address storage by a known label. The shared test suites
([`tests/oram/test_oram_common.py`](https://github.com/WeiqiNs/OblivLib/blob/main/tests/oram/test_oram_common.py),
[`tests/omap/test_omap_common.py`](https://github.com/WeiqiNs/OblivLib/blob/main/tests/omap/test_omap_common.py)) and the graph tests
([`tests/graph/`](https://github.com/WeiqiNs/OblivLib/tree/main/tests/graph)) show every scheme in use.

## License

Apache License 2.0; see [`LICENSE`](https://github.com/WeiqiNs/OblivLib/blob/main/LICENSE).
