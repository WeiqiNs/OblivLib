# OblivLib

[![OblivLib CI](https://github.com/WeiqiNs/OblivLib/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WeiqiNs/OblivLib/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/WeiqiNs/OblivLib/graph/badge.svg)](https://codecov.io/gh/WeiqiNs/OblivLib)

OblivLib implements oblivious RAM, oblivious maps and oblivious graphs for a client/server setting: the
client holds the keys and runs the algorithm, and the server only stores encrypted data.

## Schemes

| Kind | Scheme | Class |
|---|---|---|
| ORAM | [Path ORAM](https://doi.org/10.1145/2508859.2516660) | `PathOram` |
| | [Recursive Path ORAM](https://doi.org/10.1007/978-3-642-25385-0_11) | `RecursivePathOram` |
| | [Freecursive ORAM](https://doi.org/10.1145/2694344.2694353), with [probabilistic resets](https://eprint.iacr.org/2016/1084) by default | `FreecursiveOram` |
| | [DAORAM](https://doi.org/10.14778/3712221.3712235) | `DAOram` |
| | Multi-path Path ORAM | `MulPathOram` |
| | Static Path ORAM (leaks repeated keys) | `StaticOram` |
| OMAP | [AVL-tree OMAP](https://doi.org/10.1145/2660267.2660314) | `AVLOmap` |
| | [B+-tree OMAP](https://doi.org/10.14778/3364324.3364331) | `BPlusOmap` |
| | [ORAM + search tree](https://doi.org/10.14778/3712221.3712235) | `OramOstOmap` |
| | Group-by-hash OMAP | `GroupOmap` |
| Graph | [GraphOS](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf) | `GraphOS` |
| | [Grove](https://www.vldb.org/pvldb/vol19/p1826-cao.pdf) | `Grove` |

## Installation

Requires Python 3.12 or later.

```bash
pip install oblivlib
```

For development: `pip install -e . --group dev`, then `pytest --cov` (the suite requires 100% line and
branch coverage).

## Usage

Everything is imported from `oblivlib`. Every scheme is built from its config (`PathOramConfig`,
`AvlOmapConfig`, `GroveConfig`, ...). A config sets the size (`num_data`, `data_size`, and `key_size` for maps), the
`client` that reaches the server, and optionally an `encryptor` and tuning options. Invalid settings raise
when the config is created.

```python
from oblivlib import AesGcm, AVLOmap, AvlOmapConfig, Client, PathOram, PathOramConfig

with Client.local() as client:
    oram = PathOram(PathOramConfig(num_data=1024, data_size=16, client=client, encryptor=AesGcm()))
    oram.init_server_storage()
    oram.operate_on_key(3, b"hello")
    assert oram.operate_on_key(3) == b"hello"

    omap = AVLOmap(AvlOmapConfig(num_data=1024, key_size=16, data_size=16, client=client, encryptor=AesGcm()))
    omap.init_server_storage()
    omap.insert(b"alice", b"42")
    assert omap.search(b"alice") == b"42"
```

Keys and values are bytes (ORAM keys are ints in `[0, num_data)`, graph vertices are ints); anything outside
a scheme's limits raises `ContractError` before the server is touched.

`Client.local()` runs the server in the same process. For a real deployment, run a server and connect to it.
The server has no authentication, so bind it to a trusted network or put it behind an authenticated tunnel:

```python
from oblivlib import StorageServer, ZmqListener, serve

serve(ZmqListener("tcp://*:5555"), StorageServer(storage_dir="/var/lib/oblivlib"))
```

```python
client = Client.connect("tcp://server-host:5555")
```

Runnable versions are in [`demo/`](demo): `python demo/server.py`, then `python demo/client.py`.

### Metrics

`client.metrics` counts rounds, bytes and time (server, network, encryption) across every scheme on that
client, and `with client.measure() as m:` gives the same numbers for one span of work in `m.result`.

## Citation

If OblivLib is useful in your work, please cite our papers:

```bibtex
@article{cao2024oblivious,
  title   = {Towards Practical Oblivious Map},
  author  = {Cao, Xinle and Feng, Weiqi and Liu, Jian and Zhou, Jinjin and Fang, Wenjing and Wang, Lei and
             Xu, Quanqing and Yang, Chuanhui and Ren, Kui},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {18},
  number  = {3},
  pages   = {688--701},
  year    = {2024},
  doi     = {10.14778/3712221.3712235}
}

@article{feng2026grove,
  title   = {Enabling Index-free Adjacency in Oblivious Graph Processing with Delayed Duplications},
  author  = {Feng, Weiqi and Cao, Xinle and O'Neill, Adam and Yang, Chuanhui},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {19},
  number  = {8},
  pages   = {1826--1839},
  year    = {2026},
  url     = {https://www.vldb.org/pvldb/vol19/p1826-cao.pdf}
}
```
