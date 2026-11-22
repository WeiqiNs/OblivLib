import threading

import msgspec
import pytest

from oblivlib.dependency import (
    Client,
    DuplicateLabelError,
    LoopbackTransport,
    Metrics,
    ProtocolError,
    RowSizeError,
    SimulatedNetwork,
    StorageServer,
    TransportBackend,
    TransportError,
    ZmqListener,
    ZmqTransport,
    serve,
)
from oblivlib.dependency.protocol import (
    PROTOCOL_VERSION,
    ErrorReply,
    Hello,
    Ok,
    Shutdown,
    Timed,
    decode_timed,
    encode,
    expect_reply,
)
from oblivlib.dependency.tree_builder import MemoryImage

ROW = 4


def _row(tag: int) -> bytes:
    return bytes([tag]) * ROW


def _workload(client: Client) -> Metrics:
    client.host_tree("t", MemoryImage(level=3, row_bytes=ROW, rows=[_row(i + 1) for i in range(7)]))
    with client.measure() as measurement:
        client.add_write_path("t", {0: _row(9), 2: _row(9), 6: _row(9)})
        client.execute()
        client.add_read_path("t", [0, 3])
        assert client.execute().require("t")[6] == _row(9)
    return measurement.result


def _serve_in_thread(listener: ZmqListener, server: StorageServer) -> threading.Thread:
    thread = threading.Thread(target=serve, args=(listener, server), daemon=True)
    thread.start()
    return thread


def _shutdown(endpoint: str, thread: threading.Thread) -> None:
    control = ZmqTransport(endpoint, timeout_ms=5000)
    expect_reply(decode_timed(control.request(encode(Shutdown()))).reply, Ok)
    control.close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_non_requests_get_timed_error_replies():
    transport = LoopbackTransport(StorageServer())
    for data, match in [
        (msgspec.msgpack.encode(["NoSuchMessage", "s"]), "malformed request.*NoSuchMessage"),
        (msgspec.msgpack.encode(Ok()), "malformed request.*'Ok'"),
        (encode(Hello(version=PROTOCOL_VERSION))[:-1], "malformed request"),
    ]:
        timed = decode_timed(transport.request(data))
        assert isinstance(timed.reply, ErrorReply) and timed.server_ns >= 0
        with pytest.raises(ProtocolError, match=match):
            expect_reply(timed.reply, Ok)


def test_transport_backend_splits_bytes_and_time_by_direction():
    backend = TransportBackend(LoopbackTransport(StorageServer()))
    hello = Hello(version=PROTOCOL_VERSION)
    reply = backend.call(hello)
    assert backend.bytes_sent == len(encode(hello))
    assert backend.bytes_received == len(encode(Timed(server_ns=backend.server_ns, reply=reply)))
    assert 0 <= backend.server_ns <= backend.transport_ns
    assert backend.client_wire_ns >= 0


def test_error_replies_raise_mapped_errors_over_zmq():
    listener = ZmqListener("tcp://127.0.0.1:*")
    thread = _serve_in_thread(listener, StorageServer())
    client = Client.connect(listener.endpoint, timeout_ms=5000)

    client.create_tree("t", level=2, row_bytes=ROW)
    with pytest.raises(DuplicateLabelError, match="'t'"):
        client.create_tree("t", level=2, row_bytes=ROW)

    client.add_write_path("t", {0: b"short", 2: _row(1)})
    client.execute()
    client.add_read_path("t", [1])
    with pytest.raises(RowSizeError):
        client.execute()

    client.add_write_path("t", {0: _row(5), 2: _row(6)})
    client.add_read_path("t", [1])
    assert client.execute().require("t") == {0: _row(5), 2: _row(6)}

    client.close()
    _shutdown(listener.endpoint, thread)
    listener.close()


def test_timeout_raises_transport_error():
    listener = ZmqListener("tcp://127.0.0.1:*")
    transport = ZmqTransport(listener.endpoint, timeout_ms=100)
    with pytest.raises(TransportError, match="within 100 ms"):
        transport.request(encode(Hello(version=1)))
    with pytest.raises(TransportError, match="failed"):
        transport.request(encode(Hello(version=1)))
    transport.close()
    listener.close()


def test_simulated_network_models_rtt_and_bandwidth():
    network = SimulatedNetwork(LoopbackTransport(StorageServer()), rtt_s=0.01, bytes_per_s=1000)
    backend = TransportBackend(network)
    with Client(backend) as client:
        _workload(client)
    assert network.request_count > 0
    assert network.modeled_seconds == pytest.approx(
        network.request_count * 0.01 + (backend.bytes_sent + backend.bytes_received) / 1000
    )


def test_metrics_match_between_local_and_loopback():
    local = _workload(Client.local())
    remote = _workload(Client(TransportBackend(LoopbackTransport(StorageServer()))))

    for metrics in (local, remote):
        assert (metrics.rounds, metrics.row_bytes_written, metrics.row_bytes_read) == (1, 3 * ROW, 5 * ROW)
    assert local.bytes_sent == local.bytes_received == local.transport_ns == 0
    assert remote.bytes_sent > remote.row_bytes_written and remote.bytes_received > remote.row_bytes_read
    assert 0 <= remote.server_ns <= remote.transport_ns
