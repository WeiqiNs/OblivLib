from dataclasses import replace

import pytest

from oblivlib.dependency import Client, Metrics, MissingResultError, ProtocolError, RowSizeError, UnknownLabelError
from oblivlib.dependency import client as client_module
from oblivlib.dependency.protocol import WriteRange
from oblivlib.dependency.tree_builder import MemoryImage

ROW = 4


def _row(tag: int) -> bytes:
    return bytes([tag]) * ROW


def _tree(client: Client, label: str = "t", level: int = 3) -> None:
    client.host_tree(
        label, MemoryImage(level=level, row_bytes=ROW, rows=[_row(i + 1) for i in range((1 << level) - 1)])
    )


def test_local_client_hosts_rows_and_reads_root_first():
    with Client.local() as client:
        _tree(client)
        client.add_read_path("t", [3, 0])
        result = client.execute()
        rows = result.require("t")
        assert list(rows) == [0, 1, 2, 3, 6]
        assert list(rows.values()) == [_row(1), _row(2), _row(3), _row(4), _row(7)]
        with pytest.raises(MissingResultError, match="'other'"):
            result.require("other")


def test_measurements_count_what_crossed_the_wire_during_them():
    with Client.local() as client:
        _tree(client)
        with client.measure() as whole:
            with client.measure() as staged:
                client.add_write_path("t", {0: _row(9), 1: _row(9), 3: _row(9)})
                client.execute()
            with client.measure() as sent:
                client.add_read_path("t", [0])
                client.execute()
            client.add_read_path("t", [1])
            client.execute()

        assert staged.result == Metrics()
        assert replace(sent.result, server_ns=0) == Metrics(
            rounds=1, requests=1, row_bytes_written=3 * ROW, row_bytes_read=3 * ROW
        )
        assert replace(whole.result, server_ns=0) == Metrics(
            rounds=2, requests=2, row_bytes_written=3 * ROW, row_bytes_read=6 * ROW
        )
        assert 0 <= sent.result.server_ns <= whole.result.server_ns
        assert 0 <= whole.client_ns <= whole.elapsed_ns
        assert replace(client.metrics, server_ns=0) == Metrics(
            rounds=2, requests=5, row_bytes_written=10 * ROW, row_bytes_read=6 * ROW
        )


def test_staging_a_read_for_an_unknown_label_raises():
    with Client.local() as client:
        _tree(client)
        with pytest.raises(UnknownLabelError, match="'missing'"):
            client.add_read_path("missing", [0])


def test_writes_apply_before_reads_and_later_write_wins():
    with Client.local(defer_writes=False) as client:
        _tree(client)
        client.add_write_path("t", {0: _row(10), 1: _row(10), 3: _row(10)})
        client.add_write_path("t", {0: _row(20), 1: _row(20), 4: _row(20)})
        client.add_read_path("t", [0, 1])
        rows = client.execute().require("t")
        assert rows == {0: _row(20), 1: _row(20), 3: _row(10), 4: _row(20)}


def test_deferred_writes_ride_next_execute(recorded_client):
    client, recorder = recorded_client()
    _tree(client)
    client.add_write_path("t", {0: _row(9), 1: _row(9), 3: _row(9)})
    assert client.execute().results == {}
    client.add_read_path("t", [0])
    assert client.execute().require("t")[3] == _row(9)
    assert recorder.batches[-1].writes


def test_empty_execute_is_a_round():
    with Client.local() as client:
        assert client.execute().results == {}
        assert client.metrics.rounds == 1


def test_deferred_write_error_surfaces_on_next_call():
    with Client.local() as client:
        _tree(client, level=2)
        client.add_write_path("t", {0: b"short", 1: _row(9)})
        assert client.execute().results == {}

        client.add_read_path("t", [0])
        with pytest.raises(RowSizeError):
            client.execute()

        client.add_read_path("t", [0])
        assert client.execute().require("t") == {0: _row(1), 1: _row(2)}


def test_host_tree_streams_byte_bounded_chunks(recorded_client, monkeypatch):
    monkeypatch.setattr(client_module, "WRITE_RANGE_CHUNK_BYTES", 2048)
    rows = [bytes([i + 1]) * 100 for i in range(100)]
    client, recorder = recorded_client()
    client.host_tree("t", MemoryImage(level=7, row_bytes=100, rows=rows))

    chunks = [message for message in recorder.messages if isinstance(message, WriteRange)]
    assert [(chunk.start, len(chunk.rows)) for chunk in chunks] == [(0, 20), (20, 20), (40, 20), (60, 20), (80, 20)]
    client.add_read_path("t", range(64))
    stored = client.execute().require("t")
    assert [stored[i] for i in range(100)] == rows
    assert stored[100] == b""


def test_flush_sends_deferred_writes_and_close_ends_the_session(recorded_client):
    unused, unused_recorder = recorded_client()
    unused.close()
    assert unused_recorder.messages == []

    client, recorder = recorded_client()
    with client:
        _tree(client)
        client.flush()
        assert client.metrics.rounds == 0

        client.add_write_path("t", {0: _row(9), 1: _row(9), 3: _row(9)})
        client.execute()
        client.flush()
        assert client.metrics.rounds == 1

        client.add_write_path("t", {0: _row(8), 1: _row(8), 4: _row(8)})
        client.execute()

    assert recorder.batches[-1].writes and client.metrics.rounds == 2
    client.close()
    with pytest.raises(ProtocolError, match="unknown session"):
        client.execute()
