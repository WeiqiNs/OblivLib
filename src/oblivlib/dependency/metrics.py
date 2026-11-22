import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Self


@dataclass(frozen=True, slots=True, kw_only=True)
class Metrics:
    rounds: int = 0
    requests: int = 0
    row_bytes_written: int = 0
    row_bytes_read: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    client_wire_ns: int = 0
    transport_ns: int = 0
    server_ns: int = 0
    rows_sealed: int = 0
    rows_opened: int = 0
    sealed_bytes: int = 0
    opened_bytes: int = 0
    encode_ns: int = 0
    encrypt_ns: int = 0
    decrypt_ns: int = 0
    decode_ns: int = 0

    @property
    def network_ns(self) -> int:
        return max(0, self.transport_ns - self.server_ns)

    @property
    def crypto_ns(self) -> int:
        return self.encrypt_ns + self.decrypt_ns

    def __sub__(self, other: Self) -> Self:
        return type(self)(
            rounds=self.rounds - other.rounds,
            requests=self.requests - other.requests,
            row_bytes_written=self.row_bytes_written - other.row_bytes_written,
            row_bytes_read=self.row_bytes_read - other.row_bytes_read,
            bytes_sent=self.bytes_sent - other.bytes_sent,
            bytes_received=self.bytes_received - other.bytes_received,
            client_wire_ns=self.client_wire_ns - other.client_wire_ns,
            transport_ns=self.transport_ns - other.transport_ns,
            server_ns=self.server_ns - other.server_ns,
            rows_sealed=self.rows_sealed - other.rows_sealed,
            rows_opened=self.rows_opened - other.rows_opened,
            sealed_bytes=self.sealed_bytes - other.sealed_bytes,
            opened_bytes=self.opened_bytes - other.opened_bytes,
            encode_ns=self.encode_ns - other.encode_ns,
            encrypt_ns=self.encrypt_ns - other.encrypt_ns,
            decrypt_ns=self.decrypt_ns - other.decrypt_ns,
            decode_ns=self.decode_ns - other.decode_ns,
        )


class Recorder:
    def __init__(self) -> None:
        self.rows_sealed: int = 0
        self.rows_opened: int = 0
        self.sealed_bytes: int = 0
        self.opened_bytes: int = 0
        self.encode_ns: int = 0
        self.encrypt_ns: int = 0
        self.decrypt_ns: int = 0
        self.decode_ns: int = 0

    def record_seal(self, *, rows: int, sealed_bytes: int, encode_ns: int, encrypt_ns: int) -> None:
        self.rows_sealed += rows
        self.sealed_bytes += sealed_bytes
        self.encode_ns += encode_ns
        self.encrypt_ns += encrypt_ns

    def record_open(self, *, rows: int, opened_bytes: int, decrypt_ns: int, decode_ns: int) -> None:
        self.rows_opened += rows
        self.opened_bytes += opened_bytes
        self.decrypt_ns += decrypt_ns
        self.decode_ns += decode_ns


class Measurement:
    def __init__(self, snapshot: Callable[[], Metrics]) -> None:
        self._snapshot: Callable[[], Metrics] = snapshot
        self._before: Metrics = Metrics()
        self._start_ns: int = 0
        self._result: Metrics = Metrics()
        self._elapsed_ns: int = 0

    @property
    def result(self) -> Metrics:
        return self._result

    @property
    def elapsed_ns(self) -> int:
        return self._elapsed_ns

    @property
    def client_ns(self) -> int:
        spent = self._result
        accounted = (
            max(spent.transport_ns, spent.server_ns)
            + spent.client_wire_ns
            + spent.encode_ns
            + spent.decode_ns
            + spent.crypto_ns
        )
        return max(0, self._elapsed_ns - accounted)

    def __enter__(self) -> Self:
        self._before = self._snapshot()
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._elapsed_ns = time.perf_counter_ns() - self._start_ns
        self._result = self._snapshot() - self._before
