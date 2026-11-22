from abc import ABC, abstractmethod

KvList = list[tuple[bytes, bytes]]


class BaseOmap(ABC):
    @abstractmethod
    def init_server_storage(self, data: KvList | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(self, key: bytes, value: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, key: bytes, value: bytes | None = None) -> bytes | None:
        raise NotImplementedError
