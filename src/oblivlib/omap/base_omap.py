from abc import ABC, abstractmethod
from typing import Any


class BaseOmap(ABC):
    @abstractmethod
    def init_server_storage(self, data: Any = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(self, key: Any, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, key: Any, value: Any = None) -> Any:
        raise NotImplementedError
