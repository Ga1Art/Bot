from abc import ABC, abstractmethod


class BaseCollector(ABC):
    source_name: str

    @abstractmethod
    def fetch(self) -> list[dict]:
        raise NotImplementedError
