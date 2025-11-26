from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorStore(ABC):

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass
