from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLMClient(ABC):

    @classmethod
    @abstractmethod
    async def initialize(self) -> None:
        pass

    @classmethod
    @abstractmethod
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        pass

    @classmethod
    @abstractmethod
    async def simple_query(self, prompt: str) -> str:
        pass

    @classmethod
    @abstractmethod
    async def chat_query(self, user_id: int, messages: List[Dict[str, str]]) -> str:
        pass

    @classmethod
    @abstractmethod
    async def cleanup(self) -> None:
        pass