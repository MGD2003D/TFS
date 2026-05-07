from abc import ABC, abstractmethod
from typing import List, Dict, Optional


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

    async def extract_aspects(self, query: str) -> Optional[Dict]:
        """
        Анализирует сложность запроса и извлекает аспекты/хопы.
        Дефолтная реализация возвращает None (fallback to baseline).
        Переопределяется в клиентах с поддержкой aspect extraction.
        """
        return None

    async def extract_triplets(self, query: str, context: str) -> List[Dict]:
        """
        Извлекает триплеты (subject, predicate, object, confidence) из документов.
        Дефолтная реализация возвращает пустой список.
        Переопределяется в клиентах с поддержкой triplet extraction.
        """
        return []