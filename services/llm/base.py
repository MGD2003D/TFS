from abc import ABC

class BaseLLMClient(ABC):

    @classmethod
    async def init(self):
        pass

    @classmethod
    async def simple_query(self, prompt: str) -> str:
        pass

    @classmethod
    async def cleanup(self):
        pass