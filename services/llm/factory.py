import os


def create_llm_client():
    model = os.getenv("CAILA_MODEL", "Qwen3-30B-A3B")
    api_url = f"https://caila.io/api/mlpgateway/account/just-ai/model/{model}/predict"
    from services.llm.caila_client import CailaClient
    return CailaClient(api_url=api_url)
