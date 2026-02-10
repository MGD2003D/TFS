from typing import List, Dict
from collections import defaultdict
import time
import prompts_config

llm_client = None
vector_store = None
document_indexer = None
minio_storage = None
query_enhancer = None
tour_catalog = None

services_ready = False

system_prompt = prompts_config.SYSTEM_PROMPT

chat_histories: Dict[int | str, List[Dict[str, str]]] = defaultdict(list)
chat_last_activity: Dict[int | str, float] = {}

def add_role_message(user_id: int | str, content: str, role: str) -> None:
    chat_histories[user_id].append({"role": role, "content": content})
    if role == "user":
        chat_last_activity[user_id] = time.monotonic()

def get_user_messages(user_id: int | str) -> List[Dict[str, str]]:
    return chat_histories[user_id]

def delete_user_history(user_id: int | str) -> None:
    chat_histories.pop(user_id, None)
    chat_last_activity.pop(user_id, None)
