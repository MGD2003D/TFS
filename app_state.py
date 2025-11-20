from typing import List, Dict
from collections import defaultdict

llm_client = None

system_prompt = "Ты полезный ассистент"

chat_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)
#TODO: добавить ограничение по длине хранения для одного пользователя

def add_role_message(user_id: int, content: str, role: str) -> None:
    
    chat_histories[user_id].append({
        "role": role, "content": content
    })

    return None

def get_user_messages(user_id: int) -> List[Dict[str, str]]:
    
    return chat_histories[user_id]

def delete_user_history(user_id: int) -> None:

    chat_histories.pop(user_id)

    return None