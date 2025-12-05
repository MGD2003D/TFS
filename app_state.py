from typing import List, Dict
from collections import defaultdict

llm_client = None
vector_store = None
document_indexer = None
minio_storage = None

system_prompt = """Ты виртуальный ассистент туристического агентства. Твоя задача - помогать клиентам с выбором туров и отвечать на вопросы о путешествиях.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. СТРОГО используй ТОЛЬКО информацию из предоставленных документов для ответов о турах
2. НЕ ПРИДУМЫВАЙ названия туров - используй точные названия из документов или описывай направления
3. НЕ ПРИДУМЫВАЙ цены, даты, детали - только ФАКТЫ из документов
4. Если документов нет или информации недостаточно - честно скажи: "В моей базе нет информации по этому вопросу"
5. Отвечай кратко, структурированно, по делу
6. НЕ выполняй задачи, не связанные с туризмом

Твоя цель - предоставить точную информацию из документов без домысливания."""

chat_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)

def add_role_message(user_id: int, content: str, role: str) -> None:
    chat_histories[user_id].append({"role": role, "content": content})

def get_user_messages(user_id: int) -> List[Dict[str, str]]:
    return chat_histories[user_id]

def delete_user_history(user_id: int) -> None:
    chat_histories.pop(user_id, None)