import app_state
from typing import List, Dict


class ChatService:

    async def simple_query(self, prompt: str) -> str:
        response = await app_state.llm_client.simple_query(prompt)
        return response

    async def chat_query(self, user_id: str, prompt: str) -> str:
        app_state.add_role_message(user_id, prompt, role="user")
        history = app_state.get_user_messages(user_id)
        response = await app_state.llm_client.chat_query(history)
        app_state.add_role_message(user_id, response, role="assistant")
        return response

    def clear_chat_history(self, user_id: str) -> None:
        app_state.delete_user_history(user_id)

    def get_chat_history(self, user_id: str) -> List[Dict[str, str]]:
        return app_state.get_user_messages(user_id)

    def get_all_chat_histories(self) -> Dict[str, List[Dict[str, str]]]:
        return app_state.chat_histories
