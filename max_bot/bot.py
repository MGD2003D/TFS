# MAX bot stub — mirrors tg_bot/bot.py structure
# TODO: replace placeholder classes with actual MAX SDK calls once maxapi is added to requirements.txt

import os
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import suppress

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from services.rag_service import RAGService
from services.chat_service import ChatService
import app_state

load_dotenv()

logging.basicConfig(level=logging.INFO)

bot_token = os.getenv('MAX_BOT_TOKEN')

rag_service = None
chat_service = None


def initialize_services():
    global rag_service, chat_service

    enable_query_enhancement = os.getenv('ENABLE_QUERY_ENHANCEMENT', 'true').lower() == 'true'

    rag_service = RAGService(
        min_relevance=0.25,
        default_top_k=5,
        enable_query_enhancement=enable_query_enhancement
    )
    chat_service = ChatService()
    print("MAX bot сервисы инициализированы")


# TODO: replace with actual MAX SDK Bot and Dispatcher
class _StubBot:
    async def delete_webhook(self):
        pass


class _StubDispatcher:
    async def start_polling(self, bot):
        print("[MAX BOT] Polling started (stub — replace with MAX SDK)")
        while True:
            await asyncio.sleep(3600)


bot = _StubBot()
dp = _StubDispatcher()
