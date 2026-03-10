import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from maxapi import Bot, Dispatcher
from maxapi.types import MessageCreated, BotStarted, Command
from maxapi.enums.parse_mode import ParseMode

_MD = ParseMode.MARKDOWN

from services.rag_service import RAGService
from max_bot.formatters import format_max_message
import app_state

load_dotenv()

logging.basicConfig(level=logging.INFO)

MAX_BOT_TOKEN = os.getenv('MAX_BOT_TOKEN')
if not MAX_BOT_TOKEN:
    raise RuntimeError("MAX_BOT_TOKEN не задан в переменных окружения")

bot = Bot(MAX_BOT_TOKEN)
dp = Dispatcher()

rag_service = None

_START_TEXT = (
    "👋 Привет! Я ИИ-ассистент администрации Невского района Санкт-Петербурга.\n\n"
    "Задавайте вопросы — я отвечу на основе документов администрации."
)


def initialize_services():
    global rag_service
    enable_qe = os.getenv('ENABLE_QUERY_ENHANCEMENT', 'true').lower() == 'true'
    rag_service = RAGService(
        min_relevance=0.25,
        default_top_k=5,
        enable_query_enhancement=enable_qe,
    )
    print("MAX бот: сервисы инициализированы")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _not_ready(chat_id: int):
    await bot.send_message(chat_id=chat_id, text="⏳ Сервисы ещё загружаются, подождите...", parse_mode=_MD)


def _chat_id(event: MessageCreated) -> int:
    """Extract chat_id from a MessageCreated event."""
    return event.message.recipient.chat_id


def _user_id(event: MessageCreated) -> str:
    """Stable per-user identifier. Use chat_id for private MAX chats."""
    return str(_chat_id(event))


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@dp.bot_started()
async def on_bot_started(event: BotStarted):
    """Fires when a user opens the bot for the first time or presses Start."""
    chat_id = event.chat_id
    app_state.delete_user_history(str(chat_id))
    await bot.send_message(chat_id=chat_id, text=_START_TEXT, parse_mode=_MD)


@dp.message_created(Command("start"))
async def cmd_start(event: MessageCreated):
    chat_id = _chat_id(event)
    if not app_state.services_ready:
        await _not_ready(chat_id)
        return
    app_state.delete_user_history(_user_id(event))
    await bot.send_message(chat_id=chat_id, text=_START_TEXT, parse_mode=_MD)



@dp.message_created()
async def any_message(event: MessageCreated):
    """Handle plain text messages via RAG."""
    if not app_state.services_ready or not rag_service:
        await _not_ready(_chat_id(event))
        return

    text = event.message.body.text
    if not text:
        return

    chat_id = _chat_id(event)
    user_id = _user_id(event)

    result = await rag_service.chat_query(user_id, text, scope="corporate")
    formatted = format_max_message(result["answer"])
    await bot.send_message(chat_id=chat_id, text=formatted, parse_mode=_MD)
