# если нужно будет, можно разбить на хендлеры

import os
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import suppress

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.enums import ParseMode, ChatAction
from dotenv import load_dotenv
from services.rag_service import RAGService
from services.chat_service import ChatService
from tg_bot.formatters import format_telegram_message
from tg_bot import custom_emoji
import app_state
# from texts import MESSAGES

load_dotenv()

bot_token = os.getenv('BOT_TOKEN')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=bot_token)
dp = Dispatcher()

# Сервисы будут инициализированы позже (не при импорте модуля!)
rag_service = None
chat_service = None

async def initialize_custom_emoji():
    """Инициализация кастомных эмодзи из стикерпака"""
    try:
        await custom_emoji.load_custom_emoji_pack(bot_token)
    except Exception as e:
        print(f"[Custom Emoji] Ошибка при инициализации: {e}")

def initialize_services():
    """Инициализация сервисов - вызывается из main.py ПОСЛЕ инициализации app_state"""
    global rag_service, chat_service

    enable_query_enhancement = os.getenv('ENABLE_QUERY_ENHANCEMENT', 'true').lower() == 'true'

    rag_service = RAGService(
        min_relevance=0.25,
        default_top_k=5,
        enable_query_enhancement=enable_query_enhancement
    )
    chat_service = ChatService()

    asyncio.create_task(initialize_custom_emoji())

    print("Сервисы бота инициализированы")


async def _typing_indicator(chat_id: int) -> None:
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    if not app_state.services_ready or not chat_service:
        await message.answer("Сервисы еще загружаются, пожалуйста подождите...")
        return

    tg_id = message.from_user.id
    chat_service.clear_chat_history(tg_id)
    await message.answer("Привет! Я ИИ агент TFS!\n\nКоманды:\n/tours - показать все доступные туры\n/start - начать сначала")

@dp.message(Command("tours"))
async def cmd_tours(message: types.Message):
    if not app_state.services_ready or not rag_service:
        await message.answer("⏳ Сервисы еще загружаются, пожалуйста подождите...")
        return

    import time
    tg_id = message.from_user.id
    start_time = time.perf_counter()

    typing_task = asyncio.create_task(_typing_indicator(message.chat.id))
    try:
        result = await rag_service._handle_list_tours_intent(
            user_id=str(tg_id),
            query="Покажи все туры",
            enhancement_time=0.0,
            total_start=start_time
        )
    finally:
        typing_task.cancel()
        with suppress(asyncio.CancelledError):
            await typing_task

    formatted_answer = format_telegram_message(result["answer"])
    await message.answer(formatted_answer, parse_mode=ParseMode.HTML)

@dp.message(F.document)
async def upload_pdf(message: types.Message):
    pass

@dp.message()
async def any_message(message: types.Message):
    if not app_state.services_ready or not rag_service:
        await message.answer("Сервисы еще загружаются, пожалуйста подождите несколько секунд...")
        return

    tg_id = message.from_user.id

    typing_task = asyncio.create_task(_typing_indicator(message.chat.id))
    try:
        result = await rag_service.chat_query(tg_id, message.text)
    finally:
        typing_task.cancel()
        with suppress(asyncio.CancelledError):
            await typing_task

    formatted_answer = format_telegram_message(result["answer"])

    await message.answer(formatted_answer, parse_mode=ParseMode.HTML)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

