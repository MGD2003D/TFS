# если нужно будет, можно разбить на хендлеры

import os
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import suppress

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.enums import ParseMode, ChatAction
from dotenv import load_dotenv
from services.rag_service import RAGService
from services.chat_service import ChatService
from tg_bot.formatters import format_telegram_message
from tg_bot import custom_emoji
import app_state
# from texts import MESSAGES

user_mode: dict = {}  # tg_id → "search" | "compose"

load_dotenv()

bot_token = os.getenv('BOT_TOKEN')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=bot_token)
dp = Dispatcher()

# Сервисы будут инициализированы позже (не при импорте модуля!)
rag_service = None
chat_service = None

user_search_preferences = {}

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

    await message.answer(
        "👋 Привет! Я ИИ-ассистент Жилищного комитета!\n\n"
        "📚 Команды:\n"
        "/search — поиск по документам (по умолчанию)\n"
        "/compose — составление ответного письма\n"
        "/search_mode — режим поиска (personal/corporate/personal_corporate)\n"
        "/start — начать сначала\n\n"
        "💡 Вы можете:\n"
        "• Задавать вопросы по загруженным документам\n"
        "• Вставлять обращение граждан для составления ответа"
    )

@dp.message(Command("compose"))
async def cmd_compose(message: types.Message):
    if not app_state.services_ready:
        await message.answer("⏳ Сервисы еще загружаются, подождите...")
        return
    tg_id = message.from_user.id
    user_mode[tg_id] = "compose"
    await message.answer(
        "✍️ Режим составления писем активирован.\n\n"
        "Вставьте текст входящего обращения — я подготовлю черновик официального ответа."
    )


@dp.message(Command("search"))
async def cmd_search(message: types.Message):
    if not app_state.services_ready:
        await message.answer("⏳ Сервисы еще загружаются, подождите...")
        return
    tg_id = message.from_user.id
    user_mode[tg_id] = "search"
    await message.answer(
        "🔍 Режим поиска активирован.\n\n"
        "Задайте вопрос по загруженным документам."
    )


@dp.message(Command("search_mode"))
async def cmd_search_mode(message: types.Message):

    if not app_state.services_ready:
        await message.answer("⏳ Сервисы еще загружаются, подождите...")
        return

    tg_id = message.from_user.id
    args = message.text.split()[1:] if len(message.text.split()) > 1 else []

    valid_modes = ["personal", "corporate", "personal_corporate"]

    if not args or args[0] not in valid_modes:
        current = user_search_preferences.get(tg_id, "personal_corporate")
        await message.answer(
            f"🔍 Текущий режим поиска: {current}\n\n"
            "Использование: /search_mode <personal|corporate|personal_corporate>\n\n"
            "• personal - только ваши документы 📄\n"
            "• corporate - только корпоративные документы 🏢\n"
            "• personal_corporate - все документы (по умолчанию) 📚"
        )
        return

    mode = args[0]
    user_search_preferences[tg_id] = mode

    mode_names = {
        "personal": "Только личные документы 📄",
        "corporate": "Только корпоративные документы 🏢",
        "personal_corporate": "Личные + корпоративные 📚"
    }
    await message.answer(f"✅ Режим поиска: {mode_names[mode]}")




@dp.message()
async def any_message(message: types.Message):
    if not app_state.services_ready or not rag_service:
        await message.answer("Сервисы еще загружаются, пожалуйста подождите несколько секунд...")
        return

    tg_id = message.from_user.id
    mode = user_mode.get(tg_id, "search")

    typing_task = asyncio.create_task(_typing_indicator(message.chat.id))
    try:
        if mode == "compose":
            if not app_state.letter_composer:
                await message.answer("❌ Сервис составления писем недоступен.")
                return
            result = await app_state.letter_composer.compose(message.text, str(tg_id))
            draft = result.get("draft", "")
            await message.answer(f"📝 Черновик ответа:\n\n{draft}")
            return
        else:
            scope = user_search_preferences.get(tg_id, "personal_corporate")
            result = await rag_service.chat_query(
                str(tg_id),
                message.text,
                scope=scope
            )
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

