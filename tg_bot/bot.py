# если нужно будет, можно разбить на хендлеры

import os
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import suppress
import io

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

user_search_preferences = {}
user_upload_preferences = {}

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
        "👋 Привет! Я ИИ-ассистент TFS!\n\n"
        "📚 Команды:\n"
        "/tours - показать все туры\n"
        "/search_mode - режим поиска (personal/corporate/personal_corporate)\n"
        "/upload_mode - режим загрузки документов (personal/corporate)\n"
        "/start - начать сначала\n\n"
        "💡 Вы можете:\n"
        "• Задавать вопросы о турах\n"
        "• Загружать документы (.pdf, .docx)\n"
        "• Переключать хранилища для поиска"
    )

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


@dp.message(Command("upload_mode"))
async def cmd_upload_mode(message: types.Message):

    if not app_state.services_ready:
        await message.answer("⏳ Сервисы еще загружаются, подождите...")
        return

    tg_id = message.from_user.id
    args = message.text.split()[1:] if len(message.text.split()) > 1 else []

    valid_modes = ["personal", "corporate"]

    if not args or args[0] not in valid_modes:
        current = user_upload_preferences.get(tg_id, "personal")
        await message.answer(
            f"📤 Текущий режим загрузки: {current}\n\n"
            "Использование: /upload_mode <personal|corporate>\n\n"
            "• personal - в ваше личное хранилище (по умолчанию) 📄\n"
            "• corporate - в корпоративное хранилище (доступно всем) 🏢"
        )
        return

    mode = args[0]
    user_upload_preferences[tg_id] = mode

    mode_names = {
        "personal": "Личное хранилище 📄",
        "corporate": "Корпоративное хранилище 🏢"
    }
    await message.answer(f"✅ Документы будут загружаться в: {mode_names[mode]}")

@dp.message(F.document)
async def upload_document(message: types.Message):

    if not app_state.services_ready or not rag_service:
        await message.answer("⏳ Сервисы еще загружаются, подождите...")
        return

    tg_id = message.from_user.id
    document = message.document

    filename = document.file_name or "document"
    from services.document_types import is_supported_document
    if not is_supported_document(filename):
        await message.answer(
            "❌ Неподдерживаемый формат файла.\n"
            "Поддерживаются только: .pdf, .docx"
        )
        return

    status_msg = await message.answer(f"📥 Загружаю документ: {filename}...")

    try:
        file_info = await bot.get_file(document.file_id)
        file_data = await bot.download_file(file_info.file_path)

        upload_mode = user_upload_preferences.get(tg_id, "personal")
        is_corporate = (upload_mode == "corporate")

        class TelegramFile:
            def __init__(self, data, filename):
                self.file = io.BytesIO(data)
                self.filename = filename

            async def read(self):
                return self.file.read()

        file_obj = TelegramFile(file_data.read(), filename)

        result = await rag_service.upload_and_index_document(
            file_obj,
            user_id=str(tg_id),
            is_corporate=is_corporate
        )

        storage_type = "корпоративное 🏢" if is_corporate else "личное 📄"
        await status_msg.edit_text(
            f"✅ Документ загружен в {storage_type} хранилище!\n\n"
            f"📄 Файл: {result['filename']}\n"
            f"📊 Проиндексировано чанков: {result['chunks_indexed']}\n"
            f"💾 Размер: {result['size'] / 1024:.1f} KB\n\n"
            f"Теперь можете задавать вопросы по этому документу!"
        )

    except ValueError as e:
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        import traceback
        traceback.print_exc()
        await status_msg.edit_text(
            "❌ Не удалось загрузить документ. Попробуйте позже."
        )

@dp.message()
async def any_message(message: types.Message):
    if not app_state.services_ready or not rag_service:
        await message.answer("Сервисы еще загружаются, пожалуйста подождите несколько секунд...")
        return

    tg_id = message.from_user.id

    scope = user_search_preferences.get(tg_id, "personal_corporate")

    typing_task = asyncio.create_task(_typing_indicator(message.chat.id))
    try:
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

