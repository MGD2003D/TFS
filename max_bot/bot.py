import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from maxapi import Bot, Dispatcher, F
from maxapi.types import MessageCreated, BotStarted, Command
from maxapi.enums.parse_mode import ParseMode

_MD = ParseMode.MARKDOWN
import aiohttp

from services.rag_service import RAGService
from services.document_types import is_supported_document
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
user_search_preferences: dict = {}
user_upload_preferences: dict = {}

_START_TEXT = (
    "👋 Привет! Я ИИ-ассистент администрации Невского района Санкт-Петербурга.\n\n"
    "Задавайте вопросы — я отвечу на основе документов администрации.\n\n"
    "📚 Команды:\n"
    "/search_mode — режим поиска (personal / corporate / personal_corporate)\n"
    "/upload_mode — куда загружать документы (personal / corporate)\n"
    "/start — начать сначала\n\n"
    "Вы также можете прислать файл .pdf или .docx — я его проиндексирую."
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


@dp.message_created(Command("search_mode"))
async def cmd_search_mode(event: MessageCreated):
    chat_id = _chat_id(event)
    user_id = _user_id(event)
    if not app_state.services_ready:
        await _not_ready(chat_id)
        return

    text = event.message.body.text or ""
    args = text.split()[1:]
    valid = ["personal", "corporate", "personal_corporate"]

    if not args or args[0] not in valid:
        current = user_search_preferences.get(user_id, "personal_corporate")
        await bot.send_message(
            chat_id=chat_id,
            parse_mode=_MD,
            text=(
                f"🔍 Текущий режим поиска: **{current}**\n\n"
                "Использование: `/search_mode <режим>`\n\n"
                "• `personal` — только ваши документы\n"
                "• `corporate` — только корпоративные документы\n"
                "• `personal_corporate` — все (по умолчанию)"
            ),
        )
        return

    user_search_preferences[user_id] = args[0]
    names = {
        "personal": "Только личные документы",
        "corporate": "Только корпоративные документы",
        "personal_corporate": "Личные + корпоративные",
    }
    await bot.send_message(chat_id=chat_id, text=f"✅ Режим поиска: {names[args[0]]}", parse_mode=_MD)


@dp.message_created(Command("upload_mode"))
async def cmd_upload_mode(event: MessageCreated):
    chat_id = _chat_id(event)
    user_id = _user_id(event)
    if not app_state.services_ready:
        await _not_ready(chat_id)
        return

    text = event.message.body.text or ""
    args = text.split()[1:]
    valid = ["personal", "corporate"]

    if not args or args[0] not in valid:
        current = user_upload_preferences.get(user_id, "personal")
        await bot.send_message(
            chat_id=chat_id,
            parse_mode=_MD,
            text=(
                f"📤 Текущий режим загрузки: **{current}**\n\n"
                "Использование: `/upload_mode <режим>`\n\n"
                "• `personal` — в личное хранилище (по умолчанию)\n"
                "• `corporate` — в корпоративное хранилище"
            ),
        )
        return

    user_upload_preferences[user_id] = args[0]
    names = {"personal": "Личное хранилище", "corporate": "Корпоративное хранилище"}
    await bot.send_message(chat_id=chat_id, text=f"✅ Загрузка в: {names[args[0]]}", parse_mode=_MD)


@dp.message_created(F.message.body.attachments)
async def handle_attachment(event: MessageCreated):
    """Handle incoming file attachments (documents to index)."""
    if not app_state.services_ready or not rag_service:
        await _not_ready(_chat_id(event))
        return

    chat_id = _chat_id(event)
    user_id = _user_id(event)
    attachments = event.message.body.attachments or []

    processed_any = False
    for attachment in attachments:
        att_type = getattr(attachment, 'type', None)
        if att_type != 'file':
            continue

        payload = getattr(attachment, 'payload', None)
        if payload is None:
            continue

        url = getattr(payload, 'url', None)
        filename = getattr(payload, 'filename', None) or 'document'

        if not url:
            continue

        processed_any = True

        if not is_supported_document(filename):
            await bot.send_message(
                chat_id=chat_id,
                parse_mode=_MD,
                text=f"❌ Неподдерживаемый формат: `{filename}`\nПоддерживаются: .pdf, .docx",
            )
            continue

        await bot.send_message(chat_id=chat_id, text=f"📥 Загружаю документ: {filename}...", parse_mode=_MD)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=f"❌ Не удалось скачать файл (HTTP {resp.status}).",
                            parse_mode=_MD,
                        )
                        continue
                    content = await resp.read()

            upload_mode = user_upload_preferences.get(user_id, "personal")
            is_corporate = upload_mode == "corporate"

            class _MaxFile:
                def __init__(self, data: bytes, name: str):
                    self._data = data
                    self.filename = name

                async def read(self) -> bytes:
                    return self._data

            result = await rag_service.upload_and_index_document(
                _MaxFile(content, filename),
                user_id=user_id,
                is_corporate=is_corporate,
            )

            storage = "корпоративное" if is_corporate else "личное"
            await bot.send_message(
                chat_id=chat_id,
                parse_mode=_MD,
                text=(
                    f"✅ Документ загружен в {storage} хранилище!\n\n"
                    f"📄 Файл: {result['filename']}\n"
                    f"📊 Проиндексировано чанков: {result['chunks_indexed']}\n"
                    f"💾 Размер: {result['size'] / 1024:.1f} KB\n\n"
                    "Теперь можете задавать вопросы по этому документу!"
                ),
            )

        except ValueError as e:
            await bot.send_message(chat_id=chat_id, text=f"❌ Ошибка: {e}", parse_mode=_MD)
        except Exception as e:
            print(f"[MAX] Ошибка загрузки документа: {e}")
            import traceback
            traceback.print_exc()
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Не удалось загрузить документ. Попробуйте позже.",
                parse_mode=_MD,
            )

    if not processed_any:
        # Attachment exists but none was a supported file — ignore silently
        # (could be an image, sticker, etc.)
        pass


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
    scope = user_search_preferences.get(user_id, "personal_corporate")

    result = await rag_service.chat_query(user_id, text, scope=scope)
    formatted = format_max_message(result["answer"])
    await bot.send_message(chat_id=chat_id, text=formatted, parse_mode=_MD)
