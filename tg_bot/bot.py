# если нужно будет, можно разбить на хендлеры

import os
import sys
import asyncio
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from dotenv import load_dotenv
from services.rag_service import RAGService
from services.chat_service import ChatService
# from texts import MESSAGES

load_dotenv()

bot_token = os.getenv('BOT_TOKEN')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=bot_token)
dp = Dispatcher()

rag_service = RAGService(min_relevance=0.25, default_top_k=5)
chat_service = ChatService()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    tg_id = message.from_user.id

    chat_service.clear_chat_history(tg_id)

    await message.answer("Привет! Я ИИ агент TFS!")

@dp.message(F.document)
async def upload_pdf(message: types.Message):
    pass

@dp.message()
async def any_message(message: types.Message):
    tg_id = message.from_user.id

    result = await rag_service.chat_query(tg_id, message.text)

    await message.answer(result["answer"])

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

