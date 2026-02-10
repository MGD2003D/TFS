"""
Модуль для работы с кастомными эмодзи из Telegram стикерпака.

Загружает эмодзи из стикерпака при старте приложения и использует их в ответах бота.
Стикерпак настраивается через переменную окружения CUSTOM_EMOJI_PACK_NAME.

НО TG НЕ ДАЕТ БЕСПЛАТНО ПОЛЬЗОВАТЬСЯ ЭТИМ
"""

import os
from typing import Dict, Optional
import asyncio

# Глобальный словарь с загруженными эмодзи
CUSTOM_EMOJI: Dict[str, str] = {}

_initialized = False


async def load_custom_emoji_pack(bot_token: str, pack_name: Optional[str] = None) -> bool:

    global CUSTOM_EMOJI, _initialized

    if pack_name is None:
        pack_name = os.getenv('CUSTOM_EMOJI_PACK_NAME')

    if not pack_name:
        print("[Custom Emoji] CUSTOM_EMOJI_PACK_NAME не указан в .env, кастомные эмодзи не будут использоваться")
        _initialized = True
        return False

    try:
        import aiohttp

        print(f"[Custom Emoji] Загрузка стикерпака: {pack_name}")

        async with aiohttp.ClientSession() as session:
            # Получаем информацию о стикерпаке
            url = f"https://api.telegram.org/bot{bot_token}/getStickerSet"
            params = {"name": pack_name}

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    print(f"[Custom Emoji] ⚠ Не удалось загрузить стикерпак: HTTP {response.status}")
                    _initialized = True
                    return False

                data = await response.json()

                if not data.get('ok'):
                    print(f"[Custom Emoji] ⚠ Ошибка API: {data.get('description', 'Unknown error')}")
                    _initialized = True
                    return False

                sticker_set = data.get('result', {})
                stickers = sticker_set.get('stickers', [])

                for sticker in stickers:
                    if sticker.get('type') == 'custom_emoji':
                        custom_emoji_id = sticker.get('custom_emoji_id')
                        emoji_char = sticker.get('emoji', '')

                        if custom_emoji_id and emoji_char:
                            CUSTOM_EMOJI[emoji_char] = custom_emoji_id

                print(f"[Custom Emoji] Загружено {len(CUSTOM_EMOJI)} эмодзи из стикерпака '{pack_name}'")
                _initialized = True
                return True

    except Exception as e:
        print(f"[Custom Emoji] Ошибка при загрузке стикерпака: {e}")
        _initialized = True
        return False


def get_emoji(emoji_char: str, use_custom: bool = True) -> str:

    if use_custom and emoji_char in CUSTOM_EMOJI:
        emoji_id = CUSTOM_EMOJI[emoji_char]
        return f'<tg-emoji emoji-id="{emoji_id}">{emoji_char}</tg-emoji>'
    else:
        return emoji_char


def is_initialized() -> bool:
    """Проверяет, инициализированы ли кастомные эмодзи"""
    return _initialized


def plane() -> str:
    return get_emoji('✈️')

def world() -> str:
    return get_emoji('🌍')

def beach() -> str:
    return get_emoji('🏖')

def mountain() -> str:
    return get_emoji('⛰')

def sun() -> str:
    return get_emoji('☀️')

def star() -> str:
    return get_emoji('⭐')

def fire() -> str:
    return get_emoji('🔥')

def check() -> str:
    return get_emoji('✅')

def wave() -> str:
    return get_emoji('👋')

def sparkles() -> str:
    return get_emoji('✨')

def memo() -> str:
    return get_emoji('📋')
