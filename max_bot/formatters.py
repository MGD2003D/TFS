import re


def format_max_message(text: str) -> str:
    """
    Форматирует ответ модели для MAX messenger:
    - Удаляет <think> секцию
    - Конвертирует стандартный markdown в MAX-совместимый markdown
      (MAX поддерживает **bold**, _italic_, но не ### заголовки и не - списки)
    """
    # Strip <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Convert ### Heading / ## Heading / # Heading → **Heading**
    text = re.sub(r'^#{1,6}\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)

    # Convert "- item" or "* item" bullet points → "• item"
    text = re.sub(r'^[ \t]*[-*]\s+', '• ', text, flags=re.MULTILINE)

    # Convert "  - item" (indented bullets) → "  • item"
    # (already handled by the above since we strip leading spaces before -)

    # Collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
