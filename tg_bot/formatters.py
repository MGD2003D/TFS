import re


def format_telegram_message(text: str) -> str:
    """
    Форматирует ответ модели для Telegram:
    1. Извлекает <think> секцию и оборачивает её в blockquote
    2. Преобразует Markdown в HTML теги для Telegram
    """

    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)

    # think_section = ""
    main_content = text

    if think_match:
        # think_content = think_match.group(1).strip()
        # think_section = f"<blockquote>{think_content}</blockquote>\n\n"
        main_content = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()

    main_content = convert_markdown_to_html(main_content)

    # return think_section + main_content
    return main_content


def convert_markdown_to_html(text: str) -> str:

    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)

    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)

    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
